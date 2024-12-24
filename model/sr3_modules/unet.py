import copy
from copy import deepcopy
import math
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from data.util import *
import torch.distributed as dist
from einops import rearrange
from .style import EqualLinear, StyleLayer, StyleLayer_norm_scale_shift

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def warpped_feature(sample, step):
    """
    sample: batch_size*dim*h*w, uncond: 0 - batch_size//2, cond: batch_size//2 - batch_size
    step: timestep span
    """
    sample = sample.repeat(step, 1, 1, 1)  # (step * bs) * dim * h *w
    return sample


def warpped_skip_feature(block_samples, step):
    down_block_res_samples = []
    for sample in block_samples:
        sample_expand = warpped_feature(sample, step)
        down_block_res_samples.append(sample_expand)
    return down_block_res_samples


def warpped_text_emb(text_emb, step):
    """
    text_emb: batch_size*77*768, uncond: 0 - batch_size//2, cond: batch_size//2 - batch_size
    step: timestep span
    """
    bs, token_len, dim = text_emb.shape
    uncond_fea, cond_fea = text_emb.chunk(2)
    uncond_fea = uncond_fea.repeat(step, 1, 1)  # (step * bs//2) * 77 *768
    cond_fea = cond_fea.repeat(step, 1, 1)  # (step * bs//2) * 77 * 768
    return torch.cat([uncond_fea, cond_fea])  # (step*bs) * 77 *768


def warpped_timestep(timesteps, bs):
    if bs == 1:
        # 直接返回原始时间步列表并转换为Tensor，然后根据需求重复
        return torch.tensor(timesteps).repeat(1)
    else:
        semi_bs = bs // 2
        ts = []
        for timestep in timesteps:
            timestep = timestep[None]
            texp = timestep.expand(semi_bs)
            ts.append(texp)
        timesteps = torch.cat(ts)
        return timesteps.repeat(2, 1).reshape(-1)


# def compute_noise(x, timestep):
#     batch_size = x.shape[0]
#     timesteps = torch.arange(1, 2001).to(x.device)
#     betas = torch.tensor(np.linspace(1e-6, 1e-2, 2000), dtype=torch.float).to(x.device)
#     alphas = 1. - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)
#     noise_stds = torch.sqrt(1. - alphas_cumprod)
# print(noise_stds[timestep])
# return noise_stds[timestep]
# def compute_noise(x,timestep):
#     batch_size=x.shape[0]
#     betas=np.linspace(1e-6,1e-2,2000)
#     alphas_=np.cumprod((1-betas),axis=0)
#     noise=np.sqrt(np.append(1.,alphas_))
#     noise_tensor = torch.FloatTensor(noise).repeat(batch_size, 1).to(x.device)
#     return noise_tensor[timestep+1]

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, shape):
        return self.conv(F.interpolate(x, shape))


class idm(nn.Module):
    def __init__(self, dim, feat_unfold=False,
                 local_ensemble=False,
                 cell_decode=False):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        self.style = StyleLayer_norm_scale_shift(
            dim,
            dim,
            kernel_size=3,
            num_style_feat=512,
            demodulate=True,
            sample_mode=None,
            resample_kernel=(1, 3, 3, 1))
        # if self.cell_decode:
        #     self.kannet = nn.Sequential(KAN([dim + 2 + 2,256,dim]))
        # else:
        #     self.kannet = nn.Sequential(KAN([dim + 2,256,dim]))
        if self.cell_decode:
            self.imnet = nn.Sequential(nn.Linear(dim + 2 + 2, 256), nn.Linear(256, dim))
        else:
            self.imnet = nn.Sequential(nn.Linear(dim + 2, 256), nn.Linear(256, dim))

    def forward(self, x, shape, scale1, scale2, shift):
        coord = make_coord(shape).repeat(x.shape[0], 1, 1).to(dist.get_rank())
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / shape[-2]
        cell[:, 1] *= 2 / shape[-1]
        return self.query_rgb(x, scale1, scale2, shift, coord, cell)

    def query_rgb(self, x_feat, scale1, scale2, shift, coord, cell=None):

        feat = self.style(x_feat, noise=None, scale1=scale1, scale2=scale2, shift=shift)#这句代码是得到hi
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(dist.get_rank()) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                # print(rel_coord)
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                # pred = self.kannet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0];
            areas[0] = areas[3];
            areas[3] = t
            t = areas[1];
            areas[1] = areas[2];
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        # b * pixels * features +coord -> b * pixels *rgb

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if (self.with_attn):
            x = self.attn(x)
        return x


def register_time(UNet, t):
    setattr(UNet, 'order', t)


class UNet(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
            image_size=128,
    ):
        super().__init__()
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        self.conv_body_first = StyleLayer(3, pre_channel, 3, bias=True, activate=True)
        self.conv_body_down = nn.ModuleList()
        self.condition_scale1 = nn.ModuleList()
        self.condition_scale2 = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        self.skip_feature = []

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            # iss_last = (ind >= num_mults)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            # if not iss_last:
            self.conv_body_down.append(StyleLayer(pre_channel, channel_mult, 3, downsample=True))
            self.condition_scale1.append(
                EqualLinear(1, channel_mult, bias=True, bias_init_val=1, activation=None))

            self.condition_scale2.append(
                EqualLinear(1, channel_mult, bias=True, bias_init_val=1, activation=None))

            self.condition_shift.append(
                StyleLayer(pre_channel, channel_mult, 3, bias=True, activate=False))
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        # self.conv_body_down
        self.downs = nn.ModuleList(downs)
        self.final_down1 = StyleLayer(512, 512, 3, downsample=False)
        self.final_down2 = StyleLayer(512, 256, 3, downsample=True)
        self.num_latent, self.num_style_feat = 4, 512
        self.final_linear = EqualLinear(2 * 2 * 256, self.num_style_feat * self.num_latent, bias=True,
                                        activation='fused_lrelu')
        self.final_styleconv = StyleLayer(512, 512, 3)
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            # is_first = (ind == 3)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(idm(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        # self.timesteps=torch.arange(0, 2000, dtype=torch.long)

    def forward(self, x, lr=None, scaler=None, timestep=None, noise_levels=None, mod=10, mode='diffusion'):
        if mode == 'denoise':
            noise_levels=noise_levels.to(x.device)
            order = getattr(self, 'order').cpu()
            # print("order",order)
            if isinstance(timestep, list):
                timesteps = timestep[0]
                step = len(timestep)
            else:
                timesteps = timestep
                step = 1
            if not torch.is_tensor(timesteps) and (not isinstance(timesteps, list)):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = x.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=x.device)
            elif (not isinstance(timesteps, list)) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(x.device)

            if (not isinstance(timesteps, list)) and len(timesteps.shape) == 1:
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(x.shape[0])
                # print("come",timesteps)
            elif isinstance(timesteps, list):
                # timesteps list, such as [981,961,941]
                timesteps = warpped_timestep(timesteps, x.shape[0]).to(x.device)
            # print("timesteps",timesteps)
            time = noise_levels[timesteps].to(x.device)
            # time = torch.tensor([time], dtype=torch.float32).to(x.device)
            # print("time before",time)

            t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None
            # print("t",t.shape)
            feat = self.conv_body_first(lr)#f0
            scales1, scales2, shifts = [], [], []
            scale1 = self.condition_scale1[0](scaler)
            scales1.append(scale1.clone())
            scale2 = self.condition_scale2[0](scaler)
            scales2.append(scale2.clone())
            shift = self.condition_shift[0](feat)#f0
            shifts.append(shift.clone())
            j = 1
            for i in range(len(self.conv_body_down)):
                # print('1111111111111111111')
                # print(self.conv_body_down[i])
                feat = self.conv_body_down[i](feat)
                if j < len(self.condition_scale1):
                    scale1 = self.condition_scale1[j](scaler)
                    scales1.append(scale1.clone())
                    scale2 = self.condition_scale2[j](scaler)
                    scales2.append(scale2.clone())
                    shift = self.condition_shift[j](feat)
                    shifts.append(shift.clone())
                    j += 1
            # exit()
            feats = []
            # ===============
            ipow = int(np.sqrt(9 + 8 * order))
            cond = order in [0, 1, 2, 3, 5, 10, 15, 25, 35]
            if isinstance(mod, int):
                cond = order % mod == 0
            elif mod == "pro":
                cond = ipow * ipow == (9 + 8 * order)
            elif mod == "50ls":
                cond = order in [0, 1, 2, 3, 5, 10, 15, 25,
                                 35]  # 40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
            elif mod == "50ls2":
                cond = order in [0, 10, 11, 12, 15, 20, 25, 30, 35,
                                 45]  # 40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
            elif mod == "50ls3":
                cond = order in [0, 20, 25, 30, 35, 45, 46, 47, 48,
                                 49]  # 40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
            elif mod == "50ls4":
                cond = order in [0, 9, 13, 14, 15, 28, 29, 32, 36,
                                 45]  # 40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
            elif mod == "100ls":
                cond = order > 85 or order < 10 or order % 5 == 0
            elif mod == "75ls":
                cond = order > 65 or order < 10 or order % 5 == 0
            elif mod == "s2":
                cond = order < 20 or order > 40 or order % 2 == 0

            if cond:
                for layer in self.downs:
                    if isinstance(layer, ResnetBlocWithAttn):
                        x = layer(x, t)
                    else:
                        x = layer(x)
                    feats.append(x)
                setattr(self, 'skip_feature', deepcopy(feats))

                for layer in self.mid:
                    if isinstance(layer, ResnetBlocWithAttn):
                        x = layer(x, t)
                    else:
                        x = layer(x)
                setattr(self, 'feature', x.detach().clone())

                if isinstance(timestep, list):
                    # timesteps list, such as [981,961,941]
                    # print(timestep)
                    timesteps = warpped_timestep(timestep, x.shape[0]).to(x.device)
                    # print(timesteps)
                    # exit()
                    time = noise_levels[timesteps].to(x.device)  # print(time)
                    
                    t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

                feats = warpped_skip_feature(feats, step)

                x = warpped_feature(x, step)

            else:

                feats = copy.deepcopy(getattr(self, 'skip_feature'))
                feat = getattr(self, 'feature')
                feats = warpped_skip_feature(feats, step)
                x = warpped_feature(feat, step)

            for i, layer in enumerate(self.ups):
                if isinstance(layer, ResnetBlocWithAttn):
                    # print(t.shape)
                    # exit()
                    x = layer(torch.cat((x, feats.pop()), dim=1), t)
                else:
                    x = layer(x, feats[-1].shape[2:], scales1.pop(), scales2.pop(), shifts.pop())
                    x = rearrange(x, 'b (h w) c -> b c h w', h=feats[-1].shape[-1])
                # print(x.shape)
                # exit()
            return self.final_conv(x)
        elif mode == 'diffusion':
            # time=noise_levels[timestep].to(x.device)
            time = timestep
            t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None
            feat = self.conv_body_first(lr)
            scales1, scales2, shifts = [], [], []
            scale1 = self.condition_scale1[0](scaler)
            scales1.append(scale1.clone())
            scale2 = self.condition_scale2[0](scaler)
            scales2.append(scale2.clone())
            shift = self.condition_shift[0](feat)
            shifts.append(shift.clone())
            j = 1
            for i in range(len(self.conv_body_down)):
                feat = self.conv_body_down[i](feat)
                if j < len(self.condition_scale1):
                    scale1 = self.condition_scale1[j](scaler)
                    scales1.append(scale1.clone())
                    scale2 = self.condition_scale2[j](scaler)
                    scales2.append(scale2.clone())
                    shift = self.condition_shift[j](feat)
                    shifts.append(shift.clone())
                    j += 1

            feats = []
            for layer in self.downs:
                if isinstance(layer, ResnetBlocWithAttn):
                    x = layer(x, t)
                else:
                    x = layer(x)
                feats.append(x)

            for layer in self.mid:
                if isinstance(layer, ResnetBlocWithAttn):
                    x = layer(x, t)
                else:
                    x = layer(x)

            for i, layer in enumerate(self.ups):
                if isinstance(layer, ResnetBlocWithAttn):
                    x = layer(torch.cat((x, feats.pop()), dim=1), t)
                else:
                    x = layer(x, feats[-1].shape[2:], scales1.pop(), scales2.pop(), shifts.pop())#缩放因子
                    x = rearrange(x, 'b (h w) c -> b c h w', h=feats[-1].shape[-1])

            return self.final_conv(x)
        else:
            print("The mode is error")