import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
from typing import Union


def process_time_steps(time_ls, target_device):
    """
    遍历time_ls中的每个元素，确保每个张量在target_device上。
    此处未涉及具体数字映射逻辑，因为关注点在于处理和设备转移。

    Args:
        time_ls (list): 包含张量的列表。
        target_device (str): 目标设备名称，如 'cuda:0'。
    Returns:
        list: 处理后的张量列表，每个张量位于指定设备上。
    """
    processed_list = []
    for time in time_ls:
        # 如果是列表，遍历子列表
        if isinstance(time, list):
            sub_list = [t.to(target_device) for t in time]
            processed_list.extend(sub_list)
        # 如果是单个张量，直接转移到目标设备
        else:
            processed_time = time.to(target_device)
            processed_list.append(processed_time)
    return processed_list


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    # for i in range(timesteps):
    #     setattr(self.unet, 'order', i)
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def register_time(unet, t):
    setattr(unet, 'order', t)


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            encoder,
            imnet,
            denoise_fn,
            image_size,
            channels=3,
            loss_type='l1',
            conditional=True,
            feat_unfold=False,
            local_ensemble=False,
            cell_decode=False,
            schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.encoder = encoder
        self.imnet = imnet
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        self.mod = 'custom'
        self.timesteps = torch.arange(0, 2000)

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('ddim_c1', torch.sqrt(
            to_torch((1. - alphas_cumprod / alphas_cumprod_prev) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, scaler, clip_denoised: bool, curr_noise=None, condition_x=None):
        batch_size = x.shape[0]
        # print('curr_noise',curr_noise.shape)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=curr_noise)
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=curr_noise)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, scaler, clip_denoised=True, curr_noise=None, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, scaler=scaler, clip_denoised=clip_denoised, curr_noise=curr_noise, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()
    @torch.no_grad()
    def get_snr(self):
        snr = (self.sqrt_alphas_cumprod / np.sqrt(1.0 - self.alphas_cumprod)) ** 2
    @torch.no_grad()
    def generalized_steps(self, x_in, conditional_input=None, continous=False):
        device = self.betas.device
        skip = self.num_timesteps // 200
        seq = range(0, self.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        sample_inter = (1 | (self.num_timesteps // 1000))
        x, scaler = x_in['inp'], x_in['scaler']
        b = x.size(0)

        shape = x.shape
        gt_shape = list(x_in['gt'].shape)
        img = torch.randn(gt_shape, device=device)
        x_feat = self.gen_feat(x, gt_shape[2:])
        conditional_input = x_feat
        # conditional_input = F.interpolate(x_in['inp'], x_in['gt'].shape[2:])

        ret_img = img
        for i, j in tzip(reversed(seq), reversed(seq_next)):
            if i == 0:
                break
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_cumprod_prev[i + 1]]).repeat(b, 1).to(x.device)
            t = (torch.ones(b) * i).to(x.device)
            next_t = (torch.ones(b) * j).to(x.device)
            at = compute_alpha(self.betas, t.long())
            at_next = compute_alpha(self.betas, next_t.long())
            xt = ret_img[-1]
            et = self.denoise_fn(torch.cat([conditional_input, xt.unsqueeze(0)], dim=1), conditional_input, scaler,
                                 noise_level)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = (0 * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(img) + c2 * et
            # if i % sample_inter == 0:
            ret_img = torch.cat([ret_img, xt_next], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def register_parallel(self, device, mod=5):
        time_ls = []
        all_steps = self.num_timesteps - 1
        timestep = torch.arange(0, self.num_timesteps, device=device)
        if mod == '50ls':
            cond = lambda timestep: timestep in [0, 2, 3, 5, 10, 15, 25, 35, 36]
        elif isinstance(mod, int):
            cond = lambda timestep: timestep % mod == 0
        else:
            raise Exception("Currently not supported, But you can modify the code to customize the keytime")
        curr_step = 0
        total_ls = []
        while curr_step < all_steps:
            # print(curr_step)
            time_ls = [timestep[curr_step]]
            curr_step += 1
            while not cond(curr_step):
                if curr_step < all_steps:
                    time_ls.append(timestep[curr_step])
                    curr_step += 1
                else:
                    break
            total_ls.append(time_ls)
        return total_ls

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, mod=10):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        time_ls = self.register_parallel(device=device, mod=mod)
        noise_levels = []
        for time in range(self.num_timesteps):
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[time + 1]])
            noise_levels.append(noise_level)
        noise_levels = torch.cat(noise_levels, dim=0)
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            latents = img  # new added
            init_latents = latents.detach().clone()  # new added
            for i in tqdm(range(0, len(time_ls)), desc='sampling loop time step', total=len(time_ls)):
                register_time(self.denoise_fn, time_ls[i][0])
                time_ls[i] = [2000 - v - 1 for v in time_ls[i]]
                processed_time_ls = process_time_steps(time_ls[i], device)
                noise_pred = self.denoise_fn(latents, processed_time_ls, noise_levels=noise_levels, mode='denoise')

                bs = noise_pred.shape[0]
                bs_perstep = bs // len(processed_time_ls)
                denoised_latent = latents  # new added
                for i, timestep in enumerate(processed_time_ls):
                    if timestep/2000<0.5:
                        denoised_latent = denoised_latent + 0.003 * init_latents
                    curr_noise = noise_pred[i * bs_perstep:(i + 1) * bs_perstep]
                    img = self.p_sample(img, processed_time_ls)
                    denoised_latent = img

                ret_img = torch.cat([ret_img, img], dim=0)
                latents = denoised_latent
        else:
            # image_index=1
            x, scaler = x_in['inp'], x_in['scaler']
            shape = x.shape
            gt_shape = list(x_in['gt'].shape)
            img = torch.randn(gt_shape, device=device)
            x_feat = self.gen_feat(x, gt_shape[2:])
            x_con = x_feat
            ret_img = x_con
            latents=torch.cat([x_con, img], dim=1)
            init_latents = latents.detach().clone()
            denoised_latent = latents
            for i in tqdm(range(0, len(time_ls)), desc='sampling loop time step', total=len(time_ls)):
                register_time(self.denoise_fn, time_ls[i][0])
                time_ls[i] = [self.num_timesteps - v - 1 for v in time_ls[i]]
                # print(' time_ls[i]', time_ls[i])
                processed_time_ls = process_time_steps(time_ls[i], device)
                # noise_pred = self.denoise_fn(latents, lr=x_con, scaler=scaler,
                #                              timestep=processed_time_ls, noise_levels=noise_levels, mode='denoise')
                latent_model_input=latents
                noise_pred = self.denoise_fn(latent_model_input, lr=x_con, scaler=scaler,
                                             timestep=processed_time_ls, noise_levels=noise_levels, mode='denoise')
                # print(torch.cat([x_con, img], dim=1).shape)
                # print('noise_pred orgin',noise_pred.shape)
                bs = noise_pred.shape[0]
                bs_perstep = bs // len(processed_time_ls)
                # denoised_latent = latents#new added
                for i, timestep in enumerate(processed_time_ls):
                    if timestep / 1000 < 0.5:
                        denoised_latent = denoised_latent + 0.003 * init_latents
                    curr_noise = noise_pred[i * bs_perstep:(i + 1) * bs_perstep]
                    img = self.p_sample(img, timestep, scaler, curr_noise=curr_noise, condition_x=x_con)
                    denoised_latent= torch.cat([x_con, img], dim=1)

                latents = denoised_latent

                    # denoised_latent=img
                ret_img = torch.cat([ret_img, img], dim=0)
                # latents = denoised_latent
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, use_ddim=False):
        if not use_ddim:
            # print("x_in",len(x_in))
            return self.p_sample_loop(x_in, continous)
        else:
            return self.generalized_steps(x_in, conditional_input=None, continous=continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def gen_feat(self, inp, shape):
        feat = self.encoder(inp, shape)
        return feat

    def p_losses(self, x_in, noise=None):
        inp, _, _, scaler = x_in['inp'], x_in['coord'], x_in['cell'], x_in['scaler']
        x_feat = self.gen_feat(inp, x_in['gt'].shape[2:])

        x_con = x_feat

        x_start = x_in['gt']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod, mode='diffusion')
        else:
            x_recon = self.denoise_fn(torch.cat([x_con, x_noisy], dim=1), lr=x_con, scaler=scaler,
                                      timestep=continuous_sqrt_alpha_cumprod, mode='diffusion')
        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)