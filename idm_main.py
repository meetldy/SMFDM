import pdb
import time
from audioop import avg
from concurrent.futures import process
from dis import dis
import torch
from torch import optim

import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import json
import pickle
import numpy as np
import torch.distributed as dist
from sparselearning.core import add_sparse_args,CosineDecay,Masking
from sparselearning.extensions import magnitude_variance_pruning, variance_redistribution
import utils
import random
#设置随机种子，保证验证集的可重复性
def set_random_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs.
    np.random.seed(seed)
    random.seed(seed)  # Set the seed for Python's built-in random module.

    # For additional reproducibility, you can set these flags:
    # This may slow down your training, so use it only for debugging purposes.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
some_fixed_number = 250
set_random_seed(some_fixed_number)
def main(args):
    utils.init_distributed_mode(args)
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and opt['phase'] != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase, num_tasks, global_rank)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase, num_tasks, global_rank)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # print('1111111111111111')
    # for name, param in diffusion.netG.module.named_parameters():
    #     if not param.requires_grad:
    #         print(name)
    # print('----------')

    decay = args.death_rate_decay


    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
       # optimizer1 =diffusion.return_optimizer()
       #  for param in diffusion.netG.parameters():
       #      print(param)
       #  exit()
        optimizer1 = torch.optim.SGD(diffusion.netG.parameters(), lr=args.lr, momentum=args.momentum)

        if args.sparse:
            decay=CosineDecay(args.death_rate,len(train_loader)*args.epochs)
            mask=Masking(optimizer1,prune_rate_decay=decay)
            # mask.add_module(diffusion.netG,density=args.density)
            mask.add_module(diffusion.netG.module if hasattr(diffusion.netG, 'module') else diffusion.netG,density=args.density)
            # diffusion.set_mask(mask)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer1, args.decay_frequency, gamma=0.1)
        max_psnr = -1e18
        if args.flops:
            sample_data = {'inp': torch.randn(1, 3, 16, 16),'gt': torch.randn(1, 3, 128, 128)}
            diffusion.feed_data(sample_data)
            diffusion.calculate_flops()
            exit()
        while current_step < n_iter:
            current_epoch += 1
            scaler = torch.cuda.amp.GradScaler()

            for _, train_data in enumerate(train_loader):
                # if lr_scheduler is not None: lr_scheduler.step()
                # if diffusion.lr_scheduler is not None: diffusion.lr_scheduler.step()
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)

                diffusion.optimize_parameters(scaler)
                # print(optimizer1.state)
                # optimizer1.step()
                # for name, para in diffusion.netG.named_parameters():
                #     param_state = optimizer1.state[para]
                #     if 'momentum_buffer' not in param_state:
                #         print(name)
                # exit()
                # if mask is not None:
                #     mask.step()
                # else:
                #     optimizer1.step()
                # if not args.dense and args.sparse and current_epoch < args.epochs:
                #     mask.at_end_of_epoch()

                # log
                if dist.get_rank() == 0:
                    if current_step % opt['train']['print_freq'] == 0:
                        logs = diffusion.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                            current_epoch, current_step)
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                            tb_logger.add_scalar(k, v, current_step)
                        logger.info(message)

                        if wandb_logger:
                            wandb_logger.log_metrics(logs)


                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0

                    result_hr_path = opt['path']['results_hr'].rsplit('/', 1)[0] + '/{}/'.format(current_step) + opt['path']['results_hr'].rsplit('/', 1)[1]
                    result_sr_path = opt['path']['results_sr'].rsplit('/', 1)[0] + '/{}/'.format(current_step) + opt['path']['results_sr'].rsplit('/', 1)[1]
                    result_lr_path = opt['path']['results_lr'].rsplit('/', 1)[0] + '/{}/'.format(current_step) + opt['path']['results_lr'].rsplit('/', 1)[1]


                    os.makedirs('{}'.format(result_hr_path), exist_ok=True)
                    os.makedirs('{}'.format(result_sr_path), exist_ok=True)
                    os.makedirs('{}'.format(result_lr_path), exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_hr_path, idx, dist.get_rank()))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_sr_path, idx, dist.get_rank()))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_lr_path, idx, dist.get_rank()))

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((sr_img, hr_img), axis=1)
                            )

                    avg_psnr = torch.Tensor([avg_psnr]).to(dist.get_rank())
                    dist.reduce(avg_psnr, 0)
                    dist.barrier()

                    avg_psnr = avg_psnr.item() / (idx * dist.get_world_size())
                    # avg_psnr = avg_psnr.item() / idx
                    if avg_psnr >= max_psnr and dist.get_rank() == 0:
                        max_psnr = avg_psnr
                        diffusion.save_network(current_epoch, current_step, best='psnr_{}'.format(max_psnr))
                        # 保存mask到与模型相同或相关的目录下
                        # if mask is not None:
                        #     mask_filename = f"netG_mask_psnr_{max_psnr:.4f}_epoch_{current_epoch}_step_{current_step}.pth"
                        #     mask_save_path = os.path.join(opt['path']['models'], mask_filename)
                        #     torch.save(mask.masks, mask_save_path)
                        #     logger.info(
                        #         f"Mask for netG saved at epoch {current_epoch}, step {current_step} to {mask_save_path}")

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    if dist.get_rank() == 0:

                        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                        logger_val = logging.getLogger('val')  # validation logger
                        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                            current_epoch, current_step, avg_psnr))
                        # tensorboard logger
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                    if current_step % opt['train']['save_checkpoint_freq'] == 0 and dist.get_rank() == 0:
                        logger.info('Saving models and training states.')
                        diffusion.save_network(current_epoch, current_step)


                        if wandb_logger and opt['log_wandb_ckpt']:
                            wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})
            if args.sparse:
                if mask is not None:
                    mask.step()
                else:
                    optimizer1.step()
                if not args.dense and args.sparse and current_epoch < args.epochs:
                    mask.at_end_of_epoch()
            else:
                pass
            # exit()
        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        start_time = time.time()
        result_hr_path = '{}'.format(opt['path']['results_hr'])
        result_sr_path = '{}'.format(opt['path']['results_sr'])
        result_lr_path = '{}'.format(opt['path']['results_lr'])
        process_path = '{}'.format(opt['path']['process'])
        os.makedirs(result_hr_path, exist_ok=True)
        os.makedirs(result_sr_path, exist_ok=True)
        os.makedirs(result_lr_path, exist_ok=True)
        os.makedirs(process_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            # diffusion.calculate_flops()
            diffusion.test(crop=False, continous=True, use_ddim=opt['use_ddim'])
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8


            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_sr_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(process_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_sr_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_hr_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_lr_path, current_step, idx))


            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        end_time = time.time()  # 添加：记录验证结束的总时间
        total_time = end_time - start_time  # 计算总的验证时间
        avg_time_per_image = total_time / idx

        logger.info('# Validation Time # Average time per image: {:.4f} seconds'.format(avg_time_per_image))  # 添加到日志中

        # ...（保留原有的日志记录和wandb日志记录逻辑）...
        logger.info('# Validation # PSNR: {:.4f}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4f}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4f}, ssim: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim),
                'AvgTimePerImage': avg_time_per_image,  # 添加到wandb日志中
            })


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--death_rate_decay', type=float, default=0.99, help='Decay rate for the death rate.')
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode.')
    parser.add_argument('--flops', action='store_true', help='Calculate the flops.')
    parser.add_argument('-c', '--config', type=str, default='config/ffhq_liifsr3_scaler_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--local_rank', type=int,help='local rank for dist')
    parser.add_argument('-r', '--resume', type=str, default='experiments/ffhq_scaler/checkpoint/latest')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-use_ddim', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--epoch', type=int, default=100, metavar='N',
    #                     help='number of epochs to train (default: 100)')

    parser.add_argument('--prune_rate_decay', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=610, metavar='N',
                        help='number of epochs to train (default: 610)')
    parser.add_argument('--death_rate', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='adam momentum (default: 0.9)')
    # parser.add_argument('--prune', type=str, default="magnitude_variance")
    # parser.add_argument('--redistribution', type=str, default="variance")
    # parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--lr', type=float, default=10e-6)
    parser.add_argument('--l2', type=float, default=1e-6)
    add_sparse_args(parser)
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    main(args)
