import os
import time
import copy
import logging
import pprint
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
import matplotlib.pyplot as plt
import torchvision
import wandb
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from configs.config import config, update_config
from networks import get_eval_pool, get_network
from eval import eval_synthetic_set
from utils import exp_utils, train_utils, data_utils, loss_utils
from models.ema import ImageEMA, ModelEMA, FeatEMA
from pretrain import pretrain_encoder

logger = logging.getLogger(__name__)

def compute_loss(sample_real, sample_syn, H_syn, label, model, model_ema, losses, rand_idx, perceptual_loss):
    sample_syn = model.decode(H_syn.unsqueeze(1))[:,0]
    H_real = model.encode(sample_real, rand_idx)
    H_syn = model.encode(sample_syn, rand_idx)

    permute = torch.randperm(H_real.shape[0])
    H_diff =  H_real - H_syn
    H_real_ = H_diff[permute] + H_real

    Hs = torch.stack([H_real, H_real_, H_syn], dim=1)
    sample_pred = model.decode(Hs)
    sample_diff = sample_real.unsqueeze(1) - sample_pred[:, 0:2]
    
    classifier_criterion = nn.CrossEntropyLoss()
    logits = model.linear_cls(H_real)
    
    losses['recon'] = torch.mean((sample_diff ** 2))
    losses['recon_H'] = torch.mean((H_diff ** 2))
    losses['cls'] = classifier_criterion(logits, label)

    return sample_pred[:,0], sample_pred[:,1], sample_pred[:, 2], H_syn

def train_epoch(config, loader, dataset, image_syn, feat_syn, model, model_ema, perceptual_loss, optimizer, epoch, output_dir, device, rank):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()

    adjust_iter_num = config.train.adjust_iter_num

    batch_end = time.time()
    batch_size = config.train.batch_size
    img_mean = torch.tensor(config.dataset.mean).reshape(1, -1, 1, 1).to(device)
    img_std = torch.tensor(config.dataset.std).reshape(1, -1, 1, 1).to(device)
    for batch_idx, batch in enumerate(loader):
        # adjust lr
        iter_num = batch_idx + len(loader) * epoch
        if iter_num in adjust_iter_num:
            train_utils.adjust_lr(config, optimizer, iter_num, adjust_iter_num)
        
        # sampling synthetic sample or feature
        ipc = config.dataset.ipc
        batch_size = config.train.batch_size
        rand_idx = torch.randint(0, ipc, (1,)).item()

        sample_real, label = batch[0].to(device), batch[1].to(device)
        sample_syn = image_syn.val[label, rand_idx]
        H_syn = feat_syn.val[label, rand_idx]

        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        # compute reconstruction & classification loss
        losses = {}
        out_real, out_real2, out_syn, out_feat_syn = compute_loss(sample_real, sample_syn, H_syn, label, model, model_ema, losses, rand_idx, perceptual_loss)
        time_meters.add_loss_value('Reconstruction time', time.time() - end)
        end = time.time()
    
        total_loss = losses['recon'] + losses['recon_H'] + losses['cls']
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model_ema.update(model)
        image_syn.update(out_syn.detach(), label, rand_idx)
        feat_syn.update(out_feat_syn.detach(), label, rand_idx)

        for k, v in losses.items():
            if v is not None:
                loss_meters.add_loss_value(k, v)
        time_meters.add_loss_value('Batch time', time.time() - batch_end)

        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, recon {recon_time:.3f}s, all {batch_time:.3f}s, Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].val,
                recon_time=time_meters.average_meters['Reconstruction time'].val,
                batch_time=time_meters.average_meters['Batch time'].val
            )
            for k, v in loss_meters.average_meters.items():
                wandb.log({"Train Loss/{0}".format(k): loss_meters.average_meters[k].val}, step = iter_num)
                
                tmp = '{0}: {loss.val:.6f} ({loss.avg:.6f}),'.format(
                        k, loss=loss_meters.average_meters[k]
                )
                msg += tmp
            msg = msg[:-2]

            logger.info(msg)
            if config.vis_recon:
                img_ = torch.cat([sample_real[:32], sample_syn[:32], out_real[:32], out_real2[:32], out_syn[:32]], dim=0)
                img_ = torchvision.utils.make_grid((img_ * img_std + img_mean).clamp(0, 1), 32).permute(1, 2, 0)
                wandb.log({"GT_and_Pred_Images": wandb.Image(img_.cpu().numpy())}, step=iter_num)
                plt.axis('off')
                plt.imshow(img_.cpu().numpy())
                plt.savefig(output_dir+f'/images/out_{epoch}.png')

        batch_end = time.time()

def main():
    # Get args and config
    args = parse_args()
    logger, output_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # set device
    gpus = range(torch.cuda.device_count())
    device = torch.device('cuda') if len(gpus) > 0 else torch.device('cpu')
    use_ddp = len(gpus) > 1
    if use_ddp:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
    
    # get dataset and dataloader
    config.dataset.split = 'train'
    train_data = exp_utils.load_component(config.dataset, config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if use_ddp else None
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.train.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=True,
                                               sampler=train_sampler)
    
    config.dataset.split = 'test'
    test_data = exp_utils.load_component(config.dataset, config)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.eval.batch_size, shuffle=True, num_workers=0)

    # get basic config for model definition
    ipc = config.dataset.ipc
    img_size = config.dataset.img_size
    img_mean = torch.tensor(config.dataset.mean).reshape(1, -1, 1, 1).to(device)
    img_std = torch.tensor(config.dataset.std).reshape(1, -1, 1, 1).to(device)
    num_classes = config.dataset.num_classes
    img_dim = config.dataset.num_channel
    feat_dim = config.model.feat_dim

    # get encoders and a decoder
    encoders = []
    for idx, encoder_name in enumerate(config.model.encoders):
        encoder = get_network(config, encoder_name, img_dim, num_classes, (img_size, img_size), device, use_ddp, args.local_rank)
        encoders.append(encoder)
    decoder = get_network(config, config.model.decoder, feat_dim, num_classes, (img_size, img_size), device, use_ddp, args.local_rank)

    # or can get pre-trained encoder, but not tested or working for now.
    if config.use_pretrained:
        for idx, (encoder, encoder_name) in enumerate(zip(encoders, config.model.encoders)):
            ckpt_name = f"{config.dataset.name}_{encoder_name}.pth.tar"
            if os.path.exists("checkpoints/"+ ckpt_name):
                train_utils.load_checkpoint(encoder, "./checkpoints", ckpt_name=ckpt_name, device=device, use_ddp=use_ddp)
            else:
                _, model_state = pretrain_encoder(config, idx, encoder, train_loader, test_loader, device)
                train_utils.save_checkpoint(model_state, "./checkpoints", ckpt_name=ckpt_name)

            encoders.append(encoder)
    
    # get a model composed of encs/dec and prototype image/feature
    model = exp_utils.load_component(config.model, config, encoders = encoders, decoder = decoder).to(device)
    model_ema = ModelEMA(model, 0.999)
    image_syn = ImageEMA(num_classes, ipc, img_dim, img_size, device)
    feat_syn = FeatEMA(num_classes, ipc, feat_dim, device)
    label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device)

    # get models for evaluation
    eval_model_pool = get_eval_pool(config.eval.mode)
    params = list(model.parameters()) + list(encoders[0].parameters()) 

    # get optimizer
    optimizer = torch.optim.Adam(params, lr=config.train.lr)
    perceptual_loss = loss_utils.VGGPerceptualLoss().to(device)
    if use_ddp:
        model =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        device_num = len(device_ids)
    
    start_ep = 0
    end_ep = int(config.train.total_iteration / len(train_loader)) + 1

    # train
    for epoch in range(start_ep, end_ep):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        train_epoch(config, 
                    loader=train_loader,
                    dataset=train_data, 
                    image_syn=image_syn,
                    feat_syn=feat_syn,
                    model=model, 
                    model_ema=model_ema,
                    perceptual_loss = perceptual_loss,
                    optimizer=optimizer,
                    epoch=epoch, 
                    output_dir=output_dir,
                    device=device,
                    rank=args.local_rank)
        
        # Qualitative validation of image_syn
        iter_num = (epoch + 1) * len(train_loader)
        if config.vis_image_syn:
            img_ = (image_syn.val.flatten(0,1).detach() * img_std + img_mean).clamp(0, 1)
            img_ = torchvision.utils.make_grid(img_, num_classes).permute(1, 2, 0)
            wandb.log({"Synthetic_Images": wandb.Image(img_.cpu().numpy())}, step=iter_num)
            plt.axis('off')
            plt.imshow(img_.cpu().numpy())
            plt.savefig(output_dir+f'/images/proto_{epoch}.png')
        
        # Quantitative validataion of image_syn
        time_meters = exp_utils.AverageMeters()
        eval_data = data_utils.TensorDataset(image_syn.val, label_syn)
        eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=config.eval.batch_size, shuffle=True, num_workers=0)

        train_end = time.time()
        accs_all = []
        for eval_model in eval_model_pool:
            accs = []
            for eval_iter in range(config.eval.num):
                eval_net = get_network(config, eval_model, img_dim, num_classes, (img_size, img_size), device, use_ddp, args.local_rank)
                _, acc_train, acc_test = eval_synthetic_set(config, eval_iter, eval_net, eval_loader, test_loader, device)
                accs.append(acc_test)
            time_meters.add_loss_value(f'Eval time', time.time() - train_end)
            wandb.log({"Eval Acc/{0}".format(eval_model): np.mean(accs)}, step = iter_num)
            msg = 'Evaluate {0} random {1}, acc mean = {acc_mean:.4f}, std = {acc_std:.4f}, eval_time = {eval_time:.4f}'.format(
                len(accs), 
                eval_model, 
                acc_mean = np.mean(accs), 
                acc_std = np.std(accs),
                eval_time = time_meters.average_meters['Eval time'].val,
            )
            logger.info(msg)
            accs_all.append(accs)
        
        wandb.log({"Eval Acc/all model": np.mean(accs_all)}, step = iter_num)
        msg = 'Evaluate all model acc mean = {acc_mean:.4f}, std = {acc_std:.4f}'.format(
            acc_mean = np.mean(accs_all), 
            acc_std = np.std(accs_all),
        )
        logger.info(msg)

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Distillation with Invariant Constraint')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument(
        '--debug', default=False, action='store_true', help='flag to debug small issues')
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args

if __name__ == '__main__':
    main()