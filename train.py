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

from configs.config import config, update_config
from models.eval_networks import get_eval_pool, get_network
from eval import eval_synthetic_set
from utils import exp_utils, train_utils, data_utils, loss_utils
from models.ema import Feat1DEMA, ImageEMA, ModelEMA
from pretrain import pretrain_encoder
from models.base_networks import ResNetEncoder, Conv1d1x1Encoder

logger = logging.getLogger(__name__)

def compute_loss(sample_real, feat_syn, label, model, model_ema, losses, rand_idx, perceptual_loss):
    H_real, logits = model.encode(sample_real, feat_syn, rand_idx)
    H_syn = feat_syn[label]

    H_diff = H_real - H_syn
    permute = torch.randperm(H_diff.shape[0])
    H_real_ = H_real + H_diff[permute]

    Hs = torch.stack([H_real, H_real_, H_syn], dim=1)
    sample_pred = model.decode(Hs)

    sample_diff = sample_real.unsqueeze(1) - sample_pred[:, 0:1]
    losses['recon'] = torch.mean(sample_diff ** 2) 
    losses['recon_H'] = torch.mean(H_diff ** 2) 
    #losses['recon'] = torch.mean(torch.sum(sample_diff ** 2, [1, 2, 3]))
    #losses['recon_H'] = torch.mean(torch.sum(H_diff ** 2, [1, 2]))
    classifier_criterion = nn.CrossEntropyLoss()
    
    #logits = logits.flatten(0, 1)     
    #label = torch.cat([label, label], 0)
    #losses['perception'] = perceptual_loss(sample_pred[:,0], sample_pred[:,1])
    losses['cls'] = classifier_criterion(logits, label)

    sample_syn = sample_pred[:, 2]
    H_syn_update, _ = model.encode(sample_syn, feat_syn, rand_idx)
    return sample_pred[:,0], sample_pred[:,1], sample_pred[:, 2], H_syn_update

def train_epoch(config, loader, dataset, image_syn, feat_syn, model, model_ema, perceptual_loss, optimizer, epoch, output_dir, device, rank):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()

    adjust_iter_num = config.train.adjust_iter_num

    batch_end = time.time()
    batch_size = config.train.batch_size
    img_mean = torch.tensor(config.dataset.mean).reshape(1, 1, -1).to(device)
    img_std = torch.tensor(config.dataset.std).reshape(1, 1, -1).to(device)
    for batch_idx, batch in enumerate(loader):
        # adjust lr
        iter_num = batch_idx + len(loader) * epoch
        if iter_num in adjust_iter_num:
            train_utils.adjust_lr(config, optimizer, iter_num, adjust_iter_num)
        
        # sampling synthetic sample
        ipc = config.dataset.ipc
        batch_size = config.train.batch_size
        rand_idx = torch.randint(0, ipc, (1,)).item()

        sample_real, label = batch[0].to(device), batch[1].to(device)
        feat_syn_val = feat_syn.val[:, rand_idx]

        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        # compute reconstruction & classification loss
        losses = {}
        out_real, out_real2, sample_syn, feat_syn_update = compute_loss(sample_real, feat_syn_val, label, model, model_ema, losses, rand_idx, perceptual_loss)
        time_meters.add_loss_value('Reconstruction time', time.time() - end)
        end = time.time()
    
        total_loss = losses['recon'] + losses['cls'] #+ losses['recon_H']#+ losses['perception'] 
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model_ema.update(model)
        image_syn.update(sample_syn.detach(), label, rand_idx)
        feat_syn.update(feat_syn_update.detach(), label, rand_idx)

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
                tmp = '{0}: {loss.val:.6f} ({loss.avg:.6f}), '.format(
                        k, loss=loss_meters.average_meters[k]
                )
                msg += tmp
            msg = msg[:-2]

            logger.info(msg)
            if config.vis_recon:
                outs = [sample_real, sample_syn, out_real, out_real2]
                fig, axes = plt.subplots(4, 32, figsize=(5, 5))
                axes = axes.ravel()
                for i, out in enumerate(outs):
                    for j in range(32):
                        _img = out[j].permute(1, 2, 0)
                        _img = (_img * img_std + img_mean).clamp(0, 1)
                        axes[i * 32 + j].imshow(_img.detach().cpu().numpy())
                        axes[i * 32 + j].axis('off')

                plt.savefig(output_dir+f'/out_{epoch}.png')

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
    img_mean = torch.tensor(config.dataset.mean).reshape(1, 1, -1).to(device)
    img_std = torch.tensor(config.dataset.std).reshape(1, 1, -1).to(device)
    num_channel = config.dataset.num_channel
    num_classes = config.dataset.num_classes

    # get model for encoder
    encoders = [ResNetEncoder(num_channel, num_classes, config.model.k, n_blocks=3).to(device)]
    #encoders = []
    for idx, encoder in enumerate(config.pretrain.encoders):
        ckpt_name = f"{config.dataset.name}_{encoder}.pth.tar"
        encoder = get_network(encoder, num_classes, num_channel, (img_size, img_size), device, use_ddp, args.local_rank)
        if os.path.exists("checkpoints/"+ ckpt_name):
            train_utils.load_checkpoint(encoder, "./checkpoints", ckpt_name=ckpt_name, device=device, use_ddp=use_ddp)
        else:
            _, model_state = pretrain_encoder(config, idx, encoder, train_loader, test_loader, device)
            train_utils.save_checkpoint(model_state, "./checkpoints", ckpt_name=ckpt_name)

        encoders.append(encoder)
    
    model = exp_utils.load_component(config.model, config, encoders = encoders).to(device)
    model_ema = ModelEMA(model, 0.999)
    image_syn = ImageEMA(num_classes, ipc, num_channel, img_size, device)
    feat_syn = Feat1DEMA(num_classes, ipc, model.feat_dim, device)
    label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device)

    # get model for evaluation
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
        if config.vis_image_syn:
            fig, axes = plt.subplots(num_classes, ipc, figsize=(5, 5))
            axes = axes.ravel()

            for i in range(num_classes):
                for j in range(ipc):
                    _img = image_syn.val[i][j].permute(1, 2, 0)
                    _img = (_img * img_std + img_mean).clamp(0, 1)
                    if num_channel == 1:
                        _img = _img.tile(1, 1, 3)
                    axes[i * ipc + j].imshow(_img.detach().cpu().numpy())
                    axes[i * ipc + j].axis('off')
            
            plt.savefig(output_dir+f'/img_{epoch}.png')
            
        # Quantitative validataion of image_syn
        time_meters = exp_utils.AverageMeters()
        eval_data = data_utils.TensorDataset(image_syn.val, label_syn)
        eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=config.eval.batch_size, shuffle=True, num_workers=0)

        train_end = time.time()
        for eval_model in eval_model_pool:
            accs = []
            for eval_iter in range(config.eval.num):
                eval_net = get_network(eval_model, num_classes, num_channel, (img_size, img_size), device, use_ddp, args.local_rank)
                _, acc_train, acc_test = eval_synthetic_set(config, eval_iter, eval_net, eval_loader, test_loader, device)
                accs.append(acc_test)
            
        time_meters.add_loss_value(f'Eval time', time.time() - train_end)
        msg = 'Evaluate {0} random {1}, acc mean = {acc_mean:.4f}, std = {acc_std:.4f}, eval_time = {eval_time:.4f}'.format(
            len(accs), 
            eval_model, 
            acc_mean = np.mean(accs), 
            acc_std = np.std(accs),
            eval_time = time_meters.average_meters['Eval time'].val,
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