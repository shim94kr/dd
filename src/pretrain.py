import os
import time
import pprint
import logging
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from utils import exp_utils, data_utils, train_utils
from configs.config import config, update_config

logger = logging.getLogger(__name__)

def pretrain_epoch(config, mode, data_loader, net, optimizer, criterion, aug, device):
    loss_avg, acc_avg, num_exp = 0, 0, 0

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for idx, batch in enumerate(data_loader):
        img = batch[0].float().to(device)
        if aug:
            #img = DiffAugment(img, dsa_strategy, param=dsa_param)
            img = data_utils.augment(config, img, device)
        label = batch[1].long().to(device)
        batch_size = label.shape[0]

        output = net(img)
        loss = criterion(output, label)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), label.cpu().data.numpy()))

        loss_avg += loss.item()*batch_size
        acc_avg += acc
        num_exp += batch_size

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

def pretrain_encoder(config, idx, net, train_loader, test_loader, device, use_ddp=False):
    lr = float(config.pretrain.lr[idx])
    epoch = int(config.pretrain.epoch[idx])
    lr_schedule = [epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)

    start = time.time()
    for ep in range(epoch):
        train_loss, train_acc = pretrain_epoch(config, 'train', train_loader, net, optimizer, criterion, aug = True, device=device)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        msg = 'Epoch_{0}: train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}'.format(
            ep,
            train_loss = train_loss,
            train_acc = train_acc,
        )
        logger.info(msg)

    train_time = time.time() - start
    test_loss, test_acc = pretrain_epoch(config, 'test', test_loader, net, optimizer, criterion, aug = False, device=device)

    msg = 'Evaluate_{0}: train time = {1:.2f}, train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, test_acc = {test_acc:.4f}'.format(
        idx,
        train_time,
        train_loss = train_loss,
        train_acc = train_acc,
        test_acc = test_acc
    )
    logger.info(msg)
    result_dict = {
        'train_loss': train_loss,
        'train_acc': train_acc, 
        'test_acc': test_acc
    }

    if use_ddp:
        model_state = {
            'epoch': epoch+1,
            'state_dict': net.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'eval_dict': result_dict,
        }
    else:
        model_state = {
            'epoch': epoch+1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'eval_dict': result_dict,
        }
    return net, model_state

def main():
    # Get args and config
    args = parse_args()
    logger, output_dir = exp_utils.create_logger(config, args.cfg, phase='pretrain')
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
    img_size = config.dataset.img_size
    num_channel = config.dataset.num_channel
    num_classes = config.dataset.num_classes

    # get model for encoder
    for idx, encoder in enumerate(config.pretrain.encoders):
        ckpt_name = f"{config.dataset.name}_{encoder}.pth.tar"
        encoder = get_network(encoder, num_classes, num_channel, (img_size, img_size), device, use_ddp, args.local_rank)
        _, model_state = pretrain_encoder(config, idx, encoder, train_loader, test_loader, device)
        train_utils.save_checkpoint(model_state, "./checkpoints", ckpt_name=ckpt_name)
    
def main():
    # Get args and config
    args = parse_args()
    logger, output_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

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