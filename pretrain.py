import time
import logging
from utils import data_utils

import numpy as np
import torch
import torch.nn as nn

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