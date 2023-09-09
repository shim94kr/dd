
def adjust_lr(config, optimizer, iter_num, adjust_iter_num):
    if iter_num == adjust_iter_num[0]:
        lr = config.train.lr / config.train.lr_decay
    elif iter_num == adjust_iter_num[1]:
        lr = config.train.lr / (config.train.lr_decay ^ 2)
    elif iter_num == adjust_iter_num[2]:
        lr = config.train.lr / (config.train.lr_decay ^ 3)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
