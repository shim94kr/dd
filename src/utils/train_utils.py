import os
import torch
from distutils.dir_util import copy_tree

def adjust_lr(config, optimizer, iter_num, adjust_iter_num):
    if iter_num == adjust_iter_num[0]:
        lr = config.train.lr / config.train.lr_decay
    elif iter_num == adjust_iter_num[1]:
        lr = config.train.lr / (config.train.lr_decay ^ 2)
    elif iter_num == adjust_iter_num[2]:
        lr = config.train.lr / (config.train.lr_decay ^ 3)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_checkpoint(model, resume_root, ckpt_name='ckpt_last.pth.tar', strict=True, device=None, use_ddp=False):
    resume_path = os.path.join(resume_root, ckpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if use_ddp:
            if "module" in list(checkpoint["state_dict"].keys())[0]:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = {"module.{}".format(key): item for key, item in checkpoint["state_dict"].items()}
        else:
            if "module" in list(checkpoint["state_dict"].keys())[0]:
                state_dict = {key.replace("module.", ""): item for key, item in checkpoint["state_dict"].items()}
            else:
                state_dict = checkpoint["state_dict"]
        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

def save_checkpoint(state, dir_path="./checkpoints", ckpt_name="checkpoint.pth.tar"):
    file_path = os.path.join(dir_path, ckpt_name)
    os.makedirs(dir_path, exist_ok=True)
    torch.save(state, file_path)