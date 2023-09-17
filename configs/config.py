import yaml
import os
import numpy as np
from easydict import EasyDict as edict

config = edict()

# experiment config
config.exp_name = 'equiv_dd'
config.output_dir = './output/'
config.workers = 1
config.seed = 0
config.print_freq = 100
config.vis_image_syn = True
config.vis_recon = True

# dataset config
config.dataset = edict()
config.dataset.fn = './datasets/mnist.py'
config.dataset.name = 'MNIST'
config.dataset.img_size = 32
config.dataset.mean = [0.1307]
config.dataset.std = [0.3081]
config.dataset.num_classes = 10
config.dataset.num_channel = 1
config.dataset.ipc = 1

# augmentation config
config.augment = edict()
config.augment.strategy = 'crop_scale_rotate'
config.augment.scale = 0.2
config.augment.crop = 4
config.augment.rotate = 45
config.augment.noise = 0.001

# pretrain config
config.pretrain = edict()
config.pretrain.encoders = []#['MLP']
config.pretrain.batch_size = [256]
config.pretrain.lr = [0.01]
config.pretrain.epoch = [100]

# model config
config.model = edict()
config.model.fn = './models/seqae.py'
config.model.name = 'SeqAELSTSQ'
config.model.feat_dim = 512
config.model.k = 2.
config.model.predictive = True
config.model.alignment = False
config.model.change_of_basis = False

# training config
config.train = edict()
config.train.batch_size = 32
config.train.lr = 0.0003
config.train.lr_decay = 3.0
config.train.total_iteration = 1000000
config.train.adjust_iter_num = []

# evaluation config
config.eval = edict()
config.eval.num = 5
config.eval.mode = 'A'
config.eval.batch_size = 256
config.eval.lr = 0.01
config.eval.epoch = 1000

def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                     config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)