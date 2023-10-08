import os
import sys
import time
import shutil
from pathlib import Path
import logging
from distutils import dir_util
import wandb

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeters:
    def __init__(self):
        super().__init__()
        self.average_meters = {}

    def add_loss_value(self, loss_name, loss_val, n=1):
        if loss_name not in self.average_meters:
            self.average_meters[loss_name] = AverageMeter()
        self.average_meters[loss_name].update(loss_val, n=n)
        
def load_module(fn, name):
    mod_name = os.path.splitext(os.path.basename(fn))[0]
    mod_path = os.path.dirname(fn)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)

def load_component(config, config_all, **kwargs):
    class_fn = load_module(config.fn, config.name)
    return class_fn(config_all, **kwargs)

def create_logger(cfg, cfg_name, phase='train'):
    this_dir = Path(os.path.dirname(__file__))
    root_dir = (this_dir / '../..').resolve()
    output_dir = (root_dir / cfg.output_dir).resolve()
    src_dir = (root_dir / 'src').resolve()
    config_dir = (root_dir / 'configs').resolve()

    if not output_dir.exists():
        print('Creating output dir {}'.format(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.dataset.name
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    exp_name = cfg.exp_name

    final_output_dir = output_dir / dataset_name / cfg_name / exp_name
    final_images_dir = final_output_dir / 'images'
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_images_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=str(final_log_file),
                        format=head,
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    #from IPython import embed; embed()
   
    output_dir = str(final_output_dir)
    wandb.init(
        project = "dd",
        entity = cfg.entity,
        name = cfg.exp_name,
        dir = output_dir
    )

    dir_util.copy_tree(src_dir, output_dir + "/src")
    dir_util.copy_tree(config_dir, output_dir + "/configs")
    logger.info(f"source copied to {output_dir}/src")
    return logger, output_dir