from .alexnet import AlexNet
from .conv import ConvNet
from .dnfr import DNFR
from .resnet import ResNet, ResNet18, ResNet18ImageNet
from .vgg import VGG, VGG11, VGG13, VGG16, VGG19, VGG11BN
from .vit import ViT
from .conv_gap import ConvNetGAP
from .alexnet_cifar import AlexNetCIFAR
from .resnet_cifar import ResNet18CIFAR
from .vgg_cifar import VGG11CIFAR
from .vit_cifar import ViTCIFAR
from .conv_decoder import ConvNet_Decoder
from .base_network import ResNetDecoder, ResNetEncoder
from .stylegan2 import Generator

import time
import torch

def get_eval_pool(eval_mode):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ["ResNet18", "VGG11", "AlexNet", "ViT"]
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    #elif eval_mode == 'S': # itself
    #    model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'C':
        model_eval_pool = ['ConvNet']
    elif eval_mode == "big":
        model_eval_pool = ["RN18", "VGG11_big", "ViT"]
    elif eval_mode == "small":
        model_eval_pool = ["ResNet18", "VGG11", "LeNet", "AlexNet"]
    elif eval_mode == "ConvNet_Norm":
        model_eval_pool = ["ConvNet_BN", "ConvNet_IN", "ConvNet_LN", "ConvNet_NN", "ConvNet_GN"]
    elif eval_mode == "CIFAR":
        model_eval_pool = ["AlexNetCIFAR", "ResNet18CIFAR", "VGG11CIFAR", "ViTCIFAR"]
    else:
        raise ValueError("There is no evaluation available for this mode")
    return model_eval_pool


def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling, net_upsampling = 128, 3, 'relu', 'instancenorm', 'avgpooling', 'bilinear'
    return net_width, net_depth, net_act, net_norm, net_pooling, net_upsampling

def get_network(config, model, channel, num_classes, im_size=(32, 32), device=None, use_ddp=True, local_rank=-1, depth=3, width=128, norm="instancenorm"):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling, net_upsampling = get_default_convnet_setting()

    if model == 'BaseEnc':
        net = ResNetEncoder(channel, num_classes, config.model.k)
    elif model == 'BaseDec':
        net = ResNetDecoder(channel, k=config.model.k)
    elif model == 'StyleGAN2':
        net = Generator(config)
    elif model == 'AlexNet':
        net = AlexNet(channel, num_classes=num_classes, im_size=im_size)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes, norm=norm)
    elif model == "ViT":
        net = ViT(
            image_size = im_size,
            patch_size = 16,
            num_classes = num_classes,
            dim = 512,
            depth = 10,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
        )
    elif model == "AlexNetCIFAR":
        net = AlexNetCIFAR(channel=channel, num_classes=num_classes)
    elif model == "ResNet18CIFAR":
        net = ResNet18CIFAR(channel=channel, num_classes=num_classes)
    elif model == "VGG11CIFAR":
        net = VGG11CIFAR(channel=channel, num_classes=num_classes)
    elif model == "ViTCIFAR":
        net = ViTCIFAR(
                image_size = im_size,
                patch_size = 4,
                num_classes = num_classes,
                dim = 512,
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1)

    elif model == 'ConvNet_Decoder':
        net = ConvNet_Decoder(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm=norm, feat_size=(4,4))
    elif model == "ConvNet":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm=norm, im_size=im_size)
    elif model == "ConvNetGAP":
        net = ConvNetGAP(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm=norm, im_size=im_size)
    elif model == "ConvNet_BN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="batchnorm",
                      im_size=im_size)
    elif model == "ConvNet_IN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="instancenorm",
                      im_size=im_size)
    elif model == "ConvNet_LN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="layernorm",
                      im_size=im_size)
    elif model == "ConvNet_GN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="groupnorm",
                      im_size=im_size)
    elif model == "ConvNet_NN":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm="none",
                      im_size=im_size)
    else:
        net = None
        exit('DC error: unknown model')

    net = net.to(device)
    if use_ddp:
        net =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

    return net
