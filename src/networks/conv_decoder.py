import torch
import torch.nn as nn

''' ConvNet '''
class ConvNet_Decoder(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_upsampling='bilinear', feat_size = (4, 4)):
        super().__init__()

        self.linear = None
        if channel != net_width * feat_size[0] * feat_size[1]:
            self.linear = nn.Linear(channel, net_width * feat_size[0] * feat_size[1])
        self.features, shape_feat = self._make_layers(net_width, net_depth, net_norm, net_act, net_upsampling, feat_size)
        self.net_last = nn.Conv2d(shape_feat[0], 3, 3, 1, 1)

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        if self.linear is not None:
            out = self.linear(x)
        else:
            out = x
        out = out.view(out.size(0), -1, 4, 4)
        out = self.features(out)
        out = self.net_last(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_upsampling(self, net_upsampling):
        if net_upsampling == 'bilinear':
            return nn.UpsamplingBilinear2d(scale_factor=2)
        elif net_upsampling == 'nearest':
            return nn.UpsamplingNearest2d(scale_factor=2)
        elif net_upsampling == 'none':
            return None
        else:
            exit('unknown net_upsampling: %s'%net_upsampling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, net_width, net_depth, net_norm, net_act, net_upsampling, feat_size):
        layers = []

        in_channels = net_width
        shape_feat = [net_width, feat_size[0], feat_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_upsampling != 'none':
                layers += [self._get_upsampling(net_upsampling)]
                shape_feat[1] *= 2
                shape_feat[2] *= 2
        
        return nn.Sequential(*layers), shape_feat