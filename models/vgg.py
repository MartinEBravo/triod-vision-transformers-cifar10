# -*- coding: utf-8 -*-

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from triod.layers.linear import TriODLinear
from triod.layers.conv import TriODConv2d
from triod.layers.batch_norm import TriODBatchNorm2d
from triod.utils import SequentialWithP, generate_structured_masked_x


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, triangular=True, p_s=None):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], triangular=triangular)
        self.classifier = TriODLinear(512, num_classes, triangular=False)
        self.p_s = p_s

    def forward(self, x, p=None, return_prelast=False, return_allmodels=False):
        out = self.features(x, p=p)
        out = out.view(out.size(0), -1)
        if return_allmodels:
            out = generate_structured_masked_x(out, self.p_s)
        if return_prelast:
            return out
        return self.classifier(out,p=None)

    def _make_layers(self, cfg, triangular=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [TriODConv2d(in_channels, x, kernel_size=3, padding=1, triangular=triangular),
                           TriODBatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return SequentialWithP(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x, p=0.5)
    print(y.size())

# test()