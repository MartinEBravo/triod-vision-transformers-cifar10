# -*- coding: utf-8 -*-

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from triod.layers.conv import TriODConv2d
from triod.layers.linear import TriODLinear
from triod.layers.batch_norm import TriODBatchNorm2d
from triod.layers.sequential import TriODSequential
from triod.utils import generate_structured_masked_x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, triangular=False):
        super(BasicBlock, self).__init__()
        self.conv1 = TriODConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, triangular=triangular)
        self.bn1 = TriODBatchNorm2d(planes)
        self.conv2 = TriODConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, triangular=triangular)
        self.bn2 = TriODBatchNorm2d(planes)

        self.shortcut = TriODSequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = TriODSequential(
                TriODConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, triangular=triangular),
                TriODBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, p=None):
        out = F.relu(self.bn1(self.conv1(x,p)))
        out = self.bn2(self.conv2(out,p))
        out += self.shortcut(x,p)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, triangular=False):
        super(Bottleneck, self).__init__()
        self.conv1 = TriODConv2d(in_planes, planes, kernel_size=1, bias=False, triangular=triangular)
        self.bn1 = TriODBatchNorm2d(planes)
        self.conv2 = TriODConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, triangular=triangular)
        self.bn2 = TriODBatchNorm2d(planes)
        self.conv3 = TriODConv2d(planes, self.expansion*planes, kernel_size=1, bias=False, triangular=triangular)
        self.bn3 = TriODBatchNorm2d(self.expansion*planes)

        self.shortcut = TriODSequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = TriODSequential(
                TriODConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, triangular=triangular),
                TriODBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, p=None):
        out = F.relu(self.bn1(self.conv1(x,p)))
        out = F.relu(self.bn2(self.conv2(out,p)))
        out = self.bn3(self.conv3(out,p))
        out += self.shortcut(x,p)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, triangular=False, p_s=None):
        super(ResNet, self).__init__()
        if triangular:
            print("Using triangular layers")
            mult = np.sqrt(2)
        else:
            mult = 1
        self.p_s = p_s
        self.triangular = triangular
        self.in_planes = int(64 * mult)

        self.conv1 = TriODConv2d(3, int(64 * mult), kernel_size=3, stride=1, padding=1, bias=False, triangular=triangular)
        self.bn1 = TriODBatchNorm2d(int(64 * mult))
        self.layer1 = self._make_layer(block, int(64 * mult), num_blocks[0], stride=1, triangular=triangular)
        self.layer2 = self._make_layer(block, int(128 * mult), num_blocks[1], stride=2, triangular=triangular)
        self.layer3 = self._make_layer(block, int(256 * mult), num_blocks[2], stride=2, triangular=triangular)
        self.layer4 = self._make_layer(block, int(512 * mult), num_blocks[3], stride=2, triangular=triangular)
        self.classifier = TriODLinear(int(512 * mult)*block.expansion, num_classes, triangular=False)
    def _make_layer(self, block, planes, num_blocks, stride, triangular):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, triangular=triangular))
            self.in_planes = planes * block.expansion
        return TriODSequential(*layers)

    def forward(self, x, p=None, return_prelast=False, all_models=False):
        out = F.relu(self.bn1(self.conv1(x,p)))
        out = self.layer1(out,p)
        out = self.layer2(out,p)
        out = self.layer3(out,p)
        out = self.layer4(out,p)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if all_models:
            out = generate_structured_masked_x(out, self.p_s)
        if return_prelast:
            return out
        return self.classifier(out,p=None)


def ResNet18(num_classes=10, triangular=False, p_s=None):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, triangular=triangular, p_s=p_s)

def ResNet34(num_classes=10, triangular=False, p_s=None):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, triangular=triangular, p_s=p_s)

def ResNet50(num_classes=10, triangular=False, p_s=None):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, triangular=triangular, p_s=p_s)

def ResNet101(num_classes=10, triangular=False, p_s=None):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, triangular=triangular, p_s=p_s)

def ResNet152(num_classes=10, triangular=False, p_s=None):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, triangular=triangular, p_s=p_s)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()