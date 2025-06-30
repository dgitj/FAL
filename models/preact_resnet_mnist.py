"""
Modified PreActResNet for MNIST dataset.
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreAct_ResNet_MNIST(nn.Module):
    """PreAct ResNet model for MNIST dataset (1 channel grayscale images)"""
    
    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_MNIST, self).__init__()
        self.inplanes = 16
        # MNIST is 1 channel (grayscale) instead of 3 channels (RGB)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        # MNIST is 28x28, so we need a 7x7 average pooling (28/4 = 7)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        x = self.bn(out3)
        x = self.relu(x)
        x = self.avgpool(x)
        out4 = x.view(x.size(0), -1)
        x = self.fc(out4)

        return x, [out1, out2, out3, out4]


def preact_resnet8_mnist(num_classes=10):
    """Constructs a PreactResNet-8 model for MNIST dataset."""
    model = PreAct_ResNet_MNIST(PreActBlock, [1, 1, 1], num_classes=num_classes)
    return model
