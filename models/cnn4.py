"""
CNN4: A simple 4-layer convolutional network with 64 hidden channels for CIFAR-10.
"""
import torch.nn as nn
import torch.nn.functional as F


class CNN4(nn.Module):
    """
    4-layer convolutional network with 64 hidden channels.
    Designed for CIFAR-10 (3-channel, 32x32 images).

    Forward returns (logits, [out1, out2, out3, features]) to match
    the interface of preact_resnet.py and mobilenet_v2.py.
    """

    def __init__(self, num_classes=10):
        super(CNN4, self).__init__()

        # Layer 1: 3 -> 64, 32x32 -> 32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)

        # Layer 2: 64 -> 64, 32x32 -> 16x16
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 16x16

        # Layer 3: 64 -> 64, 16x16 -> 16x16
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(64)

        # Layer 4: 64 -> 64, 16x16 -> 8x8
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)  # -> 8x8

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # -> 64x1x1
        self.fc = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # Layer 1
        out1 = F.relu(self.bn1(self.conv1(x)))       # (B, 64, 32, 32)

        # Layer 2
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out2 = self.pool2(out2)                       # (B, 64, 16, 16)

        # Layer 3
        out3 = F.relu(self.bn3(self.conv3(out2)))     # (B, 64, 16, 16)

        # Layer 4
        out4 = F.relu(self.bn4(self.conv4(out3)))
        out4 = self.pool4(out4)                       # (B, 64, 8, 8)

        # Global pooling + flatten
        x = self.avgpool(out4)
        features = x.view(x.size(0), -1)             # (B, 64)

        logits = self.fc(features)

        # Return intermediate feature maps to match the preact_resnet interface
        return logits, [out1, out2, out3, features]


def cnn4_cifar(num_classes=10, **kwargs):
    """4-layer CNN with 64 hidden channels for CIFAR-10."""
    return CNN4(num_classes=num_classes)