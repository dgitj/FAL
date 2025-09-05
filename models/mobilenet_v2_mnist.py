"""
MobileNetV2 for MNIST dataset in PyTorch
Adapted to match the interface of preact_resnet_mnist.py
"""
import torch
import torch.nn as nn
import math


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_MNIST(nn.Module):
    """MobileNetV2 adapted for MNIST dataset (28x28 grayscale images)"""
    
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2_MNIST, self).__init__()
        
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 1],  # Changed stride from 2 to 1 for MNIST
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 1],  # Reduced stride for smaller images
            [6, 320, 1, 1],
        ]

        # building first layer - MNIST has 1 input channel (grayscale)
        input_channel = int(32 * width_mult)
        self.features = [self._conv_bn(1, input_channel, 1)]  # 1 channel for grayscale, stride 1 for 28x28
        
        # Store feature indices for intermediate outputs
        self.feature_indices = []
        feature_idx = 0
        
        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
                feature_idx += 1
            # Mark the end of each stage for intermediate features
            self.feature_indices.append(feature_idx)
        
        # building last several layers
        self.features.append(self._conv_bn(input_channel, int(1280 * width_mult), 1))
        
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(1280 * width_mult), num_classes)

        self._initialize_weights()

    def _conv_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    def _initialize_weights(self):
        """Initialize weights for reproducibility matching ResNet initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Store intermediate features
        intermediate_features = []
        
        # First conv layer
        x = self.features[0](x)
        
        # Process through stages and collect intermediate outputs
        feature_idx = 1
        stage_outputs = []
        
        # Stage 1: First set of blocks (corresponds to layer1 in ResNet)
        for i in range(1, 2):  # 1 block
            x = self.features[i](x)
        stage_outputs.append(x)
        
        # Stage 2: Second set of blocks (corresponds to layer2 in ResNet)
        for i in range(2, 4):  # 2 blocks
            x = self.features[i](x)
        stage_outputs.append(x)
        
        # Stage 3: Third set of blocks (corresponds to layer3 in ResNet)
        for i in range(4, 7):  # 3 blocks
            x = self.features[i](x)
        stage_outputs.append(x)
        
        # Continue through the rest of the network
        for i in range(7, len(self.features)):
            x = self.features[i](x)
        
        # Global pooling and flatten
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        stage_outputs.append(features)
        
        # Classifier
        logits = self.classifier(features)
        
        # Return format matching ResNet: (logits, [out1, out2, out3, out4])
        return logits, stage_outputs


def mobilenet_v2_mnist(num_classes=10, **kwargs):
    """
    Constructs a MobileNetV2 model for MNIST.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        MobileNetV2 model with interface matching preact_resnet8_mnist
    """
    model = MobileNetV2_MNIST(num_classes=num_classes, **kwargs)
    return model
