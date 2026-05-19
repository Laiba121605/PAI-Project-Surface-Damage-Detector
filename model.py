import torch
import torch.nn as nn

from layers import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Dropout2d,
    Linear,
    MaxPool2d,
    ReLU,
    SEBlock,
)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = BatchNorm2d(out_ch)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p, se_reduction=8):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.se    = SEBlock(out_ch, reduction=se_reduction)
        self.pool  = MaxPool2d(kernel_size=2, stride=2)
        self.drop  = Dropout2d(p=dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class SurfaceCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.25,
                 in_channels=6, dropout2d_scale=0.5,
                 extra_block=False, fc_hidden=256):
        super().__init__()
        s = dropout2d_scale

        self.block1 = ConvBlock(in_channels, 32,  dropout_p=0.05 * s)
        self.block2 = ConvBlock(32,          64,  dropout_p=0.10 * s)
        self.block3 = ConvBlock(64,          128, dropout_p=0.10 * s)
        self.block4 = ConvBlock(128,         256, dropout_p=0.15 * s)

        if extra_block:
            self.block5 = ConvBlock(256, 512, dropout_p=0.15 * s)
            gap_in = 512
        else:
            self.block5 = None
            gap_in = 256

        self.gap     = AdaptiveAvgPool2d(output_size=1)
        self.fc1     = Linear(gap_in, fc_hidden)
        self.relu_fc = ReLU()
        self.drop_fc = Dropout(p=dropout_rate)
        self.fc2     = Linear(fc_hidden, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        if self.block5 is not None:
            x = self.block5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total:,}")
        return total


def infer_config_from_state_dict(state_dict):
    has_block5  = any(k.startswith("block5.") for k in state_dict)
    fc1_weight  = state_dict.get("fc1.weight")
    fc_hidden   = fc1_weight.shape[0] if fc1_weight is not None else 256
    in_channels = state_dict["block1.conv1.conv.weight"].shape[1]
    return {
        "extra_block": has_block5,
        "fc_hidden":   int(fc_hidden),
        "in_channels": int(in_channels),
    }