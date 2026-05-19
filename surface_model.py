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
)

SURFACE_NAMES = ["tiles", "walls", "wood"]
SURFACE_TO_IDX = {name: i for i, name in enumerate(SURFACE_NAMES)}


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = BatchNorm2d(out_ch)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.pool  = MaxPool2d(kernel_size=2, stride=2)
        self.drop  = Dropout2d(p=dropout_p)

    def forward(self, x):
        return self.drop(self.pool(self.conv2(self.conv1(x))))


class SurfaceTypeCNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=3, dropout_rate=0.4):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 32,  dropout_p=0.1)
        self.block2 = ConvBlock(32,          64,  dropout_p=0.1)
        self.block3 = ConvBlock(64,          128, dropout_p=0.2)
        self.block4 = ConvBlock(128,         256, dropout_p=0.2)
        self.gap    = AdaptiveAvgPool2d(output_size=1)
        self.fc1    = Linear(256, 128)
        self.relu   = ReLU()
        self.drop   = Dropout(p=dropout_rate)
        self.fc2    = Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total:,}")
        return total


def infer_config_from_state_dict(state_dict):
    in_channels = state_dict["block1.conv1.conv.weight"].shape[1]
    fc2_weight  = state_dict.get("fc2.weight")
    num_classes = fc2_weight.shape[0] if fc2_weight is not None else 3
    return {"in_channels": int(in_channels), "num_classes": int(num_classes)}


if __name__ == "__main__":
    model = SurfaceTypeCNN()
    model.count_parameters()
    dummy = torch.zeros(2, 3, 128, 128)
    out   = model(dummy)
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, 3)
    print("surface_model.py smoke test passed.")