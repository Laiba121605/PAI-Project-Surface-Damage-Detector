"""SurfaceCNN - VGG-style classifier built entirely from custom layers.

Course rule: no pre-defined library layers. Every Conv2d / BatchNorm2d /
MaxPool2d / Linear / Dropout / ReLU / AdaptiveAvgPool2d used here is
implemented from scratch in layers.py using only raw tensor operations.

Architecture (input -> (3, 256, 256)):
    Block 1: Conv(3->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> MaxPool(2) -> Dropout2d(0.1)
    Block 2: Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> MaxPool(2) -> Dropout2d(0.1)
    Block 3: Conv(64->128) -> BN -> ReLU -> Conv(128->128) -> BN -> ReLU -> MaxPool(2) -> Dropout2d(0.2)
    Block 4: Conv(128->256) -> BN -> ReLU -> Conv(256->256) -> BN -> ReLU -> MaxPool(2) -> Dropout2d(0.2)
    Global Average Pool -> Linear(256, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 3)
"""

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


class ConvBNReLU(nn.Module):
    """One Conv -> BN -> ReLU stack. Pure composition of custom layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = BatchNorm2d(out_ch)
        self.relu = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvBlock(nn.Module):
    """Double Conv-BN-ReLU + MaxPool + Dropout2d."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.drop = Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class SurfaceCNN(nn.Module):
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.5,
                 in_channels: int = 3, dropout2d_scale: float = 1.0,
                 extra_block: bool = False, fc_hidden: int = 128):
        super().__init__()
        s = dropout2d_scale

        # (in_channels, 256, 256) -> (32, 128, 128)
        self.block1 = ConvBlock(in_channels, 32, dropout_p=0.1 * s)
        # (32, 128, 128) -> (64, 64, 64)
        self.block2 = ConvBlock(32, 64, dropout_p=0.1 * s)
        # (64, 64, 64) -> (128, 32, 32)
        self.block3 = ConvBlock(64, 128, dropout_p=0.2 * s)
        # (128, 32, 32) -> (256, 16, 16)
        self.block4 = ConvBlock(128, 256, dropout_p=0.2 * s)

        # Optional deeper 5th block: (256, 16, 16) -> (512, 8, 8). Doubles
        # representational capacity for surfaces (e.g. tiles) where the 4-block
        # network plateaus before reaching the target accuracy.
        if extra_block:
            self.block5 = ConvBlock(256, 512, dropout_p=0.3 * s)
            gap_in = 512
        else:
            self.block5 = None
            gap_in = 256

        # Global Average Pool: (C, 8, 8) or (C, 16, 16) -> (C,)
        self.gap = AdaptiveAvgPool2d(output_size=1)

        # FC head
        self.fc1 = Linear(gap_in, fc_hidden)
        self.relu_fc = ReLU()
        self.drop_fc = Dropout(p=dropout_rate)
        self.fc2 = Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total:,}")
        return total


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """Recover the SurfaceCNN constructor args from a saved state_dict.

    Lets evaluate.py / predict.py reload any checkpoint without the caller
    having to know whether the model was trained with `extra_block` or a
    different `fc_hidden` size.
    """
    has_block5 = any(k.startswith("block5.") for k in state_dict)
    fc1_weight = state_dict.get("fc1.weight")
    fc_hidden = fc1_weight.shape[0] if fc1_weight is not None else 128
    # First-layer conv weight is (out=32, in=in_channels, K, K).
    in_channels = state_dict["block1.conv1.conv.weight"].shape[1]
    return {
        "extra_block": has_block5,
        "fc_hidden": int(fc_hidden),
        "in_channels": int(in_channels),
    }


if __name__ == "__main__":
    model = SurfaceCNN()
    model.count_parameters()
    dummy = torch.zeros(2, 3, 256, 256)
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, 3), f"Expected (2, 3), got {tuple(out.shape)}"

    # Gradient flow sanity check (using our custom CrossEntropyLoss).
    from layers import CrossEntropyLoss
    target = torch.tensor([0, 1])
    loss = CrossEntropyLoss()(out, target)
    loss.backward()
    any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert any_grad, "No gradients flowed through the network"
    print("model.py smoke test passed (forward shape OK, gradients flow).")
