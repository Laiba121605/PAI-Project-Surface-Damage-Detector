import math

import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.padding      = padding

        fan_in = in_channels * kernel_size * kernel_size
        std    = math.sqrt(2.0 / fan_in)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.kernel_size
        P = self.padding

        if P > 0:
            x_pad = torch.zeros(N, C, H + 2 * P, W + 2 * P, device=x.device, dtype=x.dtype)
            x_pad[:, :, P:P + H, P:P + W] = x
        else:
            x_pad = x

        out_H   = H + 2 * P - K + 1
        out_W   = W + 2 * P - K + 1
        patches = x_pad.unfold(2, K, 1).unfold(3, K, 1)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(N, out_H * out_W, C * K * K)
        w_flat  = self.weight.view(self.out_channels, -1).t()
        out     = patches @ w_flat
        out     = out.permute(0, 2, 1).contiguous().view(N, self.out_channels, out_H, out_W)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps          = eps
        self.momentum     = momentum
        self.weight       = nn.Parameter(torch.ones(num_features))
        self.bias         = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var  = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach(),   alpha=self.momentum)
        else:
            mean = self.running_mean
            var  = self.running_var

        x_hat = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        return self.weight.view(1, -1, 1, 1) * x_hat + self.bias.view(1, -1, 1, 1)


class ReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0)


class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride

    def forward(self, x):
        N, C, H, W = x.shape
        K, S = self.kernel_size, self.stride
        out_H   = (H - K) // S + 1
        out_W   = (W - K) // S + 1
        patches = x.unfold(2, K, S).unfold(3, K, S)
        return patches.contiguous().view(N, C, out_H, out_W, K * K).max(dim=-1).values


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        if output_size != 1:
            raise NotImplementedError("Only output_size=1 supported.")

    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0,1), got {p}")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand_like(x) < keep).to(x.dtype) / keep
        return x * mask


class Dropout2d(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0,1), got {p}")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        N, C, _, _ = x.shape
        keep = 1.0 - self.p
        mask = (torch.rand(N, C, 1, 1, device=x.device, dtype=x.dtype) < keep) / keep
        return x * mask


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        bound       = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias   = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound)) if bias else None

    def forward(self, x):
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced    = max(1, channels // reduction)
        self.gap   = AdaptiveAvgPool2d(output_size=1)
        self.fc1   = Linear(channels, reduced)
        self.relu  = ReLU()
        self.fc2   = Linear(reduced, channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        s = self.gap(x).view(x.size(0), -1)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s.view(x.size(0), -1, 1, 1)


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            self.register_buffer("class_weight", weight)
        else:
            self.class_weight = None

    def forward(self, logits, target):
        max_logits  = logits.max(dim=1, keepdim=True).values
        shifted     = logits - max_logits
        log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
        log_probs   = shifted - log_sum_exp
        nll         = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.class_weight is not None:
            sample_weights = self.class_weight[target]
            return (nll * sample_weights).sum() / sample_weights.sum()
        return nll.mean()