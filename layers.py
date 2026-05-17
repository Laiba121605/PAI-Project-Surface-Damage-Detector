"""From-scratch implementations of every neural-network layer used by the
Surface Damage Detector.

Course rule: no pre-defined library layers may be used. We rely only on:
    - torch.nn.Module      (base class for parameter registration)
    - torch.nn.Parameter   (marks a tensor as a learnable weight)
    - raw torch.* tensor operations (matmul, einsum, mean, var, max, ...)
    - autograd (provided automatically by torch tensor ops)

No torch.nn.Conv2d / Linear / BatchNorm / MaxPool / Dropout / etc.
No torch.nn.functional.* equivalents either.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------

class Conv2d(nn.Module):
    """2D convolution implemented from scratch via the classic im2col trick.

    Pipeline:
        1. Zero-pad the input by `padding` pixels on each side.
        2. Use the Tensor.unfold tensor primitive to extract every K*K patch
           into a 6D tensor of shape (N, C, out_H, out_W, K, K).
        3. Reshape patches to (N, out_H*out_W, C*K*K) and the weight to
           (C*K*K, out_C), then perform a single matrix multiply.
        4. Reshape the result back to (N, out_C, out_H, out_W).

    This is mathematically identical to torch.nn.Conv2d(stride=1, groups=1)
    but built from raw tensor operations (unfold, reshape, permute, matmul).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # He (Kaiming) init for ReLU nets: weights ~ N(0, sqrt(2 / fan_in)).
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        K = self.kernel_size
        P = self.padding

        if P > 0:
            x_pad = torch.zeros(
                N, C, H + 2 * P, W + 2 * P, device=x.device, dtype=x.dtype
            )
            x_pad[:, :, P:P + H, P:P + W] = x
        else:
            x_pad = x

        out_H = H + 2 * P - K + 1
        out_W = W + 2 * P - K + 1

        # tensor.unfold(dim, size, step) returns a sliding-window view.
        # Two applications give us every K*K patch: shape (N, C, out_H, out_W, K, K).
        patches = x_pad.unfold(2, K, 1).unfold(3, K, 1)

        # Rearrange to (N, out_H*out_W, C*K*K) so we can matmul the weight.
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(N, out_H * out_W, C * K * K)

        # (out_C, C, K, K) -> (out_C, C*K*K) -> (C*K*K, out_C) for matmul.
        w_flat = self.weight.view(self.out_channels, -1).t()

        # (N, out_H*out_W, C*K*K) @ (C*K*K, out_C) -> (N, out_H*out_W, out_C)
        out = patches @ w_flat

        # Back to (N, out_C, out_H, out_W).
        out = out.permute(0, 2, 1).contiguous().view(N, self.out_channels, out_H, out_W)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# ---------------------------------------------------------------------------
# Batch normalization (2D)
# ---------------------------------------------------------------------------

class BatchNorm2d(nn.Module):
    """Per-channel batch normalization over the (N, H, W) axes.

    Training:  use batch mean/var; update running stats with momentum.
    Eval:      use running mean/var.
    Affine:    learnable gamma (weight) and beta (bias) per channel.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Buffers travel with state_dict but are not updated by the optimizer.
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach(), alpha=self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var

        mean_b = mean.view(1, -1, 1, 1)
        var_b = var.view(1, -1, 1, 1)
        x_hat = (x - mean_b) / torch.sqrt(var_b + self.eps)
        return self.weight.view(1, -1, 1, 1) * x_hat + self.bias.view(1, -1, 1, 1)


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

class ReLU(nn.Module):
    """ReLU activation: max(0, x). Implemented via torch.clamp."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

class MaxPool2d(nn.Module):
    """Max pooling with square kernel and stride. No padding."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        K, S = self.kernel_size, self.stride
        # tensor.unfold(dim, size, step) extracts sliding windows along a
        # single dim. Applied to both H and W we get a 6D tensor of patches:
        # (N, C, out_H, out_W, K, K). The final max collapses each patch.
        patches = x.unfold(2, K, S).unfold(3, K, S)
        out_H = (H - K) // S + 1
        out_W = (W - K) // S + 1
        return patches.contiguous().view(N, C, out_H, out_W, K * K).max(dim=-1).values


class AdaptiveAvgPool2d(nn.Module):
    """Adaptive average pool to a 1x1 output (global average pooling)."""

    def __init__(self, output_size: int = 1):
        super().__init__()
        if output_size != 1:
            raise NotImplementedError("AdaptiveAvgPool2d here only supports output_size=1.")
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class Dropout(nn.Module):
    """Inverted dropout: drop activations with probability p, scale survivors by 1/(1-p)."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(x.dtype) / keep_prob
        return x * mask


class Dropout2d(nn.Module):
    """Channel-wise dropout: drop entire feature-map channels with probability p."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout2d p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        N, C, _, _ = x.shape
        keep_prob = 1.0 - self.p
        mask = (torch.rand(N, C, 1, 1, device=x.device, dtype=x.dtype) < keep_prob) / keep_prob
        return x * mask


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class Linear(nn.Module):
    """Fully connected layer: y = x @ W^T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # PyTorch's default Linear init: uniform in [-1/sqrt(fan_in), +1/sqrt(fan_in)].
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CrossEntropyLoss(nn.Module):
    """Softmax + negative log-likelihood with optional per-class weights.

    Uses the log-sum-exp trick for numerical stability:
        log_softmax(x)_i = x_i - logsumexp(x).
    """

    def __init__(self, weight: torch.Tensor = None):
        super().__init__()
        if weight is not None:
            # Register weight as a buffer so .to(device) moves it with the module.
            self.register_buffer("class_weight", weight)
        else:
            self.class_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C). target: (N,) int64 class indices.
        # log_softmax = logits - logsumexp(logits, dim=-1, keepdim=True)
        max_logits = logits.max(dim=1, keepdim=True).values
        shifted = logits - max_logits  # numerically stable
        log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
        log_probs = shifted - log_sum_exp  # (N, C)

        # Gather the log-prob assigned to the true class of each sample.
        nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (N,)

        if self.class_weight is not None:
            sample_weights = self.class_weight[target]  # (N,)
            return (nll * sample_weights).sum() / sample_weights.sum()
        return nll.mean()
