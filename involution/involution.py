import torch
import torch.nn as nn
from typing import Any


class Inv2d(nn.Module):
    """
    2D Involution layer (channel-agnostic, spatial-specific) with grouping.
    https://arxiv.org/pdf/2103.06255.pdf

    Parameters
    ----------
    channels : int
        Number of input and output channels (C).
    kernel_size : int
        Spatial size of the involution kernel (k).
    stride : int
        Stride of the involution (s). Acts like Conv2d stride.
    group_ch : int, optional
        Number of channels per group (default: 16).
        Number of groups is computed as C // group_ch.
    red_ratio : int, optional
        Reduction ratio for the kernel generator bottleneck (default: 2).

    Notes
    -----
    Input  : (B, C, H, W)
    Output : (B, C, H_out, W_out)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        group_ch: int = 16,
        red_ratio: int = 2,
        **kwargs: Any
    ) -> None:
        # Call parent constructor
        super().__init__()

        # Store configuration parameters
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.group_ch = int(group_ch)
        self.red_ratio = int(red_ratio)

        # Compute number of groups and check divisibility
        if self.channels % self.group_ch != 0:
            raise ValueError(
                f"Inv2d: channels ({self.channels}) must be divisible by "
                f"group_ch ({self.group_ch})."
            )
        self.groups = self.channels // self.group_ch

        # Set involution padding and dilation for unfold
        self.dilation = 1
        self.padding = (self.kernel_size - 1) // 2

        # Define optional pooling for kernel generation when stride > 1
        self.pool = (
            nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
            if self.stride > 1
            else nn.Identity()
        )

        # Compute reduced channels for kernel generator
        reduced_channels = max(1, self.channels // self.red_ratio)

        # Define reduction 1x1 convolution for kernel generator
        self.reduce = nn.Conv2d(
            in_channels=self.channels,
            out_channels=reduced_channels,
            kernel_size=1,
            bias=True
        )

        # Define sigma mapping (BN + ReLU) for kernel generator
        self.sigma = nn.Sequential(
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        # Define span 1x1 convolution to produce k^2 * groups kernels
        self.span = nn.Conv2d(
            in_channels=reduced_channels,
            out_channels=self.kernel_size * self.kernel_size * self.groups,
            kernel_size=1,
            bias=True
        )

        # Define unfold operator to extract local k x k patches
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unpack input shape and validate channels
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                f"Inv2d: expected input with {self.channels} channels, got {c}."
            )

        # Generate kernels via pooling, reduction, sigma mapping, and span
        x_kernel = self.pool(x)
        kernel = self.reduce(x_kernel)
        kernel = self.sigma(kernel)
        kernel = self.span(kernel)

        # Reshape kernel to (B, groups, K, H_out, W_out)
        b_k, ck, h_out, w_out = kernel.shape
        if b_k != b:
            raise RuntimeError(
                f"Inv2d: batch mismatch between input ({b}) and kernel ({b_k})."
            )

        k2 = self.kernel_size * self.kernel_size
        if ck != self.groups * k2:
            raise RuntimeError(
                f"Inv2d: expected kernel channels {self.groups * k2}, got {ck}."
            )

        kernel = kernel.view(
            b,
            self.groups,
            k2,
            h_out,
            w_out
        )

        # Extract local patches using unfold
        patches = self.unfold(x)

        # Reshape patches to (B, groups, group_ch, K, H_out, W_out)
        b_u, ck_u, l = patches.shape
        if b_u != b:
            raise RuntimeError(
                f"Inv2d: batch mismatch between input ({b}) and patches ({b_u})."
            )

        expected_l = h_out * w_out
        if l != expected_l:
            raise RuntimeError(
                f"Inv2d: spatial mismatch between kernel ({h_out}x{w_out}) "
                f"and unfolded patches length ({l})."
            )

        if ck_u != self.channels * k2:
            raise RuntimeError(
                f"Inv2d: expected unfolded channels {self.channels * k2}, got {ck_u}."
            )

        patches = patches.view(
            b,
            self.groups,
            self.group_ch,
            k2,
            h_out,
            w_out
        )

        # Contract over kernel dimension using einsum to get (B, G, Cg, H_out, W_out)
        out = torch.einsum(
            "bgkij,bgckij->bgcij",
            kernel,
            patches
        )

        # Reshape back to (B, C, H_out, W_out); reshape works with non-contiguous tensors
        out = out.reshape(b, self.channels, h_out, w_out)

        # Return the involution output
        return out
