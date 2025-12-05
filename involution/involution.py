import torch
import torch.nn as nn


class Inv2d(nn.Module):
    """
    2D Involution layer (channel-agnostic, spatial-specific) with grouping.

    This implementation follows the design in
    "Involution: Inverting the Inherence of Convolution for Visual Recognition"
    (Li et al., CVPR 2021), with:

      * Dynamic kernels generated from the input feature map
      * Grouped, channel-agnostic kernels (C / group_ch groups)
      * A small "sigma" mapping (BN + ReLU) in the kernel generator

    Parameters
    ----------
    channels : int
        Number of input and output channels (C).
    kernel_size : int
        Spatial size of the involution kernel (k).
    stride : int
        Stride of the involution (acts like Conv2d stride).
    group_ch : int, optional
        Number of channels per group. Number of groups is C // group_ch.
    red_ratio : int, optional
        Reduction ratio for the kernel generator bottleneck.
    """

    def __init__(self, channels, kernel_size, stride, group_ch=16, red_ratio=2, **kwargs):
        super().__init__()

        # Core configuration
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.group_ch = int(group_ch)
        self.red_ratio = int(red_ratio)

        # Groups and divisibility check
        if self.channels % self.group_ch != 0:
            raise ValueError(
                f"Inv2d: channels ({self.channels}) must be divisible by "
                f"group_ch ({self.group_ch})."
            )
        self.groups = self.channels // self.group_ch

        # Unfold configuration
        self.dilation = 1
        self.padding = (self.kernel_size - 1) // 2

        # Optional pooling for kernel generation when stride > 1
        if self.stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        else:
            self.pool = nn.Identity()

        # Kernel generator ("reduce → sigma → span")
        reduced_channels = max(1, self.channels // self.red_ratio)

        self.reduce = nn.Conv2d(
            in_channels=self.channels,
            out_channels=reduced_channels,
            kernel_size=1,
            bias=True,
        )

        # Sigma mapping: BN + ReLU (as in common Involution refs)
        self.sigma = nn.Sequential(
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )

        self.span = nn.Conv2d(
            in_channels=reduced_channels,
            out_channels=self.kernel_size * self.kernel_size * self.groups,
            kernel_size=1,
            bias=True,
        )

        # Unfold extracts k x k neighborhoods with the same stride/padding
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

    def forward(self, x):
        """
        Input:  x of shape (B, C, H, W)
        Output: y of shape (B, C, H_out, W_out)
        """
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                f"Inv2d: expected input with {self.channels} channels, got {c}."
            )

        # ----- Kernel generation -----
        # (Optional) pool if stride > 1
        x_kernel = self.pool(x)

        # Reduce → BN+ReLU → span
        kernel = self.reduce(x_kernel)
        kernel = self.sigma(kernel)
        kernel = self.span(kernel)
        # kernel: (B, groups * k^2, H_out, W_out)

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

        # Reshape to (B, groups, K, H_out, W_out),
        # where K = k^2 is the neighborhood index
        kernel = kernel.view(b, self.groups, k2, h_out, w_out)

        # ----- Neighborhood extraction -----
        patches = self.unfold(x)
        # patches: (B, C * k^2, H_out * W_out)

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

        # Reshape to (B, groups, group_ch, K, H_out, W_out)
        patches = patches.view(
            b,
            self.groups,
            self.group_ch,
            k2,
            h_out,
            w_out,
        )

        # ----- Involution operation -----
        # We want: out[b,g,c,h,w] = sum_k kernel[b,g,k,h,w] * patches[b,g,c,k,h,w]
        #
        # kernel:  (B, G, K,   H_out, W_out)
        # patches: (B, G, Cg,  K,     H_out, W_out)
        #
        # Contract over K using einsum (no broadcasted giant intermediate):
        out = torch.einsum(
            "bgkij,bgckij->bgcij",
            kernel,
            patches,
        )
        # out: (B, G, Cg, H_out, W_out)

        # Merge groups and per-group channels back into (B, C, H_out, W_out)
        out = out.reshape(b, self.channels, h_out, w_out)

        return out
