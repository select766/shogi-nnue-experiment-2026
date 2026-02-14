"""
AobaZero neural network model re-implemented in PyTorch.

Architecture: 256 filters x 19 residual blocks (Swish activation)
Input: 362 channels x 9 x 9
Policy output: 2187 (27 x 9 x 9)
Value output: scalar (tanh, [-1, +1])

Reference: aobazero/repo/learn/aoba_256x20b_swish_predict.prototxt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """Single residual block: Conv-BN-Swish-Conv-BN + skip connection, then Swish."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)
        self.swish = Swish()

    def forward(self, x):
        residual = x
        out = self.swish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.swish(out + residual)
        return out


class AobaZeroNet(nn.Module):
    """
    AobaZero network (Swish version, 256x20b).

    Backbone outputs: feature map of shape (batch, 256, 9, 9)
    Policy head outputs: (batch, 2187)
    Value head outputs: (batch, 1) in [-1, 1]
    """

    INPUT_CHANNELS = 362
    FILTERS = 256
    RESIDUAL_BLOCKS = 19
    POLICY_INTERMEDIATE = 160
    POLICY_OUTPUT_CHANNELS = 27  # 27 * 9 * 9 = 2187
    VALUE_INTERMEDIATE_CHANNELS = 4  # 4 * 9 * 9 = 324
    VALUE_FC1_OUT = 256

    def __init__(self):
        super().__init__()

        # Initial convolution
        self.conv_initial = nn.Conv2d(self.INPUT_CHANNELS, self.FILTERS, 3, padding=1, bias=True)
        self.bn_initial = nn.BatchNorm2d(self.FILTERS)
        self.swish = Swish()

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.FILTERS) for _ in range(self.RESIDUAL_BLOCKS)
        ])

        # Policy head
        self.policy_conv1 = nn.Conv2d(self.FILTERS, self.POLICY_INTERMEDIATE, 1, bias=True)
        self.policy_bn1 = nn.BatchNorm2d(self.POLICY_INTERMEDIATE)
        self.policy_conv2 = nn.Conv2d(self.POLICY_INTERMEDIATE, self.POLICY_OUTPUT_CHANNELS, 1, bias=True)

        # Value head
        self.value_conv1 = nn.Conv2d(self.FILTERS, self.VALUE_INTERMEDIATE_CHANNELS, 1, bias=True)
        self.value_bn1 = nn.BatchNorm2d(self.VALUE_INTERMEDIATE_CHANNELS)
        self.value_fc1 = nn.Linear(self.VALUE_INTERMEDIATE_CHANNELS * 9 * 9, self.VALUE_FC1_OUT)
        self.value_fc2 = nn.Linear(self.VALUE_FC1_OUT, 1)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone (initial conv + residual blocks).

        Args:
            x: Input tensor of shape (batch, 362, 9, 9)

        Returns:
            Feature map of shape (batch, 256, 9, 9)
        """
        out = self.swish(self.bn_initial(self.conv_initial(x)))
        for block in self.residual_blocks:
            out = block(out)
        return out

    def policy_head(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute policy from backbone features.

        Args:
            feat: Feature map of shape (batch, 256, 9, 9)

        Returns:
            Policy logits of shape (batch, 2187)
        """
        out = self.swish(self.policy_bn1(self.policy_conv1(feat)))
        out = self.policy_conv2(out)
        return out.view(out.size(0), -1)  # (batch, 27*9*9) = (batch, 2187)

    def value_head(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute value from backbone features.

        Args:
            feat: Feature map of shape (batch, 256, 9, 9)

        Returns:
            Value of shape (batch, 1) in [-1, 1]
        """
        out = self.swish(self.value_bn1(self.value_conv1(feat)))
        out = out.view(out.size(0), -1)  # (batch, 4*9*9) = (batch, 324)
        out = self.swish(self.value_fc1(out))
        out = torch.tanh(self.value_fc2(out))
        return out

    def forward(self, x: torch.Tensor):
        """Full forward pass.

        Args:
            x: Input tensor of shape (batch, 362, 9, 9)

        Returns:
            Tuple of (policy_logits, value):
                policy_logits: (batch, 2187)
                value: (batch, 1) in [-1, 1]
        """
        feat = self.backbone(x)
        policy = self.policy_head(feat)
        value = self.value_head(feat)
        return policy, value
