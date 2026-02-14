"""Convert AobaZero text weight file to PyTorch state_dict.

Text weight format (version 3):
  Line 1: "3" (version)
  Then groups of 4 lines per conv+BN pair: conv_weight, conv_bias, bn_mean, bn_var
  - Initial conv+bn: 4 lines
  - 19 residual blocks × 2 pairs × 4 lines = 152 lines
  - Policy head: conv1+bn (4 lines) + conv2 (2 lines) = 6 lines
  - Value head: conv1+bn (4 lines) + fc1 (2 lines) + fc2 (2 lines) = 8 lines
  Total: 1 + 4 + 152 + 6 + 8 = 171 lines

BN values in the text file are Caffe's raw accumulated sums.
Divide by scale_factor (999.982) to get actual mean/var.
"""

import argparse

import numpy as np
import torch

from train_nnue.aobazero_model import AobaZeroNet

# Caffe BatchNorm accumulation scale factor (hardcoded in AobaZero C++ loader)
BN_SCALE_FACTOR = 999.982


def parse_weight_file(path: str) -> list[list[float]]:
    """Read text weight file, return list of lines (each line is a list of floats)."""
    lines = []
    with open(path, "r") as f:
        version_line = f.readline().strip()
        assert version_line == "3", f"Expected version 3, got {version_line}"
        for line in f:
            values = [float(x) for x in line.strip().split()]
            lines.append(values)
    assert len(lines) == 170, f"Expected 170 data lines, got {len(lines)}"
    return lines


def load_conv_bn(lines: list[list[float]], idx: int, conv: torch.nn.Conv2d,
                 bn: torch.nn.BatchNorm2d) -> int:
    """Load 4 lines (conv_w, conv_b, bn_mean, bn_var) into conv+bn modules."""
    out_ch = conv.out_channels
    in_ch = conv.in_channels
    kh, kw = conv.kernel_size

    conv_w = np.array(lines[idx], dtype=np.float32).reshape(out_ch, in_ch, kh, kw)
    conv.weight.data = torch.from_numpy(conv_w)

    conv_b = np.array(lines[idx + 1], dtype=np.float32)
    assert len(conv_b) == out_ch, f"Conv bias size mismatch: {len(conv_b)} vs {out_ch}"
    conv.bias.data = torch.from_numpy(conv_b)

    bn_mean = np.array(lines[idx + 2], dtype=np.float32) / BN_SCALE_FACTOR
    assert len(bn_mean) == out_ch
    bn.running_mean = torch.from_numpy(bn_mean)

    bn_var = np.array(lines[idx + 3], dtype=np.float32) / BN_SCALE_FACTOR
    assert len(bn_var) == out_ch
    bn.running_var = torch.from_numpy(bn_var)

    # Caffe BatchNorm has no learnable scale/bias (no Scale layer in AobaZero)
    bn.weight.data.fill_(1.0)
    bn.bias.data.fill_(0.0)

    return idx + 4


def load_conv_only(lines: list[list[float]], idx: int, conv: torch.nn.Conv2d) -> int:
    """Load 2 lines (conv_w, conv_b) into conv module."""
    out_ch = conv.out_channels
    in_ch = conv.in_channels
    kh, kw = conv.kernel_size

    conv_w = np.array(lines[idx], dtype=np.float32).reshape(out_ch, in_ch, kh, kw)
    conv.weight.data = torch.from_numpy(conv_w)

    conv_b = np.array(lines[idx + 1], dtype=np.float32)
    assert len(conv_b) == out_ch
    conv.bias.data = torch.from_numpy(conv_b)

    return idx + 2


def load_fc(lines: list[list[float]], idx: int, fc: torch.nn.Linear) -> int:
    """Load 2 lines (fc_w, fc_b) into Linear module."""
    out_f = fc.out_features
    in_f = fc.in_features

    fc_w = np.array(lines[idx], dtype=np.float32).reshape(out_f, in_f)
    fc.weight.data = torch.from_numpy(fc_w)

    fc_b = np.array(lines[idx + 1], dtype=np.float32)
    assert len(fc_b) == out_f
    fc.bias.data = torch.from_numpy(fc_b)

    return idx + 2


def convert_weights(weight_path: str) -> AobaZeroNet:
    """Load AobaZero text weights into PyTorch model."""
    lines = parse_weight_file(weight_path)
    model = AobaZeroNet()
    model.eval()

    idx = 0

    # Initial conv + bn
    idx = load_conv_bn(lines, idx, model.conv_initial, model.bn_initial)

    # 19 residual blocks
    for block in model.residual_blocks:
        idx = load_conv_bn(lines, idx, block.conv1, block.bn1)
        idx = load_conv_bn(lines, idx, block.conv2, block.bn2)

    # Policy head: conv1 + bn + conv2
    idx = load_conv_bn(lines, idx, model.policy_conv1, model.policy_bn1)
    idx = load_conv_only(lines, idx, model.policy_conv2)

    # Value head: conv1 + bn + fc1 + fc2
    idx = load_conv_bn(lines, idx, model.value_conv1, model.value_bn1)
    idx = load_fc(lines, idx, model.value_fc1)
    idx = load_fc(lines, idx, model.value_fc2)

    assert idx == 170, f"Expected to consume 170 lines, consumed {idx}"
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert AobaZero text weights to PyTorch")
    parser.add_argument("input", help="Path to AobaZero text weight file (.txt)")
    parser.add_argument("output", help="Path to save PyTorch model (.pt)")
    args = parser.parse_args()

    model = convert_weights(args.input)
    torch.save(model.state_dict(), args.output)
    print(f"Converted: {args.input} -> {args.output}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Quick sanity check: random input
    with torch.no_grad():
        x = torch.randn(1, 362, 9, 9)
        policy, value = model(x)
        print(f"Sanity check - policy shape: {policy.shape}, value shape: {value.shape}")
        print(f"  value: {value.item():.4f}")


if __name__ == "__main__":
    main()
