#!/usr/bin/env python3
"""Scale expert parameter deviations around their exact expert mean."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


PREFIX = "model.nnue_experts."
PARAMETERS = {
    "input_weight",
    "input_bias",
    "l1_weight",
    "l1_bias",
    "l2_weight",
    "l2_bias",
    "output_weight",
    "output_bias",
}


def scale_state_dict(state_dict, scale):
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    scaled = []
    max_mean_error = 0.0
    for name, tensor in state_dict.items():
        parameter = name.removeprefix(PREFIX)
        if not name.startswith(PREFIX) or parameter not in PARAMETERS:
            continue
        if tensor.ndim < 1 or tensor.shape[0] < 2 or not tensor.is_floating_point():
            raise ValueError(f"unexpected expert tensor: {name} {tuple(tensor.shape)}")
        original_mean = tensor.mean(dim=0, keepdim=True)
        transformed = original_mean + scale * (tensor - original_mean)
        error = float((transformed.mean(dim=0) - original_mean.squeeze(0)).abs().max())
        max_mean_error = max(max_mean_error, error)
        state_dict[name] = transformed
        scaled.append(name)
    if not scaled:
        raise ValueError("checkpoint contains no expert parameter tensors")
    return {"scaled_parameters": scaled, "max_expert_mean_error": max_mean_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")
    statistics = scale_state_dict(checkpoint["state_dict"], args.scale)
    checkpoint.setdefault("research_transform", {})["expert_deviation_scale"] = args.scale
    checkpoint["research_transform"].update(statistics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(
        f"scaled {len(statistics['scaled_parameters'])} tensors; "
        f"max mean error={statistics['max_expert_mean_error']:.9g}; output={args.output}"
    )


if __name__ == "__main__":
    main()
