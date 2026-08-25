#!/usr/bin/env python3
"""Widen a DNN gate adapter while preserving its logits exactly."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


FC1_WEIGHT = "model.adapter.fc1.weight"
FC1_BIAS = "model.adapter.fc1.bias"
FC2_WEIGHT = "model.adapter.fc2.weight"


def widen_adapter_state_dict(state_dict, hidden_dim):
    fc1_weight = state_dict[FC1_WEIGHT]
    fc1_bias = state_dict[FC1_BIAS]
    fc2_weight = state_dict[FC2_WEIGHT]
    old_hidden = int(fc1_weight.shape[0])
    if hidden_dim <= old_hidden or hidden_dim % old_hidden:
        raise ValueError("new hidden dimension must be a larger multiple of the old one")
    repeats = hidden_dim // old_hidden
    state_dict[FC1_WEIGHT] = fc1_weight.repeat(repeats, 1)
    state_dict[FC1_BIAS] = fc1_bias.repeat(repeats)
    state_dict[FC2_WEIGHT] = fc2_weight.repeat(1, repeats) / repeats
    return {"old_hidden_dim": old_hidden, "new_hidden_dim": hidden_dim, "repeats": repeats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")
    summary = widen_adapter_state_dict(checkpoint["state_dict"], args.hidden_dim)
    checkpoint.setdefault("research_transform", {})["widen_adapter"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"widened adapter: {summary}; output={args.output}")


if __name__ == "__main__":
    main()
