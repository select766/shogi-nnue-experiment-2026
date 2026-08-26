#!/usr/bin/env python3
"""Append zero-initialized auxiliary inputs while preserving adapter logits."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


FC1_WEIGHT = "model.adapter.fc1.weight"


def expand_adapter_input(state_dict, auxiliary_dim):
    if auxiliary_dim <= 0:
        raise ValueError("auxiliary dimension must be positive")
    weight = state_dict[FC1_WEIGHT]
    expanded = weight.new_zeros((weight.shape[0], weight.shape[1] + auxiliary_dim))
    expanded[:, : weight.shape[1]] = weight
    state_dict[FC1_WEIGHT] = expanded
    return {
        "old_input_dim": int(weight.shape[1]),
        "new_input_dim": int(expanded.shape[1]),
        "auxiliary_dim": auxiliary_dim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--auxiliary-dim", type=int, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    summary = expand_adapter_input(checkpoint["state_dict"], args.auxiliary_dim)
    checkpoint.setdefault("research_transform", {})["expand_adapter_input"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"expanded adapter input: {summary}; output={args.output}")


if __name__ == "__main__":
    main()
