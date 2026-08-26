#!/usr/bin/env python3
"""Assign one of eight root-predictable 32-ply expert roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RECORD_BYTES = 40


def roles_from_records(records, n_experts=8, phase_width=32):
    if records.ndim != 2 or records.shape[1] != RECORD_BYTES:
        raise ValueError("records must have shape (roots, 40)")
    ply = records[:, 36].astype(np.uint16) + (
        records[:, 37].astype(np.uint16) << 8
    )
    return np.minimum(np.maximum(ply, 1) - 1, n_experts * phase_width - 1) // phase_width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots-file", type=Path, required=True)
    parser.add_argument("--max-roots", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = np.memmap(args.roots_file, mode="r", dtype=np.uint8)
    if raw.size % RECORD_BYTES or not 0 < args.max_roots <= raw.size // RECORD_BYTES:
        parser.error("requested roots are outside an aligned records file")
    records = raw[: args.max_roots * RECORD_BYTES].reshape(-1, RECORD_BYTES)
    roles = roles_from_records(records).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as target:
        np.save(target, roles)
    temporary.replace(args.output)
    counts = np.bincount(roles, minlength=8)
    summary = {
        "format": "expert-phase-role-v1",
        "roots": len(roles),
        "phase_width_ply": 32,
        "counts": counts.tolist(),
        "fractions": (counts / len(roles)).tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
