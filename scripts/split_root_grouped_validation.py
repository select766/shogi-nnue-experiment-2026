#!/usr/bin/env python3
"""Split a 2K-leaf validation set into two K-leaf sets with shared roots."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


RECORD_BYTES = 40


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-a", required=True)
    parser.add_argument("--output-b", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    with (source / "metadata.json").open() as file:
        metadata = json.load(file)
    group_size = int(metadata["group_size"])
    if group_size % 2 != 0:
        raise ValueError("input group size must be even")
    subgroup_size = group_size // 2
    num_roots = int(metadata["num_roots"])
    leaves = np.memmap(
        source / "leaves.bin",
        mode="r",
        dtype=np.uint8,
        shape=(num_roots, group_size, RECORD_BYTES),
    )
    offsets = np.load(source / "offsets.npy", mmap_mode="r")
    roots = source / "roots.bin"
    for label, output_arg, leaf_slice in (
        ("A", args.output_a, slice(0, subgroup_size)),
        ("B", args.output_b, slice(subgroup_size, group_size)),
    ):
        output = Path(output_arg).resolve()
        output.mkdir(parents=True, exist_ok=True)
        roots_target = output / "roots.bin"
        if not roots_target.exists():
            os.link(roots, roots_target)
        with (output / "leaves.bin.tmp").open("wb") as file:
            for start in range(0, num_roots, 65536):
                stop = min(num_roots, start + 65536)
                file.write(
                    np.ascontiguousarray(leaves[start:stop, leaf_slice]).tobytes()
                )
            file.flush()
            os.fsync(file.fileno())
        os.replace(output / "leaves.bin.tmp", output / "leaves.bin")
        np.save(output / "offsets.npy", np.asarray(offsets[:, leaf_slice]))
        subset_metadata = dict(metadata)
        subset_metadata["group_size"] = subgroup_size
        subset_metadata["validation_subset"] = label
        subset_metadata["parent"] = str(source)
        with (output / "metadata.json").open("w") as file:
            json.dump(subset_metadata, file, indent=2)
            file.write("\n")
        print(f"Wrote validation subset {label}: {output}")


if __name__ == "__main__":
    main()
