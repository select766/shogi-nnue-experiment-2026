#!/usr/bin/env python3
"""Copy groups whose roots occur in an ordered target-root subsequence."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


RECORD_BYTES = 40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-roots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.source / "metadata.json").read_text())
    group_size = int(metadata["group_size"])
    source_count = int(metadata["num_roots"])
    target_size = args.target_roots.stat().st_size
    if target_size == 0 or target_size % RECORD_BYTES:
        raise ValueError("target roots must be a non-empty 40-byte record file")
    target_count = target_size // RECORD_BYTES
    args.output.mkdir(parents=True, exist_ok=True)
    root_tmp = args.output / "roots.bin.tmp"
    leaf_tmp = args.output / "leaves.bin.tmp"
    source_offsets_path = args.source / "offsets.npy"
    kept_offsets = [] if source_offsets_path.exists() else None
    source_offsets = (
        np.load(source_offsets_path, mmap_mode="r")
        if source_offsets_path.exists()
        else None
    )
    source_index = 0
    matched = 0
    with (args.source / "roots.bin").open("rb") as source_roots, \
            (args.source / "leaves.bin").open("rb") as source_leaves, \
            args.target_roots.open("rb") as targets, root_tmp.open("wb") as roots_out, \
            leaf_tmp.open("wb") as leaves_out:
        target = targets.read(RECORD_BYTES)
        while target and source_index < source_count:
            root = source_roots.read(RECORD_BYTES)
            leaves = source_leaves.read(group_size * RECORD_BYTES)
            if len(root) != RECORD_BYTES or len(leaves) != group_size * RECORD_BYTES:
                raise EOFError("source dataset ended before metadata count")
            if root == target:
                roots_out.write(root)
                leaves_out.write(leaves)
                if kept_offsets is not None:
                    kept_offsets.append(np.asarray(source_offsets[source_index]))
                matched += 1
                target = targets.read(RECORD_BYTES)
            source_index += 1
        if target:
            raise ValueError(
                f"target roots are not an ordered source subsequence; matched {matched}/{target_count}"
            )
        for file in (roots_out, leaves_out):
            file.flush()
            os.fsync(file.fileno())
    os.replace(root_tmp, args.output / "roots.bin")
    os.replace(leaf_tmp, args.output / "leaves.bin")
    if kept_offsets is not None:
        np.save(args.output / "offsets.npy", np.asarray(kept_offsets))
    output_metadata = dict(metadata)
    output_metadata["num_roots"] = target_count
    output_metadata["subset"] = {
        "target_roots": str(args.target_roots.resolve()),
        "source_roots_scanned": source_index,
        "matching": "ordered full 40-byte root records",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n"
    )
    print(json.dumps(output_metadata["subset"], indent=2))


if __name__ == "__main__":
    main()
