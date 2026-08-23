#!/usr/bin/env python3
"""Split flat grouped pairs into one root and K qsearch leaves per group."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


RECORD_BYTES = 40
PAIR_BYTES = 80
GAME_PLY_OFFSET = 36


def packed_game_ply(records: np.ndarray) -> np.ndarray:
    return (
        records[:, GAME_PLY_OFFSET : GAME_PLY_OFFSET + 2]
        .copy()
        .view("<u2")
        .reshape(-1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--chunk-groups", type=int, default=65536)
    args = parser.parse_args()

    if args.group_size <= 1 or args.chunk_groups <= 0:
        parser.error("group-size must exceed 1 and chunk-groups must be positive")
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    group_bytes = args.group_size * PAIR_BYTES
    input_size = input_path.stat().st_size
    if input_size == 0 or input_size % group_bytes != 0:
        raise ValueError(
            f"input size must be a non-zero multiple of {group_bytes}: {input_size}"
        )
    num_groups = input_size // group_bytes
    output_dir.mkdir(parents=True, exist_ok=True)
    roots_tmp = output_dir / "roots.bin.tmp"
    leaves_tmp = output_dir / "leaves.bin.tmp"
    offsets_tmp = output_dir / "offsets.npy.tmp"
    offsets = np.lib.format.open_memmap(
        offsets_tmp,
        mode="w+",
        dtype=np.uint8,
        shape=(num_groups, args.group_size),
    )

    processed = 0
    with input_path.open("rb") as source, roots_tmp.open("wb") as roots_file, \
            leaves_tmp.open("wb") as leaves_file:
        while processed < num_groups:
            take = min(args.chunk_groups, num_groups - processed)
            raw = source.read(take * group_bytes)
            if len(raw) != take * group_bytes:
                raise EOFError("short read while splitting grouped pairs")
            pairs = np.frombuffer(raw, dtype=np.uint8).reshape(
                take, args.group_size, PAIR_BYTES
            )
            roots = pairs[:, :, :RECORD_BYTES]
            if not np.all(roots == roots[:, :1, :]):
                raise ValueError(f"root mismatch inside group near index {processed}")
            root_records = np.ascontiguousarray(roots[:, 0, :])
            leaf_records = np.ascontiguousarray(pairs[:, :, RECORD_BYTES:])
            root_ply = packed_game_ply(root_records).astype(np.int32)
            flat_leaves = leaf_records.reshape(-1, RECORD_BYTES)
            leaf_ply = packed_game_ply(flat_leaves).reshape(
                take, args.group_size
            ).astype(np.int32)
            delta = leaf_ply - root_ply[:, None]
            if np.any(delta < 1) or np.any(delta > 50):
                bad = np.argwhere((delta < 1) | (delta > 50))[0]
                raise ValueError(
                    "invalid root-to-leaf offset at group/member "
                    f"{processed + int(bad[0])}/{int(bad[1])}: "
                    f"{int(delta[tuple(bad)])}"
                )
            roots_file.write(root_records.tobytes())
            leaves_file.write(leaf_records.tobytes())
            offsets[processed : processed + take] = delta.astype(np.uint8)
            processed += take

        roots_file.flush()
        os.fsync(roots_file.fileno())
        leaves_file.flush()
        os.fsync(leaves_file.fileno())
    offsets.flush()
    del offsets
    os.replace(roots_tmp, output_dir / "roots.bin")
    os.replace(leaves_tmp, output_dir / "leaves.bin")
    os.replace(offsets_tmp, output_dir / "offsets.npy")
    metadata = {
        "format": "root-grouped-paired-v1",
        "num_roots": num_groups,
        "group_size": args.group_size,
        "record_bytes": RECORD_BYTES,
        "offset_sampling": "uniform-with-replacement",
        "offset_min": 1,
        "offset_max": 50,
        "qsearch": True,
        "shuffle_seed": args.seed,
        "source": str(Path(args.source).resolve()),
    }
    with (output_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
