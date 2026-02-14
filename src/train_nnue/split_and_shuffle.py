#!/usr/bin/env python3
"""
Split .bin files into train/val1-4/test splits by file unit.

Creates symlink directories for each split (to be used as input for shuffle_kifu).
Data leakage is avoided by splitting at the file level, not the record level.

Usage:
    uv run python -m train_nnue.split_and_shuffle

Output: ./dataset/split_v1/
"""

import os
import json
import random
import glob

SOURCE_DIR = "./dataset/tanuki-.nnue-pytorch-2024-07-30.1"
OUTPUT_BASE = "./dataset/split_v1"
SEED = 42

# Split sizes
SPLIT_SIZES = {
    "train": 916,
    "val1": 20,
    "val2": 20,
    "val3": 20,
    "val4": 20,
    "test": 20,
}


def main():
    # 1. List and sort .bin files
    bin_files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.bin")))
    print(f"Found {len(bin_files)} .bin files in {SOURCE_DIR}")

    expected_total = sum(SPLIT_SIZES.values())
    if len(bin_files) != expected_total:
        print(f"WARNING: Expected {expected_total} files, found {len(bin_files)}")
        if len(bin_files) < expected_total:
            raise ValueError(f"Not enough files: need {expected_total}, have {len(bin_files)}")

    # 2. Shuffle with fixed seed
    rng = random.Random(SEED)
    shuffled_files = list(bin_files)
    rng.shuffle(shuffled_files)

    # 3. Assign files to splits
    manifest = {}
    idx = 0
    for split_name, count in SPLIT_SIZES.items():
        manifest[split_name] = [os.path.basename(f) for f in shuffled_files[idx:idx + count]]
        idx += count

    # 4. Create symlink directories
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for split_name, filenames in manifest.items():
        link_dir = os.path.join(OUTPUT_BASE, f"input_{split_name}")
        os.makedirs(link_dir, exist_ok=True)

        for fname in filenames:
            src = os.path.join(SOURCE_DIR, fname)
            dst = os.path.join(link_dir, fname)
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        print(f"  {split_name}: {len(filenames)} files -> {link_dir}")

    # 5. Save manifest
    manifest_path = os.path.join(OUTPUT_BASE, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    # Verification
    all_files = set()
    for split_name, filenames in manifest.items():
        file_set = set(filenames)
        overlap = all_files & file_set
        if overlap:
            raise ValueError(f"Overlap found in {split_name}: {overlap}")
        all_files |= file_set
    print(f"Verification: {len(all_files)} unique files, no overlaps")


if __name__ == "__main__":
    main()
