"""Extract a subset from floodgate.hcpe binary and save as JSONL files."""

import argparse
import json
import os

import cshogi
import numpy as np


def extract_record(hcpe):
    """Extract fields from a single HCPE record."""
    board = cshogi.Board()
    board.set_hcp(hcpe["hcp"])
    move = board.move_from_move16(hcpe["bestMove16"])
    return {
        "sfen": board.sfen(),
        "bestmove": cshogi.move_to_usi(move),
        "turn": int(board.turn),
        "gameResult": int(hcpe["gameResult"]),
        "eval": int(hcpe["eval"]),
    }


def write_jsonl(records, path):
    """Write records as JSONL file."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract subset from floodgate.hcpe and save as JSONL"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to floodgate.hcpe binary file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--count-per-split",
        type=int,
        default=1000,
        help="Number of records per split (default: 1000)",
    )
    args = parser.parse_args()

    # Load HCPE binary
    print(f"Loading {args.input}...")
    ds = np.fromfile(args.input, dtype=cshogi.HuffmanCodedPosAndEval)
    print(f"Total records: {ds.shape[0]}")

    # Sample indices without replacement
    total_needed = args.count_per_split * 3
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(ds.shape[0], size=total_needed, replace=False)

    # Split into train/val/test
    n = args.count_per_split
    splits = {
        "train": indices[:n],
        "val": indices[n : 2 * n],
        "test": indices[2 * n : 3 * n],
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, split_indices in splits.items():
        print(f"Extracting {split_name} ({len(split_indices)} records)...")
        records = [extract_record(ds[i]) for i in split_indices]
        path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        write_jsonl(records, path)
        print(f"  Written to {path}")

    print("Done!")


if __name__ == "__main__":
    main()
