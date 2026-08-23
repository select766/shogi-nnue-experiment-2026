"""Extract a subset from floodgate.hcpe binary and save as JSONL files."""

import argparse
import hashlib
import json
import os

import cshogi
import numpy as np


def extract_record(hcpe, source_index=None):
    """Extract fields from a single HCPE record."""
    board = cshogi.Board()
    board.set_hcp(hcpe["hcp"])
    move = board.move_from_move16(hcpe["bestMove16"])
    record = {
        "sfen": board.sfen(),
        "bestmove": cshogi.move_to_usi(move),
        "turn": int(board.turn),
        "gameResult": int(hcpe["gameResult"]),
        "eval": int(hcpe["eval"]),
    }
    if source_index is not None:
        record["source_index"] = int(source_index)
    return record


def write_jsonl(records, path):
    """Write records as JSONL file."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_excluded_sfens(paths):
    excluded = set()
    for path in paths:
        with open(path) as file:
            for line in file:
                line = line.strip()
                if line:
                    excluded.add(json.loads(line)["sfen"])
    return excluded


def parse_splits(values, count_per_split):
    if not values:
        return [("train", count_per_split), ("val", count_per_split), ("test", count_per_split)]
    splits = []
    seen = set()
    for value in values:
        try:
            name, count_text = value.split("=", 1)
            count = int(count_text)
        except ValueError as error:
            raise ValueError(f"invalid --split {value!r}; expected NAME=COUNT") from error
        if not name or count <= 0 or name in seen:
            raise ValueError(f"invalid or duplicate --split {value!r}")
        seen.add(name)
        splits.append((name, count))
    return splits


def select_records(dataset, split_sizes, seed, excluded_sfens):
    """Select unique positions in deterministic shuffled order."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dataset.shape[0])
    selected = []
    used_sfens = set(excluded_sfens)
    total_needed = sum(count for _, count in split_sizes)
    for source_index in permutation:
        record = extract_record(dataset[source_index], source_index=source_index)
        if record["sfen"] in used_sfens:
            continue
        used_sfens.add(record["sfen"])
        selected.append(record)
        if len(selected) == total_needed:
            return selected
    raise ValueError(
        f"only {len(selected)} unique non-excluded positions available; "
        f"{total_needed} requested"
    )


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
        "--split",
        action="append",
        metavar="NAME=COUNT",
        help=(
            "Output split and size; repeat for multiple splits. When omitted, "
            "train/val/test use --count-per-split."
        ),
    )
    parser.add_argument(
        "--exclude-jsonl",
        action="append",
        default=[],
        help="JSONL dataset whose SFEN positions must be excluded; repeatable",
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

    split_sizes = parse_splits(args.split, args.count_per_split)
    excluded_sfens = load_excluded_sfens(args.exclude_jsonl)
    print(f"Excluded positions: {len(excluded_sfens)}")
    selected = select_records(ds, split_sizes, args.seed, excluded_sfens)

    os.makedirs(args.output_dir, exist_ok=True)

    offset = 0
    manifest_splits = {}
    for split_name, count in split_sizes:
        records = selected[offset : offset + count]
        offset += count
        print(f"Extracting {split_name} ({len(records)} records)...")
        path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        write_jsonl(records, path)
        print(f"  Written to {path}")
        manifest_splits[split_name] = {
            "count": len(records),
            "file": os.path.basename(path),
            "sha256": file_sha256(path),
        }

    manifest = {
        "format_version": 1,
        "source": {
            "path": args.input,
            "records": int(ds.shape[0]),
            "size_bytes": os.path.getsize(args.input),
            "sha256": file_sha256(args.input),
        },
        "seed": args.seed,
        "identity": "unique SFEN",
        "excluded": [
            {"path": path, "sha256": file_sha256(path)}
            for path in args.exclude_jsonl
        ],
        "splits": manifest_splits,
        "notes": {
            "game_ply": (
                "HCPE does not store game ply; the move number in decoded SFEN is "
                "not treated as game-ply metadata."
            )
        },
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Manifest written to {manifest_path}")

    print("Done!")


if __name__ == "__main__":
    main()
