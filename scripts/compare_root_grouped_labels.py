#!/usr/bin/env python3
"""Compare aligned teacher scores before and after grouped-data relabeling."""

import argparse
import json
from pathlib import Path

import cshogi
import numpy as np


RECORD_BYTES = 40
SCORE_DTYPE = np.dtype({"names": ["score"], "formats": ["<i2"],
                        "offsets": [32], "itemsize": RECORD_BYTES})


def quantiles(values):
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 1.0)
    }


def correlation(first, second):
    if first.size < 2 or first.std() == 0 or second.std() == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def black_perspective_scores(path, side_to_move_scores):
    records = np.memmap(path, dtype=np.uint8, mode="r").reshape(-1, RECORD_BYTES)
    board = cshogi.Board()
    packed = np.zeros(1, dtype=cshogi.PackedSfen)
    output = side_to_move_scores.copy()
    for index, record in enumerate(records):
        packed[0]["sfen"] = record[:32]
        board.set_psfen(packed)
        if board.turn != cshogi.BLACK:
            output[index] = -output[index]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first_meta = json.loads((args.first / "metadata.json").read_text())
    second_meta = json.loads((args.second / "metadata.json").read_text())
    roots = int(first_meta["num_roots"])
    group_size = int(first_meta["group_size"])
    if (roots, group_size) != (
        int(second_meta["num_roots"]), int(second_meta["group_size"])
    ):
        raise ValueError("datasets have different root/group dimensions")
    if (args.first / "roots.bin").read_bytes() != (
        args.second / "roots.bin"
    ).read_bytes():
        raise ValueError("root records differ")
    count = roots * group_size
    first = np.memmap(
        args.first / "leaves.bin", dtype=SCORE_DTYPE, mode="r", shape=(count,)
    )["score"].astype(np.float64)
    second = np.memmap(
        args.second / "leaves.bin", dtype=SCORE_DTYPE, mode="r", shape=(count,)
    )["score"].astype(np.float64)
    delta = second - first
    first_root_mean = first.reshape(roots, group_size).mean(axis=1)
    second_root_mean = second.reshape(roots, group_size).mean(axis=1)
    first_black = black_perspective_scores(args.first / "leaves.bin", first)
    second_black = black_perspective_scores(args.second / "leaves.bin", second)
    first_black_root_mean = first_black.reshape(roots, group_size).mean(axis=1)
    second_black_root_mean = second_black.reshape(roots, group_size).mean(axis=1)
    nonmate = (np.abs(first) < 31000) & (np.abs(second) < 31000)
    both_mate = (np.abs(first) >= 31000) & (np.abs(second) >= 31000)
    output = {
        "roots": roots,
        "group_size": group_size,
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "exact_score_match": float(np.mean(first == second)),
        "sign_match": float(np.mean(np.sign(first) == np.sign(second))),
        "score_correlation": correlation(first, second),
        "black_perspective_score_correlation": correlation(
            first_black, second_black
        ),
        "black_perspective_sign_match": float(
            np.mean(np.sign(first_black) == np.sign(second_black))
        ),
        "black_perspective_root_mean_correlation": correlation(
            first_black_root_mean, second_black_root_mean
        ),
        "black_perspective_nonmate_score_correlation": correlation(
            first_black[nonmate], second_black[nonmate]
        ),
        "nonmate_score_correlation": correlation(
            first[nonmate], second[nonmate]
        ),
        "root_mean_score_correlation": correlation(
            first_root_mean, second_root_mean
        ),
        "mean_absolute_delta": float(np.mean(np.abs(delta))),
        "delta_quantiles": quantiles(delta),
        "first_mate_like_fraction": float(np.mean(np.abs(first) >= 31000)),
        "second_mate_like_fraction": float(np.mean(np.abs(second) >= 31000)),
        "both_mate_like_fraction": float(np.mean(both_mate)),
        "both_mate_sign_match": (
            float(np.mean(np.sign(first[both_mate]) == np.sign(second[both_mate])))
            if np.any(both_mate) else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
