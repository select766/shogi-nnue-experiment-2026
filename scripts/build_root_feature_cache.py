#!/usr/bin/env python3
"""Build deployment-cheap root features from PackedSfenValue records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import cshogi
import numpy as np


RECORD_BYTES = 40
PSFEN_BYTES = 32


def phase_features(board, game_ply):
    pieces = np.asarray(board.pieces, dtype=np.int16)
    occupied = pieces[pieces != 0]
    types = occupied & 15
    colors = occupied >= 16
    non_king = types != cshogi.KING
    hands = np.asarray(board.pieces_in_hand, dtype=np.float64)
    promoted = np.isin(types, [9, 10, 11, 12, 13, 14]).sum()
    black_material = np.sum(non_king & ~colors) + hands[0].sum()
    white_material = np.sum(non_king & colors) + hands[1].sum()
    black_king = int(np.flatnonzero(pieces == cshogi.BKING)[0])
    white_king = int(np.flatnonzero(pieces == cshogi.WKING)[0])
    king_distance = abs(black_king // 9 - white_king // 9) + abs(
        black_king % 9 - white_king % 9
    )
    return np.asarray(
        [
            min(max(game_ply, 1), 256) / 256.0,
            float(non_king.sum()) / 38.0,
            float(hands.sum()) / 38.0,
            float(promoted) / 8.0,
            min(len(list(board.legal_moves)), 256) / 256.0,
            (black_material - white_material) / 38.0,
            king_distance / 16.0,
        ],
        dtype=np.float32,
    )


def select_features(all_features, feature_set):
    if feature_set == "ply":
        return all_features[:1]
    if feature_set == "phase":
        return all_features[1:]
    if feature_set == "combined":
        return all_features
    raise ValueError(f"unknown feature set: {feature_set}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots-file", type=Path, required=True)
    parser.add_argument("--feature-set", choices=["ply", "phase", "combined"], required=True)
    parser.add_argument("--max-roots", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    size = args.roots_file.stat().st_size
    available = size // RECORD_BYTES
    if size % RECORD_BYTES or not 0 < args.max_roots <= available:
        parser.error("requested roots are outside an aligned records file")
    board = cshogi.Board()
    psfen = np.zeros(1, dtype=cshogi.PackedSfen)
    rows = []
    with args.roots_file.open("rb") as source:
        for _ in range(args.max_roots):
            record = source.read(RECORD_BYTES)
            psfen[0]["sfen"] = np.frombuffer(record[:PSFEN_BYTES], dtype=np.uint8)
            board.set_psfen(psfen)
            game_ply = struct.unpack_from("<H", record, 36)[0]
            rows.append(select_features(phase_features(board, game_ply), args.feature_set))
    output = np.stack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as target:
        np.save(target, output)
    temporary.replace(args.output)
    metadata = {
        "format": "root-feature-cache-v1",
        "feature_set": args.feature_set,
        "features": {
            "ply": ["game_ply"],
            "phase": [
                "board_piece_fraction", "hand_piece_fraction", "promoted_fraction",
                "legal_move_fraction", "material_count_balance", "king_distance",
            ],
            "combined": [
                "game_ply", "board_piece_fraction", "hand_piece_fraction",
                "promoted_fraction", "legal_move_fraction", "material_count_balance",
                "king_distance",
            ],
        }[args.feature_set],
        "roots": args.max_roots,
        "dimension": int(output.shape[1]),
        "source": str(args.roots_file.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
