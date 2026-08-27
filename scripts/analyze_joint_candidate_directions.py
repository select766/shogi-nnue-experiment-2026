#!/usr/bin/env python3
"""Select eight directions from a fixed axis-plus-joint candidate pool."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_disagreement_objective import bootstrap_difference


def load_pool(details_path, pooled_path):
    rows = [json.loads(line) for line in Path(details_path).read_text().splitlines()]
    pooled = [json.loads(line) for line in Path(pooled_path).read_text().splitlines()]
    if not rows or len(rows) != len(pooled):
        raise ValueError("details and pooled utility rows are empty or misaligned")
    names = rows[0].get("candidate_names")
    if not isinstance(names, list) or names[0] != "base":
        raise ValueError("details do not contain named candidates with a base")
    moves = np.asarray([row["moves"] for row in rows], dtype=object)
    if moves.shape != (len(rows), len(names)):
        raise ValueError("candidate names and move matrix differ in shape")
    utilities = np.empty(moves.shape, dtype=np.float64)
    move_codes = np.empty(moves.shape, dtype=np.int16)
    for root, (row, score_row) in enumerate(zip(rows, pooled)):
        if row.get("candidate_names") != names:
            raise ValueError("candidate names differ between roots")
        if (
            int(row["source_root"]) != int(score_row["source_root"])
            or row["sfen"] != score_row["sfen"]
        ):
            raise ValueError("details and pooled utilities are not root-aligned")
        scores = score_row["unique_move_scores_cp"]
        codes = {}
        for candidate, move in enumerate(moves[root]):
            utilities[root, candidate] = scores[move]
            move_codes[root, candidate] = codes.setdefault(move, len(codes))
    return names, move_codes, utilities


def metrics_for_directions(move_codes, utilities, directions, row_slice=slice(None)):
    rows = np.arange(move_codes.shape[0])[row_slice]
    columns = np.asarray((0,) + tuple(index + 1 for index in directions))
    selected_moves = move_codes[rows[:, None], columns]
    selected_utilities = utilities[rows[:, None], columns]
    gains = selected_utilities.max(axis=1) - selected_utilities[:, 0]
    diversity = np.asarray(
        [len(np.unique(root_moves)) for root_moves in selected_moves], dtype=np.float64
    )
    return {
        "teacher_changed": gains >= 10.0,
        "gains": gains,
        "diversity": diversity,
    }


def summarize(metrics):
    return {
        "teacher_changed_fraction": float(metrics["teacher_changed"].mean()),
        "oracle_gain_cp_mean": float(metrics["gains"].mean()),
        "candidate_move_diversity_mean": float(metrics["diversity"].mean()),
    }


def select_direction_set(move_codes, utilities, fixed, selection_roots, select_count=8):
    """Exhaustively maximize diversity subject to fixed utility constraints."""
    if len(fixed) != select_count:
        raise ValueError("fixed direction count differs from deployed candidate count")
    selection_moves = move_codes[:selection_roots]
    selection_utilities = utilities[:selection_roots]
    fixed_metrics = metrics_for_directions(
        selection_moves, selection_utilities, fixed
    )
    fixed_changed = int(fixed_metrics["teacher_changed"].sum())
    fixed_gain = float(fixed_metrics["gains"].sum())
    best = None
    direction_count = move_codes.shape[1] - 1
    combinations = itertools.combinations(range(direction_count), select_count)
    while True:
        chunk_list = list(itertools.islice(combinations, 1000))
        if not chunk_list:
            break
        chunk = np.asarray(chunk_list, dtype=np.int16)
        columns = chunk + 1
        candidate_utilities = selection_utilities[:, columns]
        gains = np.maximum(
            selection_utilities[:, :1], candidate_utilities.max(axis=2)
        ) - selection_utilities[:, :1]
        changed_counts = (gains >= 10.0).sum(axis=0)
        gain_sums = gains.sum(axis=0)

        candidate_moves = selection_moves[:, columns]
        diversity_sums = np.ones(len(chunk), dtype=np.int64) * selection_roots
        for position in range(select_count):
            current = candidate_moves[:, :, position]
            is_new = current != selection_moves[:, :1]
            for previous in range(position):
                is_new &= current != candidate_moves[:, :, previous]
            diversity_sums += is_new.sum(axis=0)

        eligible = np.flatnonzero(
            (changed_counts >= fixed_changed) & (gain_sums + 1e-12 >= fixed_gain)
        )
        for index in eligible:
            directions = tuple(int(item) for item in chunk[index])
            key = (
                int(diversity_sums[index]),
                int(changed_counts[index]),
                float(gain_sums[index]),
                tuple(-item for item in directions),
            )
            if best is None or key > best[0]:
                best = (key, directions)
    if best is None:
        return tuple(fixed), summarize(fixed_metrics), summarize(fixed_metrics)
    selected = best[1]
    return (
        selected,
        summarize(fixed_metrics),
        summarize(metrics_for_directions(selection_moves, selection_utilities, selected)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--pooled-utilities", type=Path, required=True)
    parser.add_argument("--bias-file", type=Path, required=True)
    parser.add_argument("--selection-roots", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names, move_codes, utilities = load_pool(args.details, args.pooled_utilities)
    if not 0 < args.selection_roots < len(move_codes):
        parser.error("selection roots must leave a non-empty held-out split")
    direction_names = names[1:]
    try:
        fixed = tuple(direction_names.index(f"axis_{expert}") for expert in range(8))
    except ValueError as error:
        parser.error(f"axis control is incomplete: {error}")
    selected, selection_fixed, selection_selected = select_direction_set(
        move_codes, utilities, fixed, args.selection_roots
    )

    held_out = slice(args.selection_roots, None)
    fixed_metrics = metrics_for_directions(move_codes, utilities, fixed, held_out)
    selected_metrics = metrics_for_directions(move_codes, utilities, selected, held_out)
    changed_difference = bootstrap_difference(
        selected_metrics["teacher_changed"].astype(float)
        - fixed_metrics["teacher_changed"].astype(float),
        20260827,
    )
    diversity_difference = bootstrap_difference(
        selected_metrics["diversity"] - fixed_metrics["diversity"], 20260828
    )
    gain_difference = float(
        (selected_metrics["gains"] - fixed_metrics["gains"]).mean()
    )
    if (
        changed_difference["mean"] > 0
        and diversity_difference["mean"] > 0
        and changed_difference["bootstrap_95"][0] >= 0
        and diversity_difference["bootstrap_95"][0] >= 0
        and gain_difference >= 0
    ):
        classification = "supported"
    elif (
        changed_difference["bootstrap_95"][1] <= 0
        or diversity_difference["bootstrap_95"][1] <= 0
        or gain_difference < 0
    ):
        classification = "rejected"
    else:
        classification = "measured"

    configured = json.loads(args.bias_file.read_text())["directions"]
    bias_by_name = {item["name"]: item["bias"] for item in configured}
    output = {
        "format": "joint-candidate-directions-analysis-v1",
        "selection_roots": args.selection_roots,
        "held_out_roots": len(move_codes) - args.selection_roots,
        "pool_direction_count": len(direction_names),
        "fixed_directions": [direction_names[index] for index in fixed],
        "selected_directions": [direction_names[index] for index in selected],
        "selected_biases": [bias_by_name[direction_names[index]] for index in selected],
        "selection": {"fixed": selection_fixed, "selected": selection_selected},
        "held_out": {
            "fixed": summarize(fixed_metrics),
            "selected": summarize(selected_metrics),
        },
        "selected_minus_fixed": {
            "teacher_changed_fraction": changed_difference,
            "oracle_gain_cp_mean": gain_difference,
            "candidate_move_diversity": diversity_difference,
        },
        "classification": classification,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
