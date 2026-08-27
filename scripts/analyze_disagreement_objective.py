#!/usr/bin/env python3
"""Select one radius per expert under utility constraints, then test coverage."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def load_aligned_details(specifications):
    scales = []
    rows_by_scale = []
    roots = None
    for scale_text, path_text in specifications:
        rows = [json.loads(line) for line in Path(path_text).read_text().splitlines()]
        current = [int(row["source_root"]) for row in rows]
        if roots is None:
            roots = current
        elif current != roots:
            raise ValueError("detail files do not contain the same ordered roots")
        scales.append(float(scale_text))
        rows_by_scale.append(rows)
    order = np.argsort(scales)
    return np.asarray(scales)[order], [rows_by_scale[index] for index in order]


def candidate_arrays(rows_by_scale, pooled_rows=None):
    moves = np.asarray(
        [[row["moves"] for row in rows] for rows in rows_by_scale], dtype=object
    ).transpose(1, 0, 2)
    if pooled_rows is None:
        utilities = np.asarray(
            [[row["utilities_cp"] for row in rows] for rows in rows_by_scale],
            dtype=np.float64,
        ).transpose(1, 0, 2)
    else:
        if len(pooled_rows) != moves.shape[0]:
            raise ValueError("pooled utility rows have a different length")
        utilities = np.empty(moves.shape, dtype=np.float64)
        for root, pooled in enumerate(pooled_rows):
            if int(pooled["source_root"]) != int(rows_by_scale[0][root]["source_root"]):
                raise ValueError("pooled utility roots are not aligned")
            scores = pooled["unique_move_scores_cp"]
            for radius in range(moves.shape[1]):
                for candidate in range(moves.shape[2]):
                    utilities[root, radius, candidate] = scores[moves[root, radius, candidate]]
    if moves.shape[2] != 9:
        raise ValueError("expected baseline plus eight expert-axis candidates")
    if not np.all(moves[:, :, 0] == moves[:, :1, 0]):
        raise ValueError("baseline moves differ across radius runs")
    if not np.all(utilities[:, :, 0] == utilities[:, :1, 0]):
        raise ValueError("baseline utilities differ across radius runs")
    return moves, utilities


def metrics_for_vector(moves, utilities, radius_indices, row_slice):
    row_ids = np.arange(moves.shape[0])[row_slice]
    selected_moves = [moves[row_ids, 0, 0]]
    selected_utilities = [utilities[row_ids, 0, 0]]
    for expert, radius_index in enumerate(radius_indices):
        selected_moves.append(moves[row_ids, radius_index, expert + 1])
        selected_utilities.append(utilities[row_ids, radius_index, expert + 1])
    move_matrix = np.stack(selected_moves, axis=1)
    utility_matrix = np.stack(selected_utilities, axis=1)
    gains = utility_matrix.max(axis=1) - utility_matrix[:, 0]
    diversity = np.asarray([len(set(row)) for row in move_matrix], dtype=np.float64)
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


def select_radius_vector(moves, utilities, selection_roots, fixed_index):
    selection = slice(0, selection_roots)
    fixed_vector = (fixed_index,) * 8
    fixed = summarize(metrics_for_vector(moves, utilities, fixed_vector, selection))
    best = None
    tolerance = 1e-12
    for vector in itertools.product(range(moves.shape[1]), repeat=8):
        metrics = summarize(metrics_for_vector(moves, utilities, vector, selection))
        if (
            metrics["teacher_changed_fraction"] + tolerance
            < fixed["teacher_changed_fraction"]
            or metrics["oracle_gain_cp_mean"] + tolerance
            < fixed["oracle_gain_cp_mean"]
        ):
            continue
        key = (
            metrics["candidate_move_diversity_mean"],
            metrics["teacher_changed_fraction"],
            metrics["oracle_gain_cp_mean"],
            tuple(-item for item in vector),
        )
        if best is None or key > best[0]:
            best = (key, vector, metrics)
    if best is None:
        return fixed_vector, fixed, fixed
    return best[1], fixed, best[2]


def bootstrap_difference(values, seed, iterations=20000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        sample = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[sample].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "bootstrap_95": [float(item) for item in np.quantile(means, [0.025, 0.975])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", nargs=2, action="append", required=True)
    parser.add_argument("--selection-roots", type=int, required=True)
    parser.add_argument("--pooled-utilities", type=Path)
    parser.add_argument("--fixed-radius", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scales, rows_by_scale = load_aligned_details(args.details)
    pooled_rows = None
    if args.pooled_utilities is not None:
        pooled_rows = [
            json.loads(line) for line in args.pooled_utilities.read_text().splitlines()
        ]
    moves, utilities = candidate_arrays(rows_by_scale, pooled_rows)
    if not 0 < args.selection_roots < moves.shape[0]:
        parser.error("selection roots must leave a non-empty held-out split")
    matches = np.flatnonzero(np.isclose(scales, args.fixed_radius))
    if len(matches) != 1:
        parser.error("fixed radius is absent or duplicated")
    fixed_index = int(matches[0])
    selected, selection_fixed, selection_selected = select_radius_vector(
        moves, utilities, args.selection_roots, fixed_index
    )
    held_out = slice(args.selection_roots, None)
    fixed_metrics = metrics_for_vector(
        moves, utilities, (fixed_index,) * 8, held_out
    )
    selected_metrics = metrics_for_vector(moves, utilities, selected, held_out)
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
    output = {
        "format": "disagreement-objective-analysis-v1",
        "scales": scales.tolist(),
        "selection_roots": args.selection_roots,
        "held_out_roots": moves.shape[0] - args.selection_roots,
        "fixed_radius": args.fixed_radius,
        "selected_radius_by_expert": [float(scales[index]) for index in selected],
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
