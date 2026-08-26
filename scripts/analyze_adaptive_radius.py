#!/usr/bin/env python3
"""Fit a shallow root-only radius policy and evaluate it on held-out roots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cshogi
import numpy as np


def load_aligned_details(specifications):
    scales = []
    by_scale = []
    source_roots = None
    for scale_text, path_text in specifications:
        scale = float(scale_text)
        rows = [json.loads(line) for line in Path(path_text).read_text().splitlines()]
        roots = [int(row["source_root"]) for row in rows]
        if source_roots is None:
            source_roots = roots
        elif roots != source_roots:
            raise ValueError("detail files do not contain the same ordered roots")
        scales.append(scale)
        by_scale.append(rows)
    order = np.argsort(scales)
    return (
        np.asarray(scales, dtype=np.float64)[order],
        [by_scale[index] for index in order],
    )


def gate_features(row):
    gate = np.asarray(row["gates"][0], dtype=np.float64)
    ordered = np.sort(gate)
    return [
        float(-(gate * np.log(gate + 1e-12)).sum()),
        float(ordered[-1]),
        float(ordered[-1] - ordered[-2]),
    ]


def root_features(row):
    board = cshogi.Board(row["sfen"])
    pieces = np.asarray(board.pieces, dtype=np.int16)
    occupied = pieces[pieces != 0]
    piece_types = occupied & 15
    non_king = piece_types != cshogi.KING
    hands = np.asarray(board.pieces_in_hand, dtype=np.float64)
    promoted = np.isin(piece_types, [9, 10, 11, 12, 13, 14]).sum()
    return np.asarray(
        [
            min(board.move_number, 256) / 256.0,
            float(board.turn),
            float(non_king.sum()) / 38.0,
            float(hands.sum()) / 38.0,
            float(promoted) / 8.0,
            min(len(list(board.legal_moves)), 256) / 256.0,
            *gate_features(row),
        ],
        dtype=np.float64,
    )


FEATURE_NAMES = [
    "game_ply",
    "turn",
    "board_piece_fraction",
    "hand_piece_fraction",
    "promoted_fraction",
    "legal_move_fraction",
    "gate_entropy",
    "gate_max",
    "gate_margin",
]


def radius_arrays(by_scale):
    changed = np.asarray(
        [[row["oracle_gain_cp"] >= 10.0 for row in rows] for rows in by_scale],
        dtype=np.float64,
    ).T
    gains = np.asarray(
        [[row["oracle_gain_cp"] for row in rows] for rows in by_scale],
        dtype=np.float64,
    ).T
    diversity = np.asarray(
        [[len(set(row["moves"])) for row in rows] for rows in by_scale],
        dtype=np.float64,
    ).T
    return changed, gains, diversity


def best_labels(gains):
    # Scales are sorted ascending, so argmax provides the pre-registered small-radius tie break.
    return gains.argmax(axis=1)


def majority_label(labels, n_classes):
    return int(np.bincount(labels, minlength=n_classes).argmax())


def fit_tree(features, labels, max_depth, n_classes, indices=None, depth=0):
    if indices is None:
        indices = np.arange(len(labels))
    node = {"prediction": majority_label(labels[indices], n_classes)}
    if depth >= max_depth or np.all(labels[indices] == labels[indices[0]]):
        return node
    parent_error = np.sum(labels[indices] != node["prediction"])
    best = None
    for feature in range(features.shape[1]):
        values = np.unique(features[indices, feature])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        if len(thresholds) > 32:
            thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 19))
        for threshold in np.unique(thresholds):
            left = indices[features[indices, feature] <= threshold]
            right = indices[features[indices, feature] > threshold]
            if not len(left) or not len(right):
                continue
            error = (
                np.sum(labels[left] != majority_label(labels[left], n_classes))
                + np.sum(labels[right] != majority_label(labels[right], n_classes))
            )
            candidate = (int(error), feature, float(threshold), left, right)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None or best[0] >= parent_error:
        return node
    _, feature, threshold, left, right = best
    node.update(
        {
            "feature": feature,
            "threshold": threshold,
            "left": fit_tree(features, labels, max_depth, n_classes, left, depth + 1),
            "right": fit_tree(features, labels, max_depth, n_classes, right, depth + 1),
        }
    )
    return node


def predict_tree(tree, features):
    output = np.empty(len(features), dtype=np.int64)
    for row_index, row in enumerate(features):
        node = tree
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        output[row_index] = node["prediction"]
    return output


def policy_metrics(indices, changed, gains, diversity):
    rows = np.arange(len(indices))
    return {
        "teacher_changed_fraction": float(changed[rows, indices].mean()),
        "oracle_gain_cp_mean": float(gains[rows, indices].mean()),
        "candidate_move_diversity_mean": float(diversity[rows, indices].mean()),
    }


def select_fixed(changed, gains, diversity):
    keys = [
        (changed[:, index].mean(), diversity[:, index].mean(), gains[:, index].mean(), -index)
        for index in range(changed.shape[1])
    ]
    return max(range(len(keys)), key=lambda index: keys[index])


def select_depth(features, labels, changed, gains, diversity, n_classes):
    folds = np.arange(len(labels)) % 4
    candidates = []
    for depth in (1, 2, 3):
        predicted = np.empty(len(labels), dtype=np.int64)
        for fold in range(4):
            train = np.flatnonzero(folds != fold)
            validation = np.flatnonzero(folds == fold)
            tree = fit_tree(features, labels, depth, n_classes, train)
            predicted[validation] = predict_tree(tree, features[validation])
        metrics = policy_metrics(predicted, changed, gains, diversity)
        candidates.append((metrics["teacher_changed_fraction"], metrics["candidate_move_diversity_mean"], metrics["oracle_gain_cp_mean"], -depth, depth, metrics))
    return max(candidates) [4], max(candidates)[5]


def bootstrap_differences(adaptive_values, fixed_values, iterations=20_000, seed=42):
    differences = np.asarray(adaptive_values) - np.asarray(fixed_values)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        sample = rng.integers(0, len(differences), size=(count, len(differences)))
        means[start : start + count] = differences[sample].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "bootstrap_95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
    }


def readable_tree(node, scales):
    output = {"radius": float(scales[node["prediction"]])}
    if "feature" in node:
        output.update(
            {
                "feature": FEATURE_NAMES[node["feature"]],
                "threshold": node["threshold"],
                "left": readable_tree(node["left"], scales),
                "right": readable_tree(node["right"], scales),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", nargs=2, action="append", metavar=("SCALE", "PATH"), required=True)
    parser.add_argument("--selection-roots", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scales, by_scale = load_aligned_details(args.details)
    changed, gains, diversity = radius_arrays(by_scale)
    if not 0 < args.selection_roots < len(gains):
        parser.error("selection roots must leave a non-empty held-out split")
    features = np.stack([root_features(row) for row in by_scale[0]])
    labels = best_labels(gains)
    selection = slice(0, args.selection_roots)
    held_out = slice(args.selection_roots, None)
    depth, cross_validation = select_depth(
        features[selection], labels[selection], changed[selection], gains[selection], diversity[selection], len(scales)
    )
    tree = fit_tree(features[selection], labels[selection], depth, len(scales))
    fixed = select_fixed(changed[selection], gains[selection], diversity[selection])
    adaptive_indices = predict_tree(tree, features[held_out])
    fixed_indices = np.full(len(adaptive_indices), fixed, dtype=np.int64)
    oracle_indices = labels[held_out]
    adaptive = policy_metrics(adaptive_indices, changed[held_out], gains[held_out], diversity[held_out])
    fixed_metrics = policy_metrics(fixed_indices, changed[held_out], gains[held_out], diversity[held_out])
    oracle = policy_metrics(oracle_indices, changed[held_out], gains[held_out], diversity[held_out])
    rows = np.arange(len(adaptive_indices))
    changed_difference = bootstrap_differences(
        changed[held_out][rows, adaptive_indices], changed[held_out][rows, fixed_indices]
    )
    diversity_difference = bootstrap_differences(
        diversity[held_out][rows, adaptive_indices], diversity[held_out][rows, fixed_indices]
    )
    oracle_beats_fixed = (
        oracle["teacher_changed_fraction"] > fixed_metrics["teacher_changed_fraction"]
        and oracle["candidate_move_diversity_mean"] > fixed_metrics["candidate_move_diversity_mean"]
    )
    if not oracle_beats_fixed:
        classification = "rejected"
    elif changed_difference["bootstrap_95"][0] >= 0 and diversity_difference["bootstrap_95"][0] >= 0 and changed_difference["mean"] > 0 and diversity_difference["mean"] > 0:
        classification = "supported"
    elif changed_difference["bootstrap_95"][1] <= 0 or diversity_difference["bootstrap_95"][1] <= 0:
        classification = "rejected"
    else:
        classification = "measured"
    output = {
        "format": "adaptive-radius-analysis-v1",
        "scales": scales.tolist(),
        "selection_roots": args.selection_roots,
        "held_out_roots": len(adaptive_indices),
        "selected_depth": depth,
        "selection_cross_validation": cross_validation,
        "selected_fixed_radius": float(scales[fixed]),
        "tree": readable_tree(tree, scales),
        "held_out": {"fixed": fixed_metrics, "adaptive": adaptive, "oracle": oracle},
        "adaptive_minus_fixed": {
            "teacher_changed_fraction": changed_difference,
            "candidate_move_diversity": diversity_difference,
        },
        "classification": classification,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
