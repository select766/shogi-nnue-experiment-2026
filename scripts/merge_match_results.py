#!/usr/bin/env python3
"""Merge independent, disjoint-opening run_match result files."""

import argparse
import json
from pathlib import Path

from train_nnue.run_match import elo_diff, finite_or_none


def validate_runs(runs, *, expected_games=None, require_color_pairs=False):
    if not runs:
        raise ValueError("at least one match result is required")
    reference = runs[0]
    details = []
    for chunk_index, run in enumerate(runs):
        for key in ("engine1", "engine2", "search"):
            if run[key] != reference[key]:
                raise ValueError(f"incompatible {key} across match results")
        counted_games = run["wins"] + run["losses"] + run["draws"]
        if run["games"] != counted_games:
            raise ValueError(f"chunk {chunk_index}: games does not match W/L/D")
        if len(run["details"]) != run["games"]:
            raise ValueError(f"chunk {chunk_index}: details length does not match games")
        for detail in run["details"]:
            details.append({"chunk": chunk_index, **detail})

    if expected_games is not None and len(details) != expected_games:
        raise ValueError(
            f"expected {expected_games} games, found {len(details)}"
        )

    if require_color_pairs:
        colors_by_sfen = {}
        seen = set()
        for detail in details:
            key = (detail["sfen"], detail["engine1_color"])
            if key in seen:
                raise ValueError("duplicate SFEN/color game across match results")
            seen.add(key)
            colors_by_sfen.setdefault(detail["sfen"], set()).add(
                detail["engine1_color"]
            )
        expected_colors = {"black", "white"}
        if any(colors != expected_colors for colors in colors_by_sfen.values()):
            raise ValueError("every SFEN must have one black and one white game")

    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int)
    parser.add_argument("--require-color-pairs", action="store_true")
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.input]
    reference = runs[0]
    details = validate_runs(
        runs,
        expected_games=args.expected_games,
        require_color_pairs=args.require_color_pairs,
    )

    wins = sum(run["wins"] for run in runs)
    losses = sum(run["losses"] for run in runs)
    draws = sum(run["draws"] for run in runs)
    games = wins + losses + draws
    elo, elo_se = elo_diff(wins, losses, draws)
    radius = 1.959963984540054 * elo_se
    output = {
        "engine1": reference["engine1"],
        "engine2": reference["engine2"],
        "search": reference["search"],
        "source_results": [str(path) for path in args.input],
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_rate": (wins + 0.5 * draws) / games,
        "elo_difference": finite_or_none(elo),
        "elo_standard_error": elo_se,
        "elo_95": {
            "lower": finite_or_none(elo - radius),
            "upper": finite_or_none(elo + radius),
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: output[k] for k in (
        "games", "wins", "losses", "draws", "score_rate",
        "elo_difference", "elo_standard_error", "elo_95"
    )}, indent=2))


if __name__ == "__main__":
    main()
