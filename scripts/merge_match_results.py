#!/usr/bin/env python3
"""Merge independent, disjoint-opening run_match result files."""

import argparse
import json
from pathlib import Path

from train_nnue.run_match import elo_diff, finite_or_none


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.input]
    reference = runs[0]
    for run in runs[1:]:
        for key in ("engine1", "engine2", "search"):
            if run[key] != reference[key]:
                raise ValueError(f"incompatible {key} across match results")

    wins = sum(run["wins"] for run in runs)
    losses = sum(run["losses"] for run in runs)
    draws = sum(run["draws"] for run in runs)
    games = wins + losses + draws
    elo, elo_se = elo_diff(wins, losses, draws)
    radius = 1.959963984540054 * elo_se
    details = []
    for chunk_index, run in enumerate(runs):
        for detail in run["details"]:
            details.append({"chunk": chunk_index, **detail})
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
