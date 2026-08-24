#!/usr/bin/env python3
"""Create a deterministic, reasonably balanced opening set from evaluation JSONL."""

import argparse
import json
import random
from pathlib import Path

import cshogi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-abs-eval", type=int, default=500)
    args = parser.parse_args()
    if args.count <= 0 or args.start < 0 or args.max_abs_eval < 0:
        parser.error("count must be positive; start/max-abs-eval must be non-negative")

    eligible = []
    seen = set()
    with args.input.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sfen = record["sfen"]
            if sfen in seen or abs(int(record["eval"])) > args.max_abs_eval:
                continue
            board = cshogi.Board(sfen)
            if board.is_game_over():
                continue
            seen.add(sfen)
            eligible.append(
                {
                    "sfen": sfen,
                    "source_index": record.get("source_index"),
                    "teacher_eval": int(record["eval"]),
                    "source_line": line_number,
                }
            )

    if len(eligible) < args.start + args.count:
        raise ValueError(
            f"only {len(eligible)} eligible positions for slice "
            f"[{args.start}:{args.start + args.count}]"
        )
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[args.start : args.start + args.count]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for record in selected:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "count": len(selected),
                "start": args.start,
                "seed": args.seed,
                "max_abs_eval": args.max_abs_eval,
                "eligible": len(eligible),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
