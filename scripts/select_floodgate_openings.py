"""Freeze game-disjoint cost and match subsets before observing match outcomes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from train_nnue.prepare_floodgate import dump, hashed, identity, save, sha


def select(source, output, count=10000, cost=8):
    if count <= 0 or cost <= 0:
        raise ValueError("positive count and cost required")
    manifest = json.loads((source / "manifest.json").read_text())
    pool = source / "match.jsonl"
    if not manifest["complete"] or sha(pool) != manifest["files"]["match.jsonl"]["sha256"]:
        raise ValueError("incomplete or modified source")
    rows = [json.loads(line) for line in pool.read_text().splitlines()]
    if len({r["game_hash"] for r in rows}) != len(rows) or len({identity(r["sfen"]) for r in rows}) != len(rows):
        raise ValueError("nonunique game/position pool")
    if len(rows) < count + cost:
        raise ValueError(f"insufficient pool: {len(rows)} for {count + cost}")
    if any(min(r["ratings"]) < 3500 or abs(r["eval"]) > 500 or r["game_ply"] < 24 for r in rows):
        raise ValueError("pool violates frozen eligibility conditions")
    seed = "floodgate2025-match-plan-v1"
    rows.sort(key=lambda r: hashed(seed + "|" + r["game_hash"]))
    output.mkdir(parents=True, exist_ok=False)
    parts = {"cost8.jsonl": rows[:cost], "openings10000.jsonl": rows[cost:cost+count],
             "reserve.jsonl": rows[cost+count:]}
    for name, part in parts.items():
        with (output / name).open("w") as stream:
            for row in part:
                dump(stream, row)
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
              "source_manifest": str(source / "manifest.json"), "source_manifest_sha256": sha(source / "manifest.json"),
              "source_pool_sha256": sha(pool), "code_sha256": sha(Path(__file__)),
              "files": {name: {"count": len(part), "sha256": sha(output / name)} for name, part in parts.items()},
              "purpose": "Fixed input pools only; no matches or model-strength claims",
              "history": "history_usi reconstructs original game; decide and preregister whether match runner uses it",
              "independence": "one position per original game; excludes explicitly inventoried old positions, not all pretraining ancestors",
              "reserve_policy": "Do not replace bad match outcomes with reserve entries"}
    save(output / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset/floodgate2025/curated-v1"))
    parser.add_argument("--output", type=Path, default=Path("dataset/floodgate2025/match-plan-v1"))
    args = parser.parse_args()
    print(json.dumps(select(args.source, args.output), ensure_ascii=False, indent=2))
