"""Independent integrity checks for the derived floodgate 2025 corpus (no engines)."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import cshogi

from train_nnue.prepare_floodgate import identity, load_exclusions, sha, split_for


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("partial build is not a complete corpus")
    for name, expected in manifest["files"].items():
        if sha(directory / name) != expected["sha256"]:
            raise ValueError(f"file changed: {name}")
    stats = manifest["statistics"]
    seen_games, seen_ids, splits = set(), set(), {}
    position_count = 0
    months = Counter()
    with gzip.open(directory / "games.jsonl.gz", "rt") as games, gzip.open(directory / "positions.jsonl.gz", "rt") as positions:
        for line in games:
            game = json.loads(line)
            assert len(game["ratings"]) == 2 and all(r >= 3500 for r in game["ratings"])
            assert game["started_at"].startswith("2025-")
            assert game["source_game_id"] not in seen_ids
            assert game["game_hash"] not in seen_games
            assert game["game_hash"] == hashlib.sha256((game["initial_sfen"] + "\n" + " ".join(game["moves"])).encode()).hexdigest()
            assert game["split"] == split_for(game["game_hash"], manifest["config"]["seed"])
            assert game["total_plies"] == len(game["moves"])
            seen_ids.add(game["source_game_id"])
            seen_games.add(game["game_hash"])
            splits[game["source_game_id"]] = game["split"]
            months[game["started_at"][:7]] += 1
            board = cshogi.Board(game["initial_sfen"])
            for i, usi in enumerate(game["moves"]):
                row = json.loads(next(positions))
                assert row["source_game_id"] == game["source_game_id"]
                assert row["game_hash"] == game["game_hash"]
                assert row["split"] == game["split"]
                assert row["sfen"] == board.sfen()
                assert row["played_plies"] == i and row["game_ply"] == i + 1
                assert row["turn"] == board.turn and row["bestmove"] == usi
                assert row["reported_score_black"] == game["reported_scores_black"][i]
                score = game["reported_scores_black"][i]
                assert row.get("eval") == (None if score is None else score * (1 if board.turn == 0 else -1))
                move = board.move_from_usi(usi)
                assert move and board.is_legal(move)
                board.push(move)
                position_count += 1
        assert next(positions, None) is None
    assert len(seen_games) == stats["accepted_games"]
    assert position_count == stats["positions"]
    assert dict(months) == manifest["games_by_month"]
    old, _ = load_exclusions(manifest["config"])
    sample_positions, sample_games = set(), set()
    sample_counts = {}
    for split in ("development", "validation", "test", "match"):
        count = 0
        with (directory / (split + ".jsonl")).open() as stream:
            for line in stream:
                row = json.loads(line)
                key = identity(row["sfen"])
                assert key not in old and key not in sample_positions
                assert row["source_game_id"] not in sample_games
                assert splits[row["source_game_id"]] == split == row["split"]
                assert min(row["ratings"]) >= 3500
                assert len(row["history_usi"]) == row["played_plies"] == row["game_ply"] - 1
                board = cshogi.Board(row["initial_sfen"])
                for move in row["history_usi"]:
                    parsed = board.move_from_usi(move)
                    assert board.is_legal(parsed)
                    board.push(parsed)
                assert board.sfen() == row["sfen"]
                assert board.is_legal(board.move_from_usi(row["bestmove"]))
                if split == "match":
                    assert abs(row["eval"]) <= 500
                sample_positions.add(key)
                sample_games.add(row["source_game_id"])
                count += 1
        assert count == stats.get("samples_" + split, 0)
        sample_counts[split] = count
    with gzip.open(directory / "rejections.jsonl.gz", "rt") as stream:
        rejected = sum(1 for _ in stream)
    assert rejected + len(seen_games) == manifest["archive_members"] == stats["archive_games_scanned"]
    return {"verified": True, "games": len(seen_games), "positions": position_count,
            "rejected": rejected, "samples": sample_counts, "months": dict(sorted(months.items()))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), ensure_ascii=False, indent=2))
