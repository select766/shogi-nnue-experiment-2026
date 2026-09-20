"""Offline verification of a completed pilot; never starts engines."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import cshogi

from train_nnue.match_protocol_pilot import generated_openings, sha256, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("execution", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.execution
    read = lambda name: json.loads((root / name).read_text())
    rows, openings, summary = read("records.json"), read("openings.json"), read("summary.json")
    planned = read("allocation.json")["planned_openings"]
    assert planned > 0 and len(rows) == planned * 8
    assert summary["status"] == "complete" and summary["failure"] is None
    assert summary["protocol_gate"] == "pass"
    assert openings == generated_openings()
    cells = Counter((r["opening_index"], r["history"], r["clear"], r["candidate_color"]) for r in rows)
    assert cells == Counter({(i, h, c, k): 1 for i in range(planned)
                             for h in (False, True) for c in (False, True) for k in (0, 1)})
    assert sum(bool(r.get("used_for_timing")) for r in rows) == 2
    nodes, clear_ms, terminations = [], [], Counter()
    for number, row in enumerate(rows, 1):
        assert row == read(f"game-{number:03d}.json")
        assert row["complete"] and row["stage"] == "factorial"
        trace = row["trace"]
        assert trace["start_sfen"] == openings[row["opening_index"]]["sfen"]
        assert trace["history"] == row["history"]
        board = cshogi.Board(trace["start_sfen"])
        assert len(trace["searches"]) == len(trace["moves_usi"])
        for index, search in enumerate(trace["searches"]):
            position = "sfen " + (trace["start_sfen"] if row["history"] else board.sfen())
            if row["history"] and index:
                position += " moves " + " ".join(trace["moves_usi"][:index])
            assert search["position"] == position and search["turn"] == board.turn
            assert search["elapsed_seconds"] > 0 and search["nodes"] > 0
            nodes.append(search["nodes"])
            cache = [s for s in search["info"] if s.startswith("info string dynamic_weight_cache ")]
            assert len(cache) == 1 and f"clear={int(row['clear'])} " in cache[0]
            clear_ms.append(int(cache[0].split("elapsed_ms=")[1]))
            assert sum(s.startswith("info string blending_weight=") for s in search["info"]) == 1
            token = trace["moves_usi"][index]
            assert search["bestmove"] == token
            move = board.move_from_usi(token)
            assert move and board.is_legal(move)
            board.push(move)
        assert board.sfen() == trace["final_sfen"] and board.is_game_over()
        assert trace["termination"] == "no_legal_moves"
        assert trace["result"] == (1 if board.turn == 1 else -1)
        assert row["candidate_score"] == (1 + trace["result"] * (1 if row["candidate_color"] == 0 else -1)) / 2
        terminations[trace["termination"]] += 1
    recomputed = summarize(rows)
    assert all(summary[key] == value for key, value in recomputed.items())
    resets = []
    for i in (0, 1):
        log = Path(read(f"engine{i}.json")["log"]).read_text().splitlines()
        commands = [line for line in log if line.startswith("> ")]
        games = [j for j, line in enumerate(commands) if line == "> usinewgame"]
        assert len(games) == len(rows)
        assert all(commands[j-1] == "> isready" for j in games)
        expected_positions = ["> position " + s["position"] for r in rows
                              for s in r["trace"]["searches"]
                              if (s["turn"] == r["candidate_color"]) == (i == 0)]
        assert [s for s in commands if s.startswith("> position ")] == expected_positions
        assert [s for s in commands if s.startswith("> go ")] == ["> go nodes 100000"] * len(expected_positions)
        resets.append(len(games))
    # Preserve input hashes and report any source changes made after execution.
    manifest = read("manifest.json")
    mismatches = [name for name, digest in manifest["files"].items() if sha256(name) != digest]
    assert not mismatches, mismatches
    result = dict(verified=True, games=len(rows), openings=planned, searches=len(nodes),
                  nodes_min=min(nodes), nodes_max=max(nodes), cache_events=len(clear_ms),
                  cache_clear_ms=sum(clear_ms), terminations=dict(terminations),
                  per_engine_game_resets=resets, input_hashes_verified=len(manifest["files"]),
                  review_environment_cshogi_version=version("cshogi"),
                  runtime_cshogi_version=manifest["cshogi_version"],
                  below_budget_searches=sum(n < 100000 for n in nodes),
                  below_budget_with_mate=sum(s["nodes"] < 100000 and any(" score mate " in line for line in s["info"])
                                            for r in rows for s in r["trace"]["searches"]),
                  evidence_sha256={name: sha256(root / name) for name in
                                   ("records.json", "openings.json", "summary.json", "manifest.json")})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
