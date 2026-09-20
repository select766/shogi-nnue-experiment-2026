import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cshogi

from tests.test_match_protocol import FakeEngine
from train_nnue.run_match import play_game
from train_nnue.match_protocol_qualification import FIXTURES
from train_nnue.decision_replication_cost import cost_summary, execute


class SourceHistoryTest(unittest.TestCase):
    def test_prefix_counts_for_all_repetition_adjudications(self):
        for reason, origin, cycle, expected in FIXTURES:
            with self.subTest(reason=reason):
                board = cshogi.Board(origin)
                prefix = list(cycle * 2)
                for move in prefix:
                    board.push_usi(move)
                first = FakeEngine(cycle[::2])
                second = FakeEngine(cycle[1::2])
                black, white = (first, second) if board.turn == 0 else (second, first)
                trace = {}
                self.assertEqual(play_game(black, white, {"nodes": 1}, board.sfen(),
                    max_moves=4, initial_sfen=origin, history_usi=prefix, details=trace), (expected, 4))
                self.assertEqual(trace["termination"], reason)
                self.assertEqual(first.events[2], dict(sfen="sfen " + origin, moves=prefix))
                self.assertEqual(trace["moves_usi"], list(cycle))

    def test_bad_prefix_fails_before_engine_reset(self):
        for prefix in (["7g7e"], ["7g7f"]):
            a, b = FakeEngine([]), FakeEngine([])
            with self.assertRaises(ValueError):
                play_game(a, b, {"nodes": 1}, initial_sfen=cshogi.STARTING_SFEN, history_usi=prefix)
            self.assertEqual(a.events, [])

    def test_cost_planning_ignores_scores_and_requires_complete_sample(self):
        rows = [dict(complete=True, opening_index=i, wall_seconds=10, candidate_score=1)
                for i in range(8) for _ in range(2)]
        summary = cost_summary(rows)
        self.assertEqual(summary["chunk_pairs"], 525)
        for row in rows:
            row["candidate_score"] = 0
        self.assertEqual(cost_summary(rows), summary)
        rows[-1]["complete"] = False
        self.assertIsNone(cost_summary(rows)["chunk_pairs"])

    def test_input_failure_persists_summary_without_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch("train_nnue.decision_replication_cost.verify_inputs", side_effect=ValueError("changed hash")), \
                 patch("train_nnue.decision_replication_cost.AuditedEngine") as engine:
                with self.assertRaisesRegex(ValueError, "changed hash"):
                    execute(run, 5)
                engine.assert_not_called()
            out = Path(json.loads((run / "artifacts/latest.json").read_text())["directory"])
            self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "failed")
            self.assertIn("changed hash", (run / "result-summary.md").read_text())

    def test_complete_driver_runs_exactly_fixed_color_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "input-manifest.json").write_text("{}")
            (run / "protocol.json").write_text(json.dumps(dict(binary="unused", nodes=100000,
                max_moves=512, options=dict(candidate={}, control={}))))
            rows = [dict(game_hash=str(i), sfen=cshogi.STARTING_SFEN,
                    initial_sfen=cshogi.STARTING_SFEN, history_usi=[]) for i in range(8)]
            engine = Mock()
            engine.process.pid = __import__("os").getpid()
            engine.options = {}
            engine.advertised = []
            engine.rss_kib.return_value = []
            def game(*args, **kwargs):
                self.assertEqual(kwargs["initial_sfen"], cshogi.STARTING_SFEN)
                self.assertEqual(kwargs["history_usi"], [])
                kwargs["details"].update(searches=[dict(nodes=100000)], termination="resign")
                return -1, 0
            with patch("train_nnue.decision_replication_cost.verify_inputs", return_value=rows), \
                 patch("train_nnue.decision_replication_cost.runtime_manifest", return_value={}), \
                 patch("train_nnue.decision_replication_cost.sha256", return_value="fake"), \
                 patch("train_nnue.decision_replication_cost.AuditedEngine", return_value=engine), \
                 patch("train_nnue.decision_replication_cost.play_game", side_effect=game) as play:
                execute(run, 30)
            self.assertEqual(play.call_count, 16)
            out = Path(json.loads((run / "artifacts/latest.json").read_text())["directory"])
            records = json.loads((out / "records.json").read_text())
            self.assertEqual([(r["opening_index"], r["candidate_color"]) for r in records],
                             [(i, c) for i in range(8) for c in (0, 1)])
            self.assertEqual([r["candidate_score"] for r in records], [0, 1]*8)
            self.assertLess((run / "result-summary.md").stat().st_size, 65536)
