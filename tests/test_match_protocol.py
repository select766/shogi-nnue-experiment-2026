import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import cshogi

from train_nnue.run_match import RootStatisticsEngine, play_game
from train_nnue.match_protocol_pilot import AuditedEngine, common_complete, generated_openings, main, summarize


class FakeEngine:
    def __init__(self, moves):
        self.moves = iter(moves)
        self.events = []

    def isready(self):
        self.events.append("isready")

    def usinewgame(self):
        self.events.append("usinewgame")

    def position(self, **kwargs):
        self.events.append(kwargs)

    def go(self, listener, **kwargs):
        listener("info depth 1 nodes 100000 score cp 0")
        return next(self.moves), None


class MatchProtocolTest(unittest.TestCase):
    def test_reset_full_history_and_legacy(self):
        for history in (True, False):
            a, b = FakeEngine(["7g7f", "resign"]), FakeEngine(["3c3d"])
            trace = {}
            self.assertEqual(play_game(a, b, {"nodes": 100000}, history=history, details=trace), (-1, 2))
            self.assertEqual(a.events[:2], ["isready", "usinewgame"])
            self.assertEqual(b.events[:2], ["isready", "usinewgame"])
            self.assertEqual(a.events[-1]["moves"], ["7g7f", "3c3d"] if history else [])
            self.assertEqual(trace["termination"], "resign")
            self.assertEqual(trace["searches"][0]["nodes"], 100000)
            self.assertGreater(trace["elapsed_seconds"], 0)

    def test_illegal_valid_syntax_rejected(self):
        for move in ("7g7e", "win", None):
            trace = {}
            with self.assertRaises(ValueError):
                play_game(FakeEngine([move]), FakeEngine([]), {"nodes": 1}, details=trace)
            self.assertIn(trace["termination"], ("illegal_move", "protocol_error"))
            self.assertEqual(trace["moves_usi"], [])

    def test_white_to_move_and_color_reversal(self):
        sfen = cshogi.STARTING_SFEN.replace(" b ", " w ")
        a, b = FakeEngine([]), FakeEngine(["resign"])
        self.assertEqual(play_game(a, b, {"nodes": 1}, sfen), (1, 0))
        self.assertEqual(len(a.events), 2)

    def test_repetition_draw_actual_board(self):
        trace = {}
        start = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        a = FakeEngine(["5i6i", "6i5i"] * 3)
        b = FakeEngine(["5a6a", "6a5a"] * 3)
        self.assertEqual(play_game(a, b, {"nodes": 1}, start, details=trace), (0, 12))
        self.assertEqual(trace["termination"], "repetition_draw")

    def test_three_occurrences_are_not_terminal(self):
        trace = {}
        a = FakeEngine(["5i6i", "6i5i"] * 2)
        b = FakeEngine(["5a6a", "6a5a"] * 2)
        play_game(a, b, {"nodes": 1}, "4k4/9/9/9/9/9/9/9/4K4 b - 1", max_moves=8, details=trace)
        self.assertEqual(trace["termination"], "max_moves")

    def test_checkmate_without_search(self):
        trace = {}
        result = play_game(FakeEngine([]), FakeEngine([]), {"nodes": 1},
                           "4k4/4G4/4R4/9/9/9/9/9/K8 w - 1", details=trace)
        self.assertEqual(result, (1, 0))
        self.assertEqual(trace["termination"], "no_legal_moves")

    def test_perpetual_check_actual_board(self):
        trace = {}
        start = "4k4/9/4R4/9/9/9/9/9/K8 w - 1"
        a = FakeEngine(["5c4c", "4c5c"] * 3)
        b = FakeEngine(["5a4a", "4a5a"] * 3)
        self.assertEqual(play_game(a, b, {"nodes": 1}, start, details=trace), (-1, 12))
        self.assertEqual(trace["termination"], "perpetual_check_black")

    def test_white_perpetual_check_and_fourfold_at_move_limit(self):
        trace = {}
        start = "k8/9/9/9/9/9/4r4/9/4K4 b - 1"
        a = FakeEngine(["5i4i", "4i5i"] * 3)
        b = FakeEngine(["5g4g", "4g5g"] * 3)
        self.assertEqual(play_game(a, b, {"nodes": 1}, start, max_moves=12, details=trace), (1, 12))
        self.assertEqual(trace["termination"], "perpetual_check_white")

    def test_limit_and_repeated_game_reset(self):
        a, b = FakeEngine(["7g7f", "7g7f"]), FakeEngine([])
        for _ in range(2):
            trace = {}
            self.assertEqual(play_game(a, b, {"nodes": 1}, max_moves=1, details=trace), (0, 1))
            self.assertEqual(trace["termination"], "max_moves")
        self.assertEqual(a.events.count("usinewgame"), 2)

    def test_root_statistics_history_and_current_board(self):
        candidate, shallow = Mock(), Mock()
        engine = RootStatisticsEngine(candidate, shallow)
        moves = ["7g7f", "3c3d"]
        engine.isready()
        engine.usinewgame()
        engine.position(sfen="sfen " + cshogi.STARTING_SFEN, moves=moves)
        board = cshogi.Board()
        for move in moves:
            board.push_usi(move)
        self.assertEqual(engine.sfen, board.sfen())
        candidate.position.assert_called_with(sfen="sfen " + cshogi.STARTING_SFEN, moves=moves)
        with patch("train_nnue.run_match.shallow_multipv_features", return_value=[0] * 6) as features:
            engine.go(nodes=100)
            self.assertEqual(features.call_args.args[1], cshogi.STARTING_SFEN + " moves 7g7f 3c3d")

    def test_synthetic_openings_reproducible_legal_unique(self):
        openings = generated_openings()
        self.assertEqual(openings, generated_openings())
        self.assertEqual(len({r["sfen"] for r in openings}), 16)
        for row in openings:
            board = cshogi.Board(row["source_start"])
            for token in row["source_moves"]:
                move = board.move_from_usi(token)
                self.assertTrue(board.is_legal(move))
                board.push(move)
            self.assertEqual(board.sfen(), row["sfen"])

    def test_common_complete_excludes_all_partial_cells(self):
        rows = [dict(stage="factorial", opening_index=i, history=h, clear=c, candidate_color=k,
                     complete=True) for i in (1, 2) for h in (False, True)
                for c in (False, True) for k in (0, 1)]
        rows[-1]["complete"] = False
        self.assertEqual(common_complete(rows), [1])
        self.assertEqual(summarize([])["common_complete_openings"], [])

    def test_cache_telemetry_must_match_requested_policy(self):
        for clear in (False, True):
            for actual in (False, True, None):
                engine = AuditedEngine.__new__(AuditedEngine)
                engine.options = {"ClearTTOnDynamicWeights": str(clear).lower()}
                engine.send = Mock()
                lines = ["info string blending_weight=[0.125]"]
                if actual is not None:
                    lines.append(f"info string dynamic_weight_cache clear={int(actual)} elapsed_ms=0")
                lines += ["info depth 1 nodes 100000", "bestmove 7g7f"]
                engine.read = Mock(side_effect=lines)
                if actual == clear:
                    self.assertEqual(engine.go(nodes=100000, listener=lambda _: None), ("7g7f", None))
                else:
                    with self.assertRaises(RuntimeError):
                        engine.go(nodes=100000, listener=lambda _: None)

    def test_driver_failure_and_deadline_leave_reviewable_summary(self):
        # No real USI process: exercise the persistence path after child setup.
        for budget, expected in (("-1", "budget_stop"), ("30", "failed")):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                fake = Mock()
                fake.advertised = ["option name Threads type spin default 1"]
                fake.go.side_effect = RuntimeError("simulated protocol fault")
                with patch("sys.argv", ["pilot", "--run-dir", directory, "--binary", "/unused",
                                        "--budget-seconds", budget]), \
                     patch("train_nnue.match_protocol_pilot.AuditedEngine", return_value=fake), \
                     patch("train_nnue.match_protocol_pilot.sha256", return_value="test-hash"):
                    if expected == "failed":
                        with self.assertRaisesRegex(RuntimeError, "simulated"):
                            main()
                    else:
                        main()
                run = Path(directory)
                latest = json.loads((run / "artifacts/latest.json").read_text())
                out = Path(latest["directory"])
                summary = json.loads((out / "summary.json").read_text())
                self.assertEqual(summary["status"], expected)
                self.assertEqual(summary["protocol_gate"], "hold")
                self.assertLess((run / "result-summary.md").stat().st_size, 65536)
                if expected == "failed":
                    row = json.loads((out / "game-001.json").read_text())
                    self.assertFalse(row["complete"])
                fake.quit.assert_called()

    def test_driver_reuses_timing_pair_within_128_game_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Mock()
            fake.advertised = []
            fake.rss_kib.return_value = []

            def resign(**kwargs):
                kwargs["listener"]("info nodes 100000")
                return "resign", None

            fake.go.side_effect = resign
            with patch("sys.argv", ["pilot", "--run-dir", directory, "--binary", "/unused",
                                    "--budget-seconds", "300"]), \
                 patch("train_nnue.match_protocol_pilot.AuditedEngine", return_value=fake), \
                 patch("train_nnue.match_protocol_pilot.sha256", return_value="test-hash"):
                main()
            latest = json.loads((Path(directory) / "artifacts/latest.json").read_text())
            out = Path(latest["directory"])
            rows = json.loads((out / "records.json").read_text())
            self.assertEqual(len(rows), 128)
            self.assertEqual(sum(bool(row.get("used_for_timing")) for row in rows), 2)
            self.assertEqual(len(common_complete(rows)), 16)
            self.assertEqual(json.loads((out / "summary.json").read_text())["protocol_gate"], "pass")


if __name__ == "__main__":
    unittest.main()
