import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from train_nnue.match_protocol_pilot import AuditedEngine
from train_nnue.match_protocol_qualification import FIXTURES, FIXTURE_OPTIONS, ForcedCycle, main, runtime_manifest
from train_nnue.run_match import play_game


class QualificationTests(unittest.TestCase):
    def test_cycles_and_boundaries(self):
        for name, sfen, cycle, result in FIXTURES:
            for history in (False, True):
                for limit in (8, 12):
                    engine = Mock()
                    engine.go.side_effect = lambda **kw: (kw["searchmoves"][0], None)
                    forced = ForcedCycle(engine, cycle)
                    trace = {}
                    actual = play_game(forced, forced, {"nodes": 128}, sfen,
                                       history=history, max_moves=limit, details=trace)
                    self.assertEqual(actual, (0, 8) if limit == 8 else (result, 12))
                    self.assertEqual(trace["termination"], "max_moves" if limit == 8 else name)
                    self.assertEqual(engine.go.call_count, limit)

    def test_forced_move_mismatch_fails(self):
        engine = Mock()
        engine.go.return_value = ("resign", None)
        with self.assertRaisesRegex(RuntimeError, "forced move mismatch"):
            ForcedCycle(engine, FIXTURES[0][2]).go(nodes=128, listener=lambda _: None)

    def test_real_command_retains_cache_validation(self):
        engine = AuditedEngine.__new__(AuditedEngine)
        engine.options = {"ClearTTOnDynamicWeights": "true"}
        engine.send = Mock()
        engine.read = Mock(side_effect=["info string blending_weight=[0.125]",
                           "info string dynamic_weight_cache clear=1 elapsed_ms=0", "bestmove 5i6i"])
        self.assertEqual(engine.go(nodes=128, listener=lambda _: None, searchmoves=["5i6i"])[0], "5i6i")
        engine.send.assert_called_once_with("go nodes 128 searchmoves 5i6i")

    def test_single_unpromoted_rook_searchmove(self):
        # Transport regression, not a substitute for the external real-engine run.
        engine = AuditedEngine.__new__(AuditedEngine)
        engine.options = {"ClearTTOnDynamicWeights": "true", **FIXTURE_OPTIONS}
        engine.send = Mock()
        engine.read = Mock(side_effect=["info string blending_weight=[0.125]",
                           "info string dynamic_weight_cache clear=1 elapsed_ms=0", "bestmove 5c4c"])
        forced = ForcedCycle(engine, ("5c4c",))
        forced.position(sfen="sfen " + FIXTURES[1][1], moves=["5a4a"])
        self.assertEqual(forced.go(nodes=128, listener=lambda _: None)[0], "5c4c")
        self.assertEqual(engine.send.call_args.args[0], "go nodes 128 searchmoves 5c4c")

    def test_all_legal_moves_option_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock()
            process.poll.return_value = 0
            with patch("train_nnue.match_protocol_pilot.subprocess.Popen", return_value=process), \
                 patch.object(AuditedEngine, "read", return_value="usiok"):
                with self.assertRaisesRegex(RuntimeError, "required USI option missing: GenerateAllLegalMoves"):
                    AuditedEngine("/unused", FIXTURE_OPTIONS, 0, Path(directory) / "usi.log")

    def test_run_records_required_options_and_failed_underpromotion(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "input-manifest.json").write_text("{}")
            engine = Mock()
            engine.process.pid = os.getpid()
            engine.advertised = ["option name GenerateAllLegalMoves type check default false"]

            def reply(**kw):
                move = kw["searchmoves"][0]
                actual = "5c7c+" if move == "5c4c" else move
                kw["listener"]("info string blending_weight=[0.125]")
                kw["listener"]("bestmove " + actual)
                return actual, None

            engine.go.side_effect = reply
            with patch("sys.argv", ["qualification", "--run-dir", directory, "--binary", "/unused"]), \
                 patch("train_nnue.match_protocol_qualification.runtime_manifest", return_value={}), \
                 patch("train_nnue.match_protocol_qualification.AuditedEngine", return_value=engine) as factory:
                with self.assertRaisesRegex(RuntimeError, "5c4c != 5c7c"):
                    main()
            self.assertEqual(factory.call_args.args[1]["GenerateAllLegalMoves"], "true")
            out = Path(json.loads((run / "artifacts/latest.json").read_text())["directory"])
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["protocol_version"], 2)
            self.assertEqual(manifest["fixture_options"], FIXTURE_OPTIONS)
            row = json.loads((out / "fixture-009.json").read_text())
            failure = row["failed_searches"][0]
            self.assertFalse(row["complete"])
            self.assertEqual(failure["expected"], "5c4c")
            self.assertEqual(failure["actual"], "5c7c+")
            self.assertTrue(failure["position"].startswith("position sfen "))
            self.assertEqual(failure["command"], "go nodes 128 searchmoves 5c4c")
            self.assertEqual(failure["info"], ["info string blending_weight=[0.125]"])
            self.assertIn("bestmove 5c7c+", failure["responses"])
            self.assertEqual(json.loads((out / "records.json").read_text())[-1], row)
            self.assertEqual(json.loads((out / "summary.json").read_text())["complete"], 8)

    def test_distribution_version_and_lock_hash(self):
        manifest = runtime_manifest(["uv.lock"])
        self.assertNotEqual(manifest["cshogi_version"], "unknown")
        self.assertEqual(len(manifest["files"][str(Path("uv.lock").resolve())]), 64)

    def test_failure_summary_without_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "input-manifest.json").write_text("{}")
            with patch("sys.argv", ["qualification", "--run-dir", directory, "--binary", "/unused"]), \
                 patch("train_nnue.match_protocol_qualification.runtime_manifest", return_value={}), \
                 patch("train_nnue.match_protocol_qualification.AuditedEngine", side_effect=RuntimeError("fixture setup fault")):
                with self.assertRaisesRegex(RuntimeError, "fixture setup fault"):
                    main()
            out = Path(json.loads((run / "artifacts/latest.json").read_text())["directory"])
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["complete"], 0)
            self.assertLess((run / "result-summary.md").stat().st_size, 65536)
