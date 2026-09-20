import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from train_nnue.match_protocol_pilot import AuditedEngine
from train_nnue.match_protocol_qualification import FIXTURES, ForcedCycle, main, runtime_manifest
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
