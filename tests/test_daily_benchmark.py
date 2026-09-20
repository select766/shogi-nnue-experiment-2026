from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cshogi

from train_nnue import daily_benchmark as daily
from train_nnue.research_loop import Store, Worker, choose, write_json


class DailyBenchmarkTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / ".research-loop")
        write_json(self.store.directory / "state.json", {
            "jobs": [], "paused": False, "interrupt": False, "active": None})
        for name in ("engine", "reference", "eval/nn.bin", "release/head.bin", "release/backbone.onnx",
                     "checkpoint", "evidence.md", "src/train_nnue/daily_benchmark.py",
                     "src/train_nnue/accuracy_statistics.py", "src/train_nnue/eval_accuracy.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        champion = {"id": "initial", "checkpoint": "checkpoint", "evidence": "evidence.md",
                    "selection_note": "independent evidence", "engine_path": "engine",
                    "engine_options": {"Threads": 1, "EvalDir": "eval", "ExpertBlendingDir": "release"},
                    "assets": ["engine"]}
        write_json(self.root / "champion.json", champion)
        board = cshogi.Board()
        rows = []
        for _ in range(6):
            move = next(iter(board.legal_moves))
            rows.append({"sfen": board.sfen(), "eval": 0, "bestmove": cshogi.move_to_usi(move)})
            board.push(move)
        for name, values in (("accuracy.jsonl", rows[:2]), ("source.jsonl", rows)):
            (self.root / name).write_text("".join(json.dumps(r) + "\n" for r in values))
        self.config = {"series": "series1", "champion": "champion.json", "accuracy_dataset": "accuracy.jsonl",
                       "accuracy_positions": 2, "accuracy_nodes": 3, "opening_source": "source.jsonl",
                       "opening_count": 2, "opening_seed": 42, "max_abs_eval": 500,
                       "match_nodes": 2, "max_moves": 6, "workers": 1,
                       "baseline": {"name": "fixed", "engine_path": "reference", "assets": ["reference"],
                                    "engine_options": {"Threads": 1, "EvalDir": "eval"}}}
        write_json(self.root / "config.json", self.config)
        self.settings = {"enabled": True, "config": "config.json", "interval_seconds": 86400, "timeout_seconds": 60}
        self.renderer = patch.object(daily, "render")
        self.renderer.start()
        self.addCleanup(self.renderer.stop)

    def initialize(self):
        return daily.initialize(self.root, self.store.directory, Path("config.json"))

    def test_fixed_data_and_opponent_are_pinned(self):
        directory = self.initialize()
        accuracy = {daily.identity(v["sfen"]) for v in daily.records(directory / "accuracy.jsonl")}
        openings = {daily.identity(v["sfen"]) for v in daily.records(directory / "openings.jsonl")}
        self.assertFalse(accuracy & openings)
        (self.root / "reference").write_text("changed")
        with self.assertRaisesRegex(ValueError, "pinned asset"):
            self.initialize()

    def test_cannot_mix_changed_protocol_or_model_under_old_id(self):
        self.initialize()
        self.config["match_nodes"] += 1
        write_json(self.root / "config.json", self.config)
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            self.initialize()
        self.config["match_nodes"] -= 1
        write_json(self.root / "config.json", self.config)
        (self.root / "release/head.bin").write_text("new weights")
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.initialize()

    def test_due_and_no_duplicate_jobs(self):
        job_id = daily.schedule(self.root, self.store, self.settings)
        state = self.store.snapshot()
        self.assertEqual(choose(state, {})["id"], job_id)
        self.assertIsNone(daily.schedule(self.root, self.store, self.settings))
        with self.store.edit() as state:
            state["jobs"][0]["status"] = "done"
        clock = datetime.fromisoformat(self.store.snapshot()["jobs"][0]["created_at"])
        self.assertFalse(daily.due(self.store.snapshot(), 86400, clock + timedelta(hours=23)))
        self.assertTrue(daily.due(self.store.snapshot(), 86400, clock + timedelta(hours=24)))

    def test_scheduled_champion_does_not_follow_later_pointer_changes(self):
        daily.schedule(self.root, self.store, self.settings)
        run_dir = Path(self.store.snapshot()["jobs"][0]["attempts"][0])
        champion = daily.read_json(self.root / "champion.json")
        champion["id"] = "next"
        write_json(self.root / "champion.json", champion)
        self.assertEqual(daily.read_json(run_dir / "benchmark.json")["champion"]["descriptor"]["id"], "initial")

    def test_metric_completeness_and_draw_accounting(self):
        protocol = {"config": self.config}
        accuracy = [{"index": 0, "match": True}, {"index": 1, "match": False}]
        matches = [{"opening": i, "sfen": f"position-{i} b - 1", "color": c, "outcome": outcome}
                   for i, outcomes in enumerate((("win", "draw"), ("loss", "win")))
                   for c, outcome in enumerate(outcomes)]
        result = daily.summarize(accuracy, matches, protocol)
        self.assertEqual(result["accuracy"]["accuracy"], .5)
        self.assertEqual(result["win_rate"]["value"], .5)
        self.assertEqual(result["score_rate"]["value"], .625)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            daily.summarize(accuracy, matches[:-1], protocol)
        matches[1]["color"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate"):
            daily.summarize(accuracy, matches, protocol)

    def test_game_sends_history_and_resets_each_side(self):
        class Engine:
            def __init__(self):
                self.positions = []
                self.resets = 0

            def isready(self):
                self.resets += 1

            def usinewgame(self):
                self.resets += 1

            def setoption(self, *args):
                pass

            def position(self, *, sfen):
                self.positions.append(sfen)

            def go(self, **kwargs):
                position = self.positions[-1].removeprefix("sfen ").split(" moves ")
                board = cshogi.Board(position[0])
                if len(position) > 1:
                    for move in position[1].split():
                        board.push_usi(move)
                return cshogi.move_to_usi(next(iter(board.legal_moves))), None

            def quit(self):
                pass

        black, white = Engine(), Engine()
        _, moves, _ = daily.game(black, white, cshogi.STARTING_SFEN, 2, 4)
        self.assertEqual(len(moves), 4)
        self.assertIn(" moves ", black.positions[-1])
        self.assertEqual(black.resets, 2)
        self.assertEqual(white.resets, 2)

    def test_failed_measurement_never_enters_history(self):
        daily.schedule(self.root, self.store, self.settings)
        run_dir = Path(self.store.snapshot()["jobs"][0]["attempts"][0])
        with patch.object(daily, "measure", side_effect=RuntimeError("engine failed")):
            with self.assertRaises(RuntimeError):
                daily.execute(self.root, run_dir)
        self.assertEqual(list((self.initialize() / "measurements").glob("*.json")), [])

    def test_completed_retry_does_not_compute_again(self):
        daily.schedule(self.root, self.store, self.settings)
        run_dir = Path(self.store.snapshot()["jobs"][0]["attempts"][0])
        accuracy = [{"index": 0, "match": True}, {"index": 1, "match": True}]
        matches = [{"opening": i, "sfen": f"p{i} b - 1", "color": c, "outcome": "draw"}
                   for i in range(2) for c in range(2)]
        with patch.object(daily, "measure", return_value=(accuracy, matches)) as compute:
            daily.execute(self.root, run_dir)
            daily.execute(self.root, run_dir)
            self.assertEqual(compute.call_count, 1)

    def test_builtin_daily_phase_never_launches_codex(self):
        daily.schedule(self.root, self.store, self.settings)
        job = self.store.snapshot()["jobs"][0]
        run_dir = Path(job["attempts"][0])
        write_json(run_dir / "benchmark-complete.json", {"dashboard": "growth.html"})
        worker = Worker(self.root, self.store, {})
        with patch.object(worker, "run_process", return_value={"exit_code": 0, "reason": "exited"}) as process:
            worker.phase(job, None)
        self.assertIn("train_nnue.daily_benchmark", process.call_args.args[0])
        self.assertNotIn("codex", process.call_args.args[0])
        self.assertEqual(self.store.snapshot()["jobs"][0]["status"], "done")


if __name__ == "__main__":
    unittest.main()
