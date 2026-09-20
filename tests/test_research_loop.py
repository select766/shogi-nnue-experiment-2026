import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from train_nnue.research_loop import Store, Worker, add_jobs, choose, finish_review, main, registry, write_json

FIXTURE = Path(__file__).parent / "fixtures/research_fake_codex.py"


class ResearchLoopTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "docs/research").mkdir(parents=True)
        (self.root / "docs/research/hypotheses.md").write_text(
            "<!-- hypothesis id=H-ONE status=unverified priority=2 -->\n"
            "<!-- hypothesis id=H-TWO status=unverified priority=1 -->\n")
        (self.root / ".gitignore").write_text(".research-loop/\n")
        (self.root / "scripts").mkdir()
        wrapper = self.root / "scripts/project_python.sh"
        wrapper.write_text("#!/bin/sh\nexit 0\n")
        wrapper.chmod(0o755)
        for args in (["init", "-q"], ["config", "user.name", "Test"],
                     ["config", "user.email", "test@example.invalid"],
                     ["add", "."], ["commit", "-qm", "initial"]):
            subprocess.run(["git", *args], cwd=self.root, check=True)
        self.store = Store(self.root / ".research-loop")
        write_json(self.store.directory / "state.json", {
            "version": 1, "paused": False, "interrupt": False, "active": None, "jobs": []})
        self.mode = self.store.directory / "mode"
        self.mode.write_text("success")
        self.config = {"codex_command": [sys.executable, str(FIXTURE)],
                       "session_timeout_seconds": 10, "max_compute_seconds": 20,
                       "poll_seconds": 0.01, "terminate_grace_seconds": 0.1}

    def enqueue(self, key="first", hypothesis="H-ONE", dependencies=None):
        with self.store.edit() as state:
            add_jobs(state, [{"id": key, "hypothesis": hypothesis, "task": "Offline test",
                              "depends_on": dependencies or []}], registry(self.root))

    def run_worker(self, max_jobs=1):
        return Worker(self.root, self.store, self.config).run(max_jobs)

    def events(self):
        path = self.store.directory / "invocations.jsonl"
        return [json.loads(v) for v in path.read_text().splitlines()] if path.exists() else []

    def test_full_cycle_fresh_sessions_and_no_codex_during_compute(self):
        self.enqueue()
        self.assertEqual(self.run_worker(), 0)
        state = self.store.snapshot()
        self.assertEqual(state["jobs"][0]["status"], "done")
        events = self.events()
        self.assertEqual([v["phase"] for v in events], ["prepare", "review"])
        self.assertNotEqual(events[0]["pid"], events[1]["pid"])
        self.assertNotIn("resume", events[1]["args"])
        directory = Path(state["jobs"][0]["attempts"][0])
        # Fixture run.sh exits 17 if the preparer process is still alive.
        self.assertEqual(json.loads((directory / "execution.json").read_text())["exit_code"], 0)

    def test_unlimited_idle_waits_without_daily_and_accepts_new_job(self):
        sleeps = []

        def idle(_):
            sleeps.append(1)
            if len(sleeps) == 1:
                self.enqueue()
            else:
                with self.store.edit() as state:
                    state["paused"] = True

        def complete(worker, job, lock):
            with self.store.edit() as state:
                state["jobs"][0]["status"] = "done"

        with patch("train_nnue.research_loop.time.sleep", side_effect=idle), patch.object(Worker, "phase", complete):
            self.assertEqual(self.run_worker(max_jobs=0), 0)
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(self.events(), [])
        self.assertEqual(self.store.snapshot()["jobs"][0]["status"], "done")

    def test_bounded_empty_queue_exits(self):
        self.assertEqual(self.run_worker(max_jobs=1), 0)

    def test_deferred_prepare_is_reviewed_without_compute_then_independent_job_runs(self):
        self.mode.write_text("deferred")
        self.enqueue("missing", "H-TWO")
        self.enqueue("dependent", "H-ONE", ["missing"])
        self.enqueue("independent", "H-ONE")
        self.assertEqual(self.run_worker(max_jobs=2), 0)
        state = self.store.snapshot()
        self.assertEqual([j["status"] for j in state["jobs"]], ["abandoned", "cancelled", "abandoned"])
        self.assertFalse(state["paused"])
        self.assertEqual([v["phase"] for v in self.events()], ["prepare", "review", "prepare", "review"])
        directory = Path(state["jobs"][0]["attempts"][0])
        execution = json.loads((directory / "execution.json").read_text())
        self.assertEqual(execution["reason"], "preparation_deferred")
        self.assertIsNone(execution["exit_code"])
        self.assertFalse((directory / "result-summary.md").exists())
        self.assertIsNone(choose(state, registry(self.root)))

    def test_legacy_blocked_preparation_can_be_explicitly_deferred(self):
        self.mode.write_text("blocked")
        self.enqueue()
        self.assertEqual(self.run_worker(), 2)
        with patch("train_nnue.research_loop.ROOT", self.root):
            self.assertEqual(main(["defer", "first"]), 0)
            self.assertTrue(self.store.snapshot()["paused"])
            self.assertEqual(main(["resume"]), 0)
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual(self.store.snapshot()["jobs"][0]["status"], "abandoned")
        self.assertEqual([v["phase"] for v in self.events()], ["prepare", "review"])

    def test_failed_session_cannot_be_deferred(self):
        self.mode.write_text("prepare_failure")
        self.enqueue()
        self.assertEqual(self.run_worker(), 2)
        with patch("train_nnue.research_loop.ROOT", self.root):
            self.assertEqual(main(["defer", "first"]), 2)
        self.assertEqual(self.store.snapshot()["jobs"][0]["status"], "blocked")

    def test_replans_are_bounded_persisted_and_no_compute_runs(self):
        self.mode.write_text("replan")
        self.enqueue()
        self.assertEqual(self.run_worker(), 0)
        state = self.store.snapshot()
        job = state["jobs"][0]
        self.assertEqual(job["status"], "abandoned")
        self.assertFalse(state["paused"])
        self.assertEqual(len(job["attempts"]), 3)
        self.assertEqual(len(job["replan_history"]), 2)
        last = Path(job["attempts"][-1])
        context = json.loads((last / "job.json").read_text())
        self.assertEqual(context["replans_remaining"], 0)
        self.assertEqual(len(context["replan_history"]), 2)
        self.assertEqual(context["task"], "Use a recorded alternative")
        self.assertEqual([e["phase"] for e in self.events()], ["prepare", "review"] * 3)
        self.assertFalse((last / "result-summary.md").exists())

    def test_replan_budget_is_shared_across_job_ids(self):
        self.mode.write_text("replan")
        self.enqueue()
        self.assertEqual(self.run_worker(), 0)
        self.enqueue("new-id")
        self.assertEqual(self.run_worker(), 0)
        job = self.store.snapshot()["jobs"][1]
        self.assertEqual(job["status"], "abandoned")
        self.assertEqual(len(job["attempts"]), 1)

    def test_human_wait_does_not_stop_independent_work_and_answer_is_carried(self):
        self.mode.write_text("needs_human")
        self.enqueue("question", "H-TWO")
        self.enqueue("dependent", "H-ONE", ["question"])
        self.enqueue("independent", "H-ONE")
        self.assertEqual(self.run_worker(max_jobs=2), 0)
        state = self.store.snapshot()
        self.assertEqual([j["status"] for j in state["jobs"]], ["needs_human", "queued", "needs_human"])
        self.assertFalse(state["paused"])
        answer = self.store.directory / "answer.txt"
        answer.write_text("Do not use the unknown file; use the catalog corpus.")
        with patch("train_nnue.research_loop.ROOT", self.root):
            self.assertEqual(main(["answer", "question", str(answer)]), 0)
        self.mode.write_text("success")
        self.assertEqual(self.run_worker(), 0)
        job = self.store.snapshot()["jobs"][0]
        context = json.loads((Path(job["attempts"][-1]) / "job.json").read_text())
        self.assertEqual(context["human_answers"][0]["answer"], answer.read_text())
        self.assertEqual(job["status"], "done")
        self.assertEqual(self.run_worker(), 0)  # dependent job sees the recorded answer too
        dependent = self.store.snapshot()["jobs"][1]
        inherited = json.loads((Path(dependent["attempts"][-1]) / "job.json").read_text())
        self.assertEqual(inherited["human_decisions"][0]["job_id"], "question")
        self.assertEqual(inherited["human_decisions"][0]["answer"], answer.read_text())

    def test_normal_shortage_cannot_be_classified_as_human_wait(self):
        self.mode.write_text("invalid_human")
        self.enqueue()
        self.assertEqual(self.run_worker(), 2)
        self.assertIn("allowed category", self.store.snapshot()["jobs"][0]["message"])

    def test_abandoned_cancels_transitive_dependencies(self):
        self.mode.write_text("deferred")
        self.enqueue()
        self.enqueue("child", dependencies=["first"])
        self.enqueue("grandchild", dependencies=["child"])
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual([j["status"] for j in self.store.snapshot()["jobs"]],
                         ["abandoned", "cancelled", "cancelled"])

    def test_unresolved_review_cannot_bypass_budget_with_followup(self):
        self.enqueue()
        state = self.store.snapshot()
        result = {"status": "abandoned", "summary": "closed", "commit": "test", "report": "test",
                  "next_jobs": [{"id": "bypass", "hypothesis": "H-ONE", "task": "same", "depends_on": []}]}
        with self.assertRaisesRegex(ValueError, "same-hypothesis"):
            finish_review(state, "first", result, self.store.directory, registry(self.root))
        self.assertEqual(len(state["jobs"]), 1)

    def test_deferred_review_failure_stops_before_independent_work(self):
        self.mode.write_text("blocked")
        self.enqueue("missing", "H-TWO")
        self.enqueue("independent")
        self.assertEqual(self.run_worker(), 2)
        with patch("train_nnue.research_loop.ROOT", self.root):
            self.assertEqual(main(["defer", "missing"]), 0)
            self.assertEqual(main(["resume"]), 0)
        self.mode.write_text("review_failure")
        self.assertEqual(self.run_worker(), 2)
        state = self.store.snapshot()
        self.assertTrue(state["paused"])
        self.assertEqual([j["status"] for j in state["jobs"]], ["blocked", "queued"])
        self.assertEqual([v["phase"] for v in self.events()], ["prepare", "review"])

    def test_failed_compute_is_reviewed(self):
        self.mode.write_text("compute_failure")
        self.enqueue()
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual([v["phase"] for v in self.events()], ["prepare", "review"])
        directory = Path(self.store.snapshot()["jobs"][0]["attempts"][0])
        self.assertEqual(json.loads((directory / "execution.json").read_text())["exit_code"], 19)

    def test_bad_or_failed_preparation_never_executes(self):
        for mode in ("prepare_failure", "malformed"):
            with self.subTest(mode=mode):
                self.mode.write_text(mode)
                self.enqueue(key=mode)
                with self.store.edit() as state:
                    for job in state["jobs"][:-1]:
                        job["status"] = "cancelled"
                    state["paused"] = False
                self.assertEqual(self.run_worker(), 2)
                job = self.store.snapshot()["jobs"][-1]
                self.assertEqual(job["status"], "blocked")
                self.assertFalse((Path(job["attempts"][0]) / "execution.json").exists())

    def test_queue_update_is_atomic_on_invalid_proposal(self):
        self.mode.write_text("invalid_review")
        self.enqueue()
        self.assertEqual(self.run_worker(), 2)
        state = self.store.snapshot()
        self.assertEqual(len(state["jobs"]), 1)
        self.assertEqual(state["jobs"][0]["status"], "blocked")

    def test_followup_is_enqueued_without_running_when_limit_reached(self):
        self.mode.write_text("next_job")
        self.enqueue()
        self.assertEqual(self.run_worker(max_jobs=1), 0)
        state = self.store.snapshot()
        self.assertEqual([j["status"] for j in state["jobs"]], ["done", "queued"])
        self.assertEqual(choose(state, registry(self.root))["id"], "followup")

    def test_priority_dependencies_and_cycles(self):
        self.enqueue()
        self.enqueue("dependent", "H-TWO", ["first"])
        self.enqueue("independent", "H-TWO")
        self.assertEqual(choose(self.store.snapshot(), registry(self.root))["id"], "independent")
        with self.assertRaisesRegex(ValueError, "cycle"):
            with self.store.edit() as state:
                add_jobs(state, [
                    {"id": "a", "hypothesis": "H-ONE", "task": "a", "depends_on": ["b"]},
                    {"id": "b", "hypothesis": "H-ONE", "task": "b", "depends_on": ["a"]},
                ], registry(self.root))
        self.assertEqual(len(self.store.snapshot()["jobs"]), 3)

    def test_pause_before_start_launches_nothing(self):
        self.enqueue()
        with self.store.edit() as state:
            state["paused"] = True
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual(self.events(), [])

    def watch_and_pause(self, phase, immediate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self.store.edit() as state:
                active = state["active"]
                if active and active["phase"] == phase and active.get("pid"):
                    state.update(paused=True, interrupt=immediate)
                    return
            time.sleep(0.005)
        raise AssertionError("phase was not observed")

    def test_boundary_pause_preserves_next_phase_then_resumes(self):
        self.enqueue()
        watcher = threading.Thread(target=self.watch_and_pause, args=("prepare", False))
        watcher.start()
        self.assertEqual(self.run_worker(), 0)
        watcher.join(timeout=6)
        state = self.store.snapshot()
        self.assertEqual(state["jobs"][0]["phase"], "execute")
        self.assertEqual(len(self.events()), 1)
        with self.store.edit() as state:
            state["paused"] = False
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual(len(self.events()), 2)

    def test_immediate_pause_terminates_compute_and_skips_review(self):
        self.mode.write_text("long")
        self.enqueue()
        watcher = threading.Thread(target=self.watch_and_pause, args=("execute", True))
        watcher.start()
        self.assertEqual(self.run_worker(), 2)
        watcher.join(timeout=6)
        job = self.store.snapshot()["jobs"][0]
        self.assertEqual(job["status"], "blocked")
        self.assertEqual(len(self.events()), 1)
        directory = Path(job["attempts"][0])
        self.assertEqual(json.loads((directory / "execution.json").read_text())["reason"], "interrupted")

    def test_timeout_goes_to_review(self):
        self.mode.write_text("long")
        self.enqueue()
        self.assertEqual(self.run_worker(), 0)
        directory = Path(self.store.snapshot()["jobs"][0]["attempts"][0])
        self.assertEqual(json.loads((directory / "execution.json").read_text())["reason"], "timeout")
        pid = int((directory / "child.pid").read_text())
        stat = Path(f"/proc/{pid}/stat")
        self.assertTrue(not stat.exists() or stat.read_text().split()[2] == "Z")

    def test_stale_active_requires_recovery(self):
        self.enqueue()
        with self.store.edit() as state:
            state["active"] = {"job": "first", "pid": None}
        with self.assertRaisesRegex(ValueError, "recover"):
            self.run_worker()

    def test_dirty_workspace_does_not_start_model(self):
        self.enqueue()
        (self.root / "user-edit.txt").write_text("preserve me")
        self.assertEqual(self.run_worker(), 2)
        self.assertEqual(self.events(), [])
        self.assertEqual((self.root / "user-edit.txt").read_text(), "preserve me")

    def test_duplicate_worker_lock(self):
        with self.store.worker_lock():
            self.assertTrue(self.store.worker_present())
            with self.assertRaisesRegex(ValueError, "another worker"):
                self.run_worker()

    def test_retry_review_uses_saved_compute_and_new_session(self):
        self.mode.write_text("review_failure")
        self.enqueue()
        self.assertEqual(self.run_worker(), 2)
        job = self.store.snapshot()["jobs"][0]
        directory = Path(job["attempts"][0])
        execution = (directory / "execution.json").read_bytes()
        self.mode.write_text("success")
        with patch("train_nnue.research_loop.ROOT", self.root):
            self.assertEqual(main(["retry", "first", "--phase", "review"]), 0)
            self.assertEqual(main(["resume"]), 0)
        self.assertEqual(self.run_worker(), 0)
        self.assertEqual((directory / "execution.json").read_bytes(), execution)
        self.assertEqual([v["phase"] for v in self.events()], ["prepare", "review", "review"])

    def test_prepared_script_tamper_blocks_execution(self):
        self.enqueue()
        watcher = threading.Thread(target=self.watch_and_pause, args=("prepare", False))
        watcher.start()
        self.assertEqual(self.run_worker(), 0)
        watcher.join(timeout=6)
        directory = Path(self.store.snapshot()["jobs"][0]["attempts"][0])
        (directory / "run.sh").write_text("exit 0\n")
        with self.store.edit() as state:
            state["paused"] = False
        self.assertEqual(self.run_worker(), 2)
        self.assertFalse((directory / "execution.json").exists())

    def test_status_snapshot_does_not_write_state(self):
        self.enqueue()
        path = self.store.directory / "state.json"
        before = path.read_bytes()
        self.store.snapshot()
        self.assertEqual(path.read_bytes(), before)

    def test_recovery_marks_unknown_completion_blocked(self):
        self.enqueue()
        with self.store.edit() as state:
            state["active"] = {"job": "first", "pid": None}
        with patch("train_nnue.research_loop.ROOT", self.root):
            self.assertEqual(main(["recover"]), 0)
        state = self.store.snapshot()
        self.assertIsNone(state["active"])
        self.assertTrue(state["paused"])
        self.assertEqual(state["jobs"][0]["status"], "blocked")

    def test_phase_limit_stops_after_replan_review(self):
        self.mode.write_text('replan')
        self.enqueue()
        self.assertEqual(Worker(self.root,self.store,self.config).run(max_phases=2),0)
        job=self.store.snapshot()['jobs'][0]
        self.assertEqual(job['phase'],'prepare')
        self.assertEqual(len(job['attempts']),1)
        self.assertEqual(len(job['replan_history']),1)

    def test_phase_limit_stops_after_prepare_without_compute(self):
        self.enqueue()
        self.assertEqual(Worker(self.root,self.store,self.config).run(max_phases=1),0)
        job=self.store.snapshot()['jobs'][0]
        self.assertEqual(job['phase'],'execute')
        self.assertFalse((Path(job['attempts'][0])/'execution.json').exists())

    def test_apply_saved_review_omits_only_explicit_existing_job_and_stays_paused(self):
        self.enqueue()
        self.enqueue('existing','H-TWO')
        directory=self.store.directory/'saved'
        directory.mkdir()
        proposal=dict(id='existing',hypothesis='H-TWO',task='Reworded task',depends_on=[])
        result=dict(status='replan',summary='repair',commit='abc1234',report='report',next_jobs=[proposal])
        write_json(directory/'review.json',result)
        write_json(directory/'review-process.json',dict(exit_code=0,reason='exited'))
        write_json(directory/'resolution.json',dict(obstacle='bug',attempted='checked',next_task='fix'))
        config=self.root/'config.json'
        write_json(config,self.config)
        with self.store.edit() as state:
            state['paused']=True
            state['jobs'][0].update(status='blocked',phase='review',attempts=[str(directory)])
        argv=['--config',str(config),'apply-review','first']
        with patch('train_nnue.research_loop.ROOT',self.root), patch.object(Worker,'validate_review'):
            self.assertEqual(main(argv),2)
            self.assertEqual(self.store.snapshot()['jobs'][0]['status'],'blocked')
            self.assertEqual(main(argv+['--omit-existing-job','existing']),0)
            self.assertEqual(main(argv+['--omit-existing-job','existing']),2)
        state=self.store.snapshot()
        self.assertTrue(state['paused'])
        self.assertEqual(state['jobs'][0]['phase'],'prepare')
        self.assertEqual(len(state['jobs'][0]['replan_history']),1)
        self.assertEqual(len(state['jobs']),2)
        self.assertEqual(len(state['review_applications']),1)
        self.assertEqual(json.loads((directory/'review.json').read_text()),result)
        self.assertEqual(self.events(),[])


if __name__ == "__main__":
    unittest.main()
