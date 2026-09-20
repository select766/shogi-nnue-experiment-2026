"""Disk-backed research queue; separate fresh Codex preparation/review processes.

Linux only (flock, process groups). No API SDK and no model process during compute.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
JOB_KEYS = {"id", "hypothesis", "task", "depends_on"}
ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
OPEN = {"unverified", "in_progress", "held"}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".atomic-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text())


def registry(root):
    text = (root / "docs/research/hypotheses.md").read_text()
    entries = re.findall(r"<!-- hypothesis id=(H-[A-Z0-9-]+) status=([a-z_]+)(?: priority=(\d+))? -->", text)
    return {key: {"status": status, "priority": int(priority) if priority else None}
            for key, status, priority in entries}


def validate_job(job, hypotheses):
    if set(job) != JOB_KEYS or not isinstance(job["id"], str) or not ID.fullmatch(job["id"]):
        raise ValueError("job needs id, hypothesis, task, depends_on; invalid ID")
    if job["hypothesis"] not in hypotheses:
        raise ValueError(f"unregistered hypothesis: {job['hypothesis']}")
    if hypotheses[job["hypothesis"]]["status"] not in OPEN:
        raise ValueError(f"job hypothesis must be unfinished: {job['hypothesis']}")
    if not isinstance(job["task"], str) or not job["task"].strip():
        raise ValueError("job task must be nonempty text")
    if not isinstance(job["depends_on"], list) or any(
        not isinstance(v, str) or not ID.fullmatch(v) for v in job["depends_on"]
    ) or job["id"] in job["depends_on"]:
        raise ValueError("invalid dependencies")


def job_schema():
    return {"type": "object", "additionalProperties": False,
            "properties": {"id": {"type": "string"}, "hypothesis": {"type": "string"},
                           "task": {"type": "string"},
                           "depends_on": {"type": "array", "items": {"type": "string"}}},
            "required": sorted(JOB_KEYS)}


def schema(phase):
    properties = ({"status": {"type": "string", "enum": ["ready", "blocked"]},
                   "summary": {"type": "string"}, "timeout_seconds": {"type": "integer"}}
                  if phase == "prepare" else
                  {"status": {"type": "string", "enum": ["complete", "blocked"]},
                   "summary": {"type": "string"}, "commit": {"type": "string"},
                   "report": {"type": "string"},
                   "next_jobs": {"type": "array", "items": job_schema()}})
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


class Store:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def edit(self):
        with (self.directory / "state.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / "state.json"
            if not path.exists():
                raise ValueError("queue is not initialized; use init")
            state = read_json(path)
            yield state
            state["updated_at"] = now()
            write_json(path, state)

    def snapshot(self):
        # Readers use the same lock; writes use atomic rename.
        with (self.directory / "state.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            path = self.directory / "state.json"
            if not path.exists():
                raise ValueError("queue is not initialized; use init")
            return read_json(path)

    @contextmanager
    def worker_lock(self):
        with (self.directory / "worker.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("another worker or its child still owns this queue") from None
            yield lock

    def worker_present(self):
        try:
            with self.worker_lock():
                return False
        except ValueError:
            return True


def add_jobs(state, jobs, hypotheses):
    existing = {v["id"] for v in state["jobs"]}
    added = set()
    for job in jobs:
        validate_job(job, hypotheses)
        if job["id"] in existing | added:
            raise ValueError(f"duplicate job: {job['id']}")
        added.add(job["id"])
    all_jobs = {j["id"]: j for j in state["jobs"] + jobs}
    visiting, visited = set(), set()

    def visit(key):
        if key not in all_jobs:
            raise ValueError(f"missing dependency: {key}")
        if key in visiting:
            raise ValueError("dependency cycle")
        if key in visited:
            return
        visiting.add(key)
        for dependency in all_jobs[key]["depends_on"]:
            visit(dependency)
        visiting.remove(key)
        visited.add(key)

    for key in all_jobs:
        visit(key)
    for job in jobs:
        state["jobs"].append({**job, "status": "queued", "phase": "prepare",
                              "attempts": [], "created_at": now(), "message": ""})


def choose(state, hypotheses):
    # Finish a previously prepared/computed job before another edits the workspace.
    ongoing = [j for j in state["jobs"] if j["status"] == "queued" and j["phase"] != "prepare"]
    if ongoing:
        return ongoing[0]
    done = {j["id"] for j in state["jobs"] if j["status"] == "done"}
    candidates = [j for j in state["jobs"] if j["status"] == "queued"
                  and set(j["depends_on"]) <= done
                  and hypotheses.get(j["hypothesis"], {}).get("status") in OPEN]
    if not candidates:
        return None
    return min(candidates, key=lambda j: (hypotheses[j["hypothesis"]]["priority"] or 10**9,
                                         j["created_at"], j["id"]))


COMMON = """あなたはtrain-nnueの研究ジョブ担当です。このセッションは新規であり、過去の会話を仮定しない。
AGENTS.md、docs/README.md、operationsのPython環境・学習評価、仮説台帳を読む。
今回のjob.jsonと引継ぎファイルを主な作業範囲にし、別実験の大量ログを無差別に読み込まない。
1コマンドの実行は最大300秒。短い検査もtimeout 300を使う。5分超の計算、学習、監視・待機は
このセッションでは実行せず、外部run.shまたは次ジョブにする。バックグラウンド起動も禁止。
Pythonはプロジェクトwrapperを使い、学習出力は/tmp/*.logへ保存する。GPU fallbackは禁止。
キューのstate.jsonを直接編集しない。出力JSONを管理プロセスが検査して反映する。
無関係な変更の削除・巻戻し、リモートへのpush、外部へのメッセージ送信はしない。
"""


def prompt(phase, root, run_dir):
    text = COMMON + f"\nRepository: {root}\n実験フォルダ: {run_dir}\njob.jsonを読んで開始する。\n"
    if phase == "prepare":
        return text + """
このセッションの役割は実装と実行準備。実験本体を起動せず、以下を実験フォルダに生成して終了する。
- plan.md: 仮説、対照、データ分離、成功/棄却/保留条件、費用、打切り規則。
- handoff.md: 別セッションが解釈できる実装、入力、出力、既知の制約、変更ファイル一覧。
- run.sh: bashで実行可能な本体。cwdはrepository root。RUN_DIRは本実験フォルダの絶対パス。
  set -euo pipefailを使い、外部プロセスをforegroundで待つ。Codex/APIを呼ばない。daemon化しない。
  結果はRUN_DIR/artifacts/へ、大量stdoutは/tmp/*.logへ。RUN_DIR/result-summary.mdへ
  読みやすい64KiB以内の要約と詳細成果物パスを保存する。失敗時にも分かる進捗・ログを残す。
  retry時の挙動（checkpoint再開か最初からか）をhandoff.mdへ明記する。
必要なrepoコードと5分以内の検査を実装し、対象仮説を検証中にする。新仮説は台帳に安定IDで登録する。
run.shを自分で起動してはいけない。prepare後にあなたのプロセスが終了してから外側が実行する。
最終JSONのstatusはreadyかblocked。timeout_secondsは本体の上限秒（job.jsonの設定上限以下）。
blockedならsummaryに不足情報を明記し、長時間待たず終了する。
"""
    return text + """
このセッションの役割は結果解釈と研究判断。plan.md、handoff.md、execution.json、
result-summary.md（存在すれば）、必要なartifactsだけを読み、ログはtail/rgで絞る。
exit code 0だけで成功扱いしない。計画の標本数・完遂・データ分離を検査し、失敗/timeoutは
未完了または保留とする。追加で5分超の計算が必要ならnext_jobsへ独立課題を追加して終了する。
analysis.mdへ判断と次の課題を記録し、docs/research/results/へ恒久的な結果文書を作る。
仮説台帳の判定・新仮説・未完了優先順位・結果登録を更新し、
timeout 300 scripts/project_python.sh scripts/check_research_hypotheses.py を通す。
この実験に属する実装・結果文書・台帳をコミットする（既存の無関係な変更を含めない）。
submoduleを変えた場合は先にそのコミットを作る。run.sh/plan/handoffは外側が既にsnapshotした。
最終JSON: status completeは検証結果の記録・コミットまで完了した意味（仮説支持とは別）。
commitは実在コミットSHA、reportはrepo相対の恒久結果文書パス、summaryは判断の短い要約。
next_jobsはid/hypothesis/task/depends_onの配列。台帳に登録済みの未完了仮説から、
独立した具体的課題を作る。実施価値がある未完了仮説を残すなら必要な次ジョブを提案する。
依存する次ジョブのdepends_onには今回のjob IDを入れる。長時間計算を自分で開始しない。
権限・入力不足などで記録/コミットを完遂できなければblockedとし、理由をsummaryに書く。
モデルを作った実験ではconfigs/research_champion.jsonの現最良候補と比較する。
独立した選抜実験の根拠があれば最良候補manifestを新ID・checkpoint・配備ファイル・根拠文書付きで更新し、
実験コミットへ含める。日次growth評価は監視専用で、同データでの係数調整や候補選抜をしない。
単なる新checkpointやloss改善だけで最良候補を置き換えない。不確定なら現候補を維持する。
"""


def terminate_group(process, grace):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit before its descendants. Always clean the entire group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class Worker:
    def __init__(self, root, store, config):
        self.root, self.store, self.config = Path(root), store, config
        self.stopping = False

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True).strip()

    def block(self, job_id, message):
        with self.store.edit() as state:
            job = next(j for j in state["jobs"] if j["id"] == job_id)
            job.update(status="blocked", message=message)
            state.update(active=None, paused=True, interrupt=False)

    def run_process(self, command, phase, run_dir, job_id, timeout, lock, input_path=None):
        fd, log_name = tempfile.mkstemp(prefix=f"research-{job_id}-{phase}-", suffix=".log", dir="/tmp")
        os.close(fd)
        link = run_dir / f"{phase}.log"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(log_name)
        started = time.monotonic()
        with open(log_name, "ab") as log, (input_path or Path("/dev/null")).open("rb") as source:
            env = {**os.environ, "RUN_DIR": str(run_dir), "RESEARCH_JOB_ID": job_id}
            process = subprocess.Popen(command, cwd=self.root, stdin=source, stdout=log, stderr=log,
                                       start_new_session=True,
                                       pass_fds=(lock.fileno(), self.workspace_fd), env=env)
            with self.store.edit() as state:
                state["active"].update(pid=process.pid, log=log_name, heartbeat=now())
            reason = "exited"
            try:
                while process.poll() is None:
                    with self.store.edit() as state:
                        state["active"]["heartbeat"] = now()
                        interrupt = state["interrupt"]
                    if self.stopping or interrupt:
                        reason = "interrupted"
                        break
                    if time.monotonic() - started >= timeout:
                        reason = "timeout"
                        break
                    time.sleep(self.config["poll_seconds"])
            finally:
                terminate_group(process, self.config["terminate_grace_seconds"])
            result = {"phase": phase, "command": command, "exit_code": process.returncode,
                      "reason": reason, "elapsed_seconds": time.monotonic() - started,
                      "finished_at": now(), "log": log_name}
            write_json(run_dir / f"{phase}-process.json", result)
            sequence = len(list(run_dir.glob(f"{phase}-process-[0-9]*.json"))) + 1
            write_json(run_dir / f"{phase}-process-{sequence:03d}.json", result)
            return result

    def validate_review(self, result, run_dir):
        if result["status"] != "complete":
            raise ValueError(result["summary"])
        sha, report = result["commit"], result["report"]
        if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
            raise ValueError("review did not supply a commit SHA")
        baseline = read_json(run_dir / "job.json")["base_commit"]
        if self.git("rev-parse", sha) == baseline:
            raise ValueError("review requires a new experiment commit")
        self.git("merge-base", "--is-ancestor", baseline, sha)
        if not (run_dir / "analysis.md").is_file():
            raise ValueError("review did not leave analysis.md")
        target = (self.root / report).resolve()
        if not target.is_relative_to(self.root / "docs/research/results") or not target.is_file():
            raise ValueError("review report must be a result document in the repository")
        self.git("merge-base", "--is-ancestor", sha, "HEAD")
        if self.git("show", f"{sha}:{report}") != target.read_text().strip():
            raise ValueError("report differs from committed content")
        for path in ("docs/research/hypotheses.md",):
            if self.git("show", f"{sha}:{path}") != (self.root / path).read_text().strip():
                raise ValueError("registry differs from review commit")
        if self.git("status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("review left uncommitted changes; inspect and retry review")
        subprocess.run([str(self.root / "scripts/project_python.sh"),
                        "scripts/check_research_hypotheses.py"], cwd=self.root,
                       check=True, timeout=300, stdout=subprocess.DEVNULL)

    def phase(self, job, lock):
        phase, job_id = job["phase"], job["id"]
        with self.store.edit() as state:
            if state["paused"]:
                return
            state["active"] = {"job": job_id, "phase": phase, "run_dir": None,
                               "worker_pid": os.getpid(), "started_at": now(), "pid": None}
        if phase == "prepare":
            if not job["attempts"] and self.git("status", "--porcelain", "--untracked-files=normal"):
                raise ValueError("workspace is dirty; commit or isolate changes before preparing a new job")
            run_dir = self.store.directory / "experiments" / job_id / f"attempt-{len(job['attempts']) + 1:03d}"
            run_dir.mkdir(parents=True)
            (run_dir / "artifacts").mkdir()
            write_json(run_dir / "job.json", {**job, "base_commit": self.git("rev-parse", "HEAD"),
                                              "max_compute_seconds": self.config["max_compute_seconds"]})
            with self.store.edit() as state:
                saved = next(j for j in state["jobs"] if j["id"] == job_id)
                saved["attempts"].append(str(run_dir))
        else:
            run_dir = Path(job["attempts"][-1])
        with self.store.edit() as state:
            state["active"] = {"job": job_id, "phase": phase, "run_dir": str(run_dir),
                               "worker_pid": os.getpid(), "started_at": now(), "pid": None}
        if job.get("kind") == "daily_benchmark":
            process = self.run_process(
                [str(self.root / "scripts/project_python.sh"), "-m", "train_nnue.daily_benchmark",
                 "--run-dir", str(run_dir)], "execute", run_dir, job_id, job["timeout_seconds"], lock)
            write_json(run_dir / "execution.json", process)
            if process["exit_code"] != 0 or process["reason"] != "exited":
                raise ValueError(f"daily benchmark failed: {process['reason']}; inspect logs and retry execute")
            completion = read_json(run_dir / "benchmark-complete.json")
            with self.store.edit() as state:
                saved = next(j for j in state["jobs"] if j["id"] == job_id)
                saved.update(status="done", completed_at=now(), message=completion["dashboard"])
                state["active"] = None
            return
        if phase in {"prepare", "review"}:
            output = run_dir / f"{phase}.json"
            # Preserve failed session outputs on retry; never accept an old answer.
            index = len(list(run_dir.glob(f"{phase}-prompt-*.md"))) + 1
            if output.exists():
                output.rename(run_dir / f"{phase}-previous-{index:03d}.json")
            prompt_path = run_dir / f"{phase}-prompt-{index:03d}.md"
            prompt_path.write_text(prompt(phase, self.root, run_dir))
            schema_path = run_dir / f"{phase}.schema.json"
            write_json(schema_path, schema(phase))
            command = self.config["codex_command"] + ["--cd", str(self.root), "--output-schema",
                        str(schema_path), "--add-dir", str(self.store.directory),
                        "--output-last-message", str(output), "-"]
            process = self.run_process(command, phase, run_dir, job_id,
                                       self.config["session_timeout_seconds"], lock, prompt_path)
            if process["exit_code"] != 0 or process["reason"] != "exited":
                raise ValueError(f"{phase} session failed: {process}; see log")
            result = read_json(output)
            if set(result) != set(schema(phase)["required"]):
                raise ValueError("unexpected structured response keys")
            if phase == "prepare":
                if result["status"] != "ready":
                    raise ValueError(result["summary"])
                seconds = result["timeout_seconds"]
                if type(seconds) is not int or not 1 <= seconds <= self.config["max_compute_seconds"]:
                    raise ValueError("compute timeout is outside configured limit")
                snapshots = {}
                for name in ("plan.md", "handoff.md", "run.sh"):
                    path = run_dir / name
                    if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                        raise ValueError(f"missing or invalid {name}")
                    snapshots[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                subprocess.run(["bash", "-n", str(run_dir / "run.sh")], check=True, timeout=10)
                write_json(run_dir / "prepared.json", {"sha256": snapshots, "timeout_seconds": seconds})
                next_phase = "execute"
            else:
                self.validate_review(result, run_dir)
                with self.store.edit() as state:
                    add_jobs(state, result["next_jobs"], registry(self.root))
                    saved = next(j for j in state["jobs"] if j["id"] == job_id)
                    saved.update(status="done", message=result["summary"], commit=result["commit"])
                    state["active"] = None
                return
        else:
            prepared = read_json(run_dir / "prepared.json")
            for name, digest in prepared["sha256"].items():
                if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"prepared file changed: {name}; retry preparation")
            process = self.run_process(["bash", str(run_dir / "run.sh")], phase, run_dir, job_id,
                                       prepared["timeout_seconds"], lock)
            write_json(run_dir / "execution.json", process)
            if process["reason"] == "interrupted":
                raise ValueError("execution interrupted; inspect artifacts before retry")
            # A failed computation also needs a fresh interpretation session.
            next_phase = "review"
        with self.store.edit() as state:
            saved = next(j for j in state["jobs"] if j["id"] == job_id)
            saved.update(phase=next_phase, message=f"{phase} finished")
            state["active"] = None

    def run(self, max_jobs=0):
        # A second state directory must not circumvent workspace mutual exclusion.
        workspace = Store(self.root / ".research-loop")
        with workspace_lock(workspace) as workspace_file, self.store.worker_lock() as lock:
            self.workspace_fd = workspace_file.fileno()
            state = self.store.snapshot()
            if state["active"] is not None:
                raise ValueError("unclean previous exit; use recover after checking recorded processes")
            completed = 0
            previous = {}
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, lambda *_: setattr(self, "stopping", True))
            try:
                while not self.stopping:
                    state = self.store.snapshot()
                    if state["paused"]:
                        return 0
                    if any(j["status"] == "blocked" for j in state["jobs"]):
                        return 2
                    daily = self.config.get("daily_benchmark", {})
                    ongoing = any(j["status"] == "queued" and j["phase"] != "prepare" for j in state["jobs"])
                    if daily.get("enabled") and not ongoing:
                        from train_nnue.daily_benchmark import schedule
                        try:
                            schedule(self.root, self.store, daily)
                        except Exception as error:
                            with self.store.edit() as saved:
                                saved.update(paused=True, benchmark_error=str(error))
                            print(f"Daily scheduling blocked: {error}", file=sys.stderr)
                            return 2
                        state = self.store.snapshot()
                    job = choose(state, registry(self.root))
                    if job is None:
                        if daily.get("enabled") and not max_jobs:
                            time.sleep(self.config["poll_seconds"])
                            continue
                        return 0
                    try:
                        self.phase(job, lock)
                    except Exception as error:
                        self.block(job["id"], str(error))
                        print(f"Blocked {job['id']}: {error}", file=sys.stderr)
                        return 2
                    state = self.store.snapshot()
                    if job.get("kind") != "daily_benchmark" and next(j for j in state["jobs"] if j["id"] == job["id"])["status"] == "done":
                        completed += 1
                    if max_jobs and completed >= max_jobs:
                        return 0
                return 0
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)


@contextmanager
def workspace_lock(store):
    with (store.directory / "workspace.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another queue is using this repository") from None
        yield lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / ".research-loop")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research_loop.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    add = sub.add_parser("enqueue")
    add.add_argument("job_file", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--max-jobs", type=int, default=0, help="0: until queue empty or paused")
    pause = sub.add_parser("pause")
    pause.add_argument("--now", action="store_true", help="terminate active process group")
    sub.add_parser("resume")
    retry = sub.add_parser("retry")
    retry.add_argument("job_id")
    retry.add_argument("--phase", choices=["prepare", "execute", "review"], required=True)
    cancel = sub.add_parser("cancel")
    cancel.add_argument("job_id")
    sub.add_parser("recover")
    sub.add_parser("benchmark-init")
    sub.add_parser("benchmark-enqueue")
    args = parser.parse_args(argv)
    store = Store(args.state_dir)
    try:
        if args.command == "init":
            with (store.directory / "state.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                path = store.directory / "state.json"
                if path.exists():
                    raise ValueError("queue already initialized")
                write_json(path, {"version": 1, "paused": False, "interrupt": False,
                                  "active": None, "jobs": [], "updated_at": now()})
        elif args.command == "status":
            state = store.snapshot()
            state["worker_present"] = store.worker_present()
            hypotheses = registry(ROOT)
            for job in state["jobs"]:
                job["priority"] = hypotheses.get(job["hypothesis"], {}).get("priority")
            if args.config.exists():
                from train_nnue.daily_benchmark import due
                settings = read_json(args.config).get("daily_benchmark", {})
                state["daily_benchmark"] = {"enabled": settings.get("enabled", False),
                                             "due": due(state, settings.get("interval_seconds", 86400)),
                                             "config": settings.get("config")}
            print(json.dumps(state, indent=2, ensure_ascii=False))
        elif args.command == "enqueue":
            with store.edit() as state:
                add_jobs(state, [read_json(args.job_file)], registry(ROOT))
        elif args.command in {"benchmark-init", "benchmark-enqueue"}:
            from train_nnue.daily_benchmark import initialize, schedule
            settings = read_json(args.config)["daily_benchmark"]
            with workspace_lock(Store(ROOT / ".research-loop")), store.worker_lock():
                if args.command == "benchmark-init":
                    print(initialize(ROOT, store.directory, Path(settings["config"])))
                else:
                    print(schedule(ROOT, store, settings, force=True))
        elif args.command in {"pause", "resume"}:
            with store.edit() as state:
                state["paused"] = args.command == "pause"
                state["interrupt"] = args.command == "pause" and args.now
        elif args.command in {"retry", "recover", "cancel"}:
            with store.worker_lock(), store.edit() as state:
                if args.command == "recover":
                    active = state["active"]
                    if active:
                        if active.get("pid"):
                            try:
                                os.killpg(active["pid"], 0)
                            except ProcessLookupError:
                                pass
                            else:
                                raise ValueError("recorded process group still exists; stop it before recovery")
                        job = next(j for j in state["jobs"] if j["id"] == active["job"])
                        job.update(status="blocked", message="unclean exit; inspect before explicit retry")
                    state.update(active=None, paused=True, interrupt=False)
                elif args.command == "cancel":
                    if state["active"]:
                        raise ValueError("pause and finish/recover the active phase before cancel")
                    job = next((j for j in state["jobs"] if j["id"] == args.job_id), None)
                    if job is None or job["status"] == "done":
                        raise ValueError("cancel requires an unfinished job")
                    job.update(status="cancelled", message="cancelled by operator; artifacts preserved")
                else:
                    if state["active"]:
                        raise ValueError("recover the previous active phase first")
                    job = next((j for j in state["jobs"] if j["id"] == args.job_id), None)
                    if job is None or job["status"] != "blocked":
                        raise ValueError("retry requires a blocked job")
                    if job.get("kind") == "daily_benchmark" and args.phase != "execute":
                        raise ValueError("daily benchmark retry uses --phase execute (no Codex session)")
                    if args.phase != "prepare":
                        if not job["attempts"]:
                            raise ValueError("no prior attempt to retry")
                        required = "benchmark.json" if job.get("kind") == "daily_benchmark" else "prepared.json" if args.phase == "execute" else "execution.json"
                        if not (Path(job["attempts"][-1]) / required).exists():
                            raise ValueError(f"retry requires {required}")
                    job.update(status="queued", phase=args.phase, message="explicit retry requested")
        elif args.command == "run":
            config = read_json(args.config)
            if args.max_jobs < 0 or any(config[k] <= 0 for k in (
                "session_timeout_seconds", "max_compute_seconds", "poll_seconds", "terminate_grace_seconds"
            )):
                raise ValueError("limits must be positive")
            if not config["codex_command"] or not all(isinstance(v, str) for v in config["codex_command"]):
                raise ValueError("codex_command must be a nonempty argv array")
            if any(v in {"resume", "fork", "--last"} for v in config["codex_command"]):
                raise ValueError("research phases must use fresh sessions, not resume/fork")
            return Worker(ROOT, store, config).run(args.max_jobs)
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"research-loop: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
