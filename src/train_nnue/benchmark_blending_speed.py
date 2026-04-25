"""
エキスパート合成 (Expert Blending) の速度測定スクリプト。

YaneuraOu-expert-blending エンジンに対して `go nodes <N>` を投げ、
bestmove が返るまでの時間を計測する。N を小さく取ること (例: 1000) で、
探索時間より合成パイプライン (Python 推論 + IPC + UpdateWeightsFromBuffer)
が支配的になり、合成コストの代理指標として使える。

dnn_inference_server のログからは ``infer=...s blend=...s`` が読み取れるため、
そちらも併せてサマリ表示する。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.benchmark_blending_speed \
        --checkpoint ../logs/expert_blending_8experts_v4_paired_uniform50_noise0_lambda05/checkpoints/180.ckpt \
        --backbone-weights ../tmp/dlshogi-model/model_resnet10_swish-072 \
        --baseline-eval-dir ../bin/eval \
        --n-experts 8 \
        --nodes 1000 \
        --iters 20

または scripts/benchmark_expert_blending_speed.sh から起動する。
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


STARTPOS_MOVES = [
    "",  # 初期局面
    "7g7f",
    "7g7f 3c3d",
    "7g7f 3c3d 2g2f",
    "7g7f 3c3d 2g2f 8c8d",
    "7g7f 3c3d 2g2f 8c8d 2f2e",
    "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e",
    "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h",
    "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b",
    "7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d",
]


def position_command(moves: str) -> str:
    if not moves:
        return "position startpos"
    return f"position startpos moves {moves}"


def parse_server_log(log_path: Path):
    """dnn_inference_server.log から infer/blend/size を取り出す。"""
    if not log_path.is_file():
        return []
    pattern = re.compile(r"\[(\d+)\].*infer=([0-9.]+)s blend=([0-9.]+)s size=(\d+)")
    rows = []
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            idx, infer_s, blend_s, size = m.groups()
            rows.append(
                {
                    "i": int(idx),
                    "infer_s": float(infer_s),
                    "blend_s": float(blend_s),
                    "size": int(size),
                }
            )
    return rows


def summarize(label: str, values):
    if not values:
        print(f"  {label}: (no samples)")
        return
    n = len(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if n >= 2 else 0.0
    vmin = min(values)
    vmax = max(values)
    median = statistics.median(values)
    print(
        f"  {label}: n={n} mean={mean*1000:.1f}ms median={median*1000:.1f}ms "
        f"min={vmin*1000:.1f}ms max={vmax*1000:.1f}ms stdev={stdev*1000:.1f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Expert blending speed benchmark")
    parser.add_argument("--engine", default=None,
                        help="YaneuraOu-expert-blending のパス (省略時は repo の bin/)")
    parser.add_argument("--checkpoint", required=True,
                        help="Expert blending checkpoint (.ckpt)")
    parser.add_argument("--backbone-weights", default=None,
                        help="dlshogi backbone weights (--backbone-type dnn のとき必要)")
    parser.add_argument("--backbone-type", default="dnn", choices=["dnn", "nnue"])
    parser.add_argument("--features", default="HalfKP")
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--baseline-eval-dir", default=None,
                        help="EvalDir (nn.bin の置き場)。省略時は repo の bin/eval")
    parser.add_argument("--nodes", type=int, default=1000,
                        help="go nodes の値 (デフォルト 1000)")
    parser.add_argument("--iters", type=int, default=20,
                        help="計測する go の回数")
    parser.add_argument("--warmup", type=int, default=2,
                        help="集計から除外する初回 go の数")
    parser.add_argument("--threads", type=int, default=1,
                        help="USI Threads (デフォルト 1)")
    parser.add_argument("--server-log", default=None,
                        help="dnn_inference_server のログ出力先 (省略時はテンポラリ)")
    parser.add_argument("--python", default=None,
                        help="DNN サーバーを動かす python (省略時は nnue-pytorch/.venv/bin/python)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    engine = Path(args.engine) if args.engine else repo_root / "bin" / "YaneuraOu-expert-blending"
    eval_dir = Path(args.baseline_eval_dir) if args.baseline_eval_dir else repo_root / "bin" / "eval"
    python_exe = Path(args.python) if args.python else repo_root / "nnue-pytorch" / ".venv" / "bin" / "python"
    checkpoint = Path(args.checkpoint).resolve()

    if not engine.is_file():
        sys.exit(f"engine not found: {engine}")
    if not (eval_dir / "nn.bin").is_file():
        sys.exit(f"baseline eval not found: {eval_dir}/nn.bin")
    if not python_exe.is_file():
        sys.exit(f"python not found: {python_exe}")
    if not checkpoint.is_file():
        sys.exit(f"checkpoint not found: {checkpoint}")
    if args.backbone_type == "dnn":
        if not args.backbone_weights:
            sys.exit("--backbone-weights is required for --backbone-type dnn")
        backbone_weights = Path(args.backbone_weights).resolve()
        if not backbone_weights.is_file():
            sys.exit(f"backbone weights not found: {backbone_weights}")
    else:
        backbone_weights = None

    # サーバー側ログ
    if args.server_log:
        server_log = Path(args.server_log).resolve()
        server_log.parent.mkdir(parents=True, exist_ok=True)
        # 古い行が残っていると infer=/blend= の集計に混ざるので消しておく
        if server_log.exists():
            server_log.unlink()
        cleanup_log = False
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="dnn_server_", suffix=".log", delete=False)
        tmp.close()
        server_log = Path(tmp.name)
        cleanup_log = True

    pythonpath = f"{repo_root / 'src'}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    cmd_parts = [
        f"PYTHONPATH={pythonpath}",
        str(python_exe),
        "-m", "train_nnue.dnn_inference_server",
        "--checkpoint", str(checkpoint),
        "--backbone-type", args.backbone_type,
        "--features", args.features,
        "--n-experts", str(args.n_experts),
        "--log", str(server_log),
    ]
    if backbone_weights:
        cmd_parts += ["--backbone-weights", str(backbone_weights)]
    dnn_cmd = " ".join(cmd_parts)

    print("=== Expert Blending Speed Benchmark ===")
    print(f"  engine          : {engine}")
    print(f"  checkpoint      : {checkpoint}")
    print(f"  backbone weights: {backbone_weights}")
    print(f"  EvalDir         : {eval_dir}")
    print(f"  n_experts       : {args.n_experts}")
    print(f"  nodes           : {args.nodes}")
    print(f"  iters           : {args.iters}  (warmup discarded: {args.warmup})")
    print(f"  threads         : {args.threads}")
    print(f"  server log      : {server_log}")
    print()

    proc = subprocess.Popen(
        [str(engine)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(engine.parent),
    )

    def send(cmd: str):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def wait_for(token: str, *, echo: bool = False, timeout: float = 600.0):
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"timeout waiting for '{token}'")
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f"engine EOF while waiting for '{token}'")
            line = line.rstrip("\n")
            if echo:
                print(f"  <<< {line}")
            if token in line:
                return line

    try:
        send("usi")
        wait_for("usiok")

        # オプション設定。EvalDir → DNNServerCmd → Threads の順。
        send(f"setoption name EvalDir value {eval_dir}")
        send(f"setoption name DNNServerCmd value {dnn_cmd}")
        send(f"setoption name Threads value {args.threads}")

        print("Initializing engine + DNN server (isready)...")
        t_ready_start = time.time()
        send("isready")
        wait_for("readyok", timeout=600.0)
        t_ready = time.time() - t_ready_start
        print(f"  isready elapsed: {t_ready:.2f}s\n")

        send("usinewgame")

        per_iter = []
        print(f"Running {args.iters} go-cycles ...")
        for i in range(args.iters):
            moves = STARTPOS_MOVES[i % len(STARTPOS_MOVES)]
            send(position_command(moves))
            t0 = time.time()
            send(f"go nodes {args.nodes}")
            wait_for("bestmove")
            elapsed = time.time() - t0
            per_iter.append(elapsed)
            tag = "warmup" if i < args.warmup else "measure"
            print(f"  [{i+1:3d}/{args.iters}] ({tag}) {elapsed*1000:7.1f} ms  moves='{moves[:40]}'")

        send("quit")
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()

    measured = per_iter[args.warmup:]

    print()
    print("=== Wall-clock per `go nodes %d` ===" % args.nodes)
    summarize("warmup ", per_iter[: args.warmup])
    summarize("measure", measured)

    # サーバー側ログから infer/blend を集計
    server_rows = parse_server_log(server_log)
    if server_rows:
        # warmup 分は除外。go の発行数 = サーバーへのリクエスト数 と仮定。
        srv_measure = server_rows[args.warmup :]
        infer = [r["infer_s"] for r in srv_measure]
        blend = [r["blend_s"] for r in srv_measure]
        sizes = [r["size"] for r in srv_measure]
        print()
        print("=== Python DNN server breakdown (from --log) ===")
        summarize("infer  ", infer)
        summarize("blend  ", blend)
        if sizes:
            print(f"  payload size: {sizes[0]} bytes")
    else:
        print()
        print("(server log had no infer=/blend= entries)")

    if cleanup_log:
        try:
            server_log.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
