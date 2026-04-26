"""
Expert Blending 評価関数合成 (新アーキ: ExpertBlendingDir 直ロード) の速度測定。

iter6 以降、Python サブプロセス + IPC は廃止し、やねうら王が
backbone.onnx + head.bin + head.json をロードして C++ 内で完結させる。
よって本スクリプトの役割は「合成 + 探索の wall-clock を測る」のみ。

Usage:
    bash scripts/benchmark_expert_blending_speed.sh \\
        logs/.../checkpoints/180.ckpt

スクリプト内部処理:
    1. checkpoint + dlshogi 重みから export_for_yaneuraou.py で
       一時ディレクトリに backbone.onnx / head.bin / head.json を生成
       (--blending-dir で既存ディレクトリを指定すれば skip)。
    2. やねうら王に EvalDir + ExpertBlendingDir を設定して isready。
    3. 複数局面で go nodes <N> → bestmove までの wall-clock を計測。
"""

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


STARTPOS_MOVES = [
    "",
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


def export_blending_dir(checkpoint, backbone_weights, n_experts, features, out_dir):
    """既存の export_for_yaneuraou.py を呼んで blending dir を作る。"""
    print(f"Exporting expert blending dir to: {out_dir}")
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join([
        str(repo_root / "src"),
        str(repo_root / "dlshogi-source"),
        str(repo_root / "nnue-pytorch"),
        env.get("PYTHONPATH", ""),
    ]).rstrip(":")
    subprocess.run(
        [sys.executable, "-m", "train_nnue.export_for_yaneuraou",
         "--checkpoint", str(checkpoint),
         "--backbone-weights", str(backbone_weights),
         "--features", features,
         "--n-experts", str(n_experts),
         "--output-dir", str(out_dir)],
        check=True, env=env,
    )


def main():
    parser = argparse.ArgumentParser(description="Expert blending speed benchmark (new arch)")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Expert blending checkpoint (.ckpt). --blending-dir 指定時は不要")
    parser.add_argument("--backbone-weights", default=None,
                        help="dlshogi backbone weights")
    parser.add_argument("--features", default="HalfKP")
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--baseline-eval-dir", default=None,
                        help="EvalDir (nn.bin の置き場)。省略時は repo の bin/eval")
    parser.add_argument("--blending-dir", default=None,
                        help="既存の expert blending dir (backbone.onnx + head.bin)。"
                             "省略時は --checkpoint から一時ディレクトリにエクスポートする")
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--onnxruntime-libdir", default=None,
                        help="LD_LIBRARY_PATH に追加する onnxruntime lib ディレクトリ")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    engine = Path(args.engine) if args.engine else repo_root / "bin" / "YaneuraOu-expert-blending"
    eval_dir = Path(args.baseline_eval_dir) if args.baseline_eval_dir else repo_root / "bin" / "eval"

    if not engine.is_file():
        sys.exit(f"engine not found: {engine}")
    if not (eval_dir / "nn.bin").is_file():
        sys.exit(f"baseline eval not found: {eval_dir}/nn.bin")

    # blending dir 準備
    tmpdir_obj = None
    if args.blending_dir:
        blending_dir = Path(args.blending_dir).resolve()
        if not (blending_dir / "head.bin").is_file():
            sys.exit(f"--blending-dir does not contain head.bin: {blending_dir}")
    else:
        if not args.checkpoint or not args.backbone_weights:
            sys.exit("--blending-dir または (--checkpoint + --backbone-weights) のどちらかが必要")
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="expert_blending_bench_")
        blending_dir = Path(tmpdir_obj.name)
        export_blending_dir(
            Path(args.checkpoint).resolve(),
            Path(args.backbone_weights).resolve(),
            args.n_experts, args.features, blending_dir,
        )

    # onnxruntime lib path
    env = dict(os.environ)
    libdir = args.onnxruntime_libdir
    if libdir is None:
        candidate = repo_root / "YaneuraOu" / "extra" / "onnxruntime" / "linux" / "current" / "lib"
        if candidate.is_dir():
            libdir = str(candidate)
    if libdir:
        env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    print("=== Expert Blending Speed Benchmark (iter6+: in-process) ===")
    print(f"  engine          : {engine}")
    print(f"  blending dir    : {blending_dir}")
    print(f"  EvalDir         : {eval_dir}")
    print(f"  nodes           : {args.nodes}")
    print(f"  iters           : {args.iters}  (warmup discarded: {args.warmup})")
    print(f"  threads         : {args.threads}")
    print()

    proc = subprocess.Popen(
        [str(engine)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(engine.parent), env=env,
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

    per_iter = []
    try:
        send("usi")
        wait_for("usiok")
        send(f"setoption name EvalDir value {eval_dir}")
        send(f"setoption name ExpertBlendingDir value {blending_dir}")
        send(f"setoption name Threads value {args.threads}")

        print("Initializing engine (isready)...")
        t_ready_start = time.time()
        send("isready")
        wait_for("readyok", timeout=600.0)
        t_ready = time.time() - t_ready_start
        print(f"  isready elapsed: {t_ready:.2f}s\n")

        send("usinewgame")
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
    print(f"=== Wall-clock per `go nodes {args.nodes}` ===")
    summarize("warmup ", per_iter[: args.warmup])
    summarize("measure", measured)

    if tmpdir_obj is not None:
        tmpdir_obj.cleanup()


if __name__ == "__main__":
    main()
