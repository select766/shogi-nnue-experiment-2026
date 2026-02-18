"""Validate hypotheses for train_loss down / val_loss up behavior.

Generates a markdown report with concrete checks and results.

Usage:
  source nnue-pytorch/.venv/bin/activate
  PYTHONPATH=src python -m train_nnue.validate_val_loss_hypotheses \
    --logdir logs/expert_blending_8experts_v4_paired_noise0 \
    --train-bin dataset/split_v1_paired/train.bin \
    --val-bin dataset/split_v1_paired/val1.bin \
    --paired \
    --paired-cache-dir tmp/paired_nnue_cache \
    --output docs/val_loss_hypotheses_report.md
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LEGACY_RECORD_BYTES = 40
PAIRED_RECORD_BYTES = 80


@dataclass
class HypothesisResult:
    title: str
    method: str
    result: str
    verdict: str


def _load_events(logdir: str):
    version_dir = Path(logdir) / "lightning_logs" / "version_0"
    event_files = sorted(version_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {version_dir}")
    ea = EventAccumulator(str(event_files[-1]))
    ea.Reload()
    return ea, str(event_files[-1])


def _iter_epoch_slices(train_scalars, val_scalars):
    start_step = 0
    for v in val_scalars:
        xs = [t.value for t in train_scalars if start_step < t.step <= v.step]
        yield v.step, xs, v.value
        start_step = v.step


def validate_hypothesis_1_val_window_shift(val_bin: str, paired: bool, batch_size: int, max_val_positions: int):
    record_bytes = PAIRED_RECORD_BYTES if paired else LEGACY_RECORD_BYTES
    val_records = os.path.getsize(val_bin) // record_bytes
    num_val_batches = (min(val_records, max_val_positions) + batch_size - 1) // batch_size
    consumed_per_epoch = num_val_batches * batch_size
    shift = consumed_per_epoch % val_records
    cycle = val_records // math.gcd(val_records, shift) if shift else 1

    starts = []
    pos = 0
    for _ in range(6):
        starts.append(pos)
        pos = (pos + consumed_per_epoch) % val_records

    method = (
        "`create_data_loaders()` の式と同じ計算で val 1epoch あたりの消費局面数を算出し、"
        "循環読み出し時の開始オフセットをシミュレーション。"
    )
    result = (
        f"val_records={val_records}, batch_size={batch_size}, num_val_batches={num_val_batches}, "
        f"consumed/epoch={consumed_per_epoch}, shift/epoch={shift}, cycle={cycle}。"
        f" 先頭オフセット遷移(6epoch): {starts}"
    )
    verdict = "supported" if shift != 0 else "not_supported"
    return HypothesisResult(
        title="H1: 検証集合が毎epochでずれており val_loss が比較不能",
        method=method,
        result=result,
        verdict=verdict,
    )


def validate_hypothesis_2_logging_granularity(ea: EventAccumulator):
    tags = ea.Tags().get("scalars", [])
    if "train_loss" not in tags or "val_loss" not in tags:
        raise RuntimeError("train_loss/val_loss scalar tags not found")
    train = ea.Scalars("train_loss")
    val = ea.Scalars("val_loss")

    train_steps = [s.step for s in train]
    val_steps = [s.step for s in val]
    train_delta = float(np.mean(np.diff(train_steps))) if len(train_steps) > 1 else float("nan")
    val_delta = float(np.mean(np.diff(val_steps))) if len(val_steps) > 1 else float("nan")

    method = "TensorBoardイベントを直接読んで `train_loss` と `val_loss` の記録頻度を比較。"
    result = (
        f"train_loss points={len(train)}, val_loss points={len(val)}, "
        f"mean step delta(train)={train_delta:.1f}, mean step delta(val)={val_delta:.1f}。"
    )
    verdict = "supported" if len(train) > len(val) * 20 else "partially_supported"
    return HypothesisResult(
        title="H2: train/val のログ粒度差で見え方が歪む",
        method=method,
        result=result,
        verdict=verdict,
    )


def validate_hypothesis_3_overfitting_signal(ea: EventAccumulator):
    train = ea.Scalars("train_loss")
    val = ea.Scalars("val_loss")
    epoch_train = []
    epoch_val = []
    for _, xs, vy in _iter_epoch_slices(train, val):
        if xs:
            epoch_train.append(float(np.mean(xs)))
            epoch_val.append(float(vy))

    x = np.arange(len(epoch_train), dtype=np.float64)
    train_slope = float(np.polyfit(x, np.array(epoch_train), 1)[0]) if len(x) >= 2 else float("nan")
    val_slope = float(np.polyfit(x, np.array(epoch_val), 1)[0]) if len(x) >= 2 else float("nan")
    gap = np.array(epoch_val) - np.array(epoch_train)
    gap_slope = float(np.polyfit(x, gap, 1)[0]) if len(x) >= 2 else float("nan")

    method = (
        "val境界(step)ごとに train_loss を平均化し、epoch方向の一次傾き(train/val/gap)を計測。"
    )
    result = (
        f"slope(train)={train_slope:+.6f}, slope(val)={val_slope:+.6f}, slope(gap)={gap_slope:+.6f}; "
        f"first(val-train)={gap[0]:+.6f}, last(val-train)={gap[-1]:+.6f}"
    )
    verdict = "supported" if (train_slope < 0 and val_slope > 0 and gap_slope > 0) else "not_supported"
    return HypothesisResult(
        title="H3: 実際に過学習傾向が出ている",
        method=method,
        result=result,
        verdict=verdict,
    )


def _sample_indices(num_records: int, n: int, seed: int) -> list[int]:
    rnd = random.Random(seed)
    if num_records <= n:
        return list(range(num_records))
    return sorted(rnd.sample(range(num_records), n))


def validate_hypothesis_4_paired_misalignment(val_bin: str, paired_cache_dir: str | None, sample_n: int, seed: int):
    paired_path = Path(val_bin)
    cache_base = Path(paired_cache_dir) if paired_cache_dir else paired_path.parent
    cache_path = cache_base / f"{paired_path.name}.nnue40.bin"
    if not cache_path.exists():
        return HypothesisResult(
            title="H4: pairedデータ境界で DNN/NNUE がずれている",
            method="paired後半40Bと抽出キャッシュ(.nnue40.bin)の同一インデックス比較。",
            result=f"cache not found: {cache_path}",
            verdict="inconclusive",
        )

    paired_size = paired_path.stat().st_size
    cache_size = cache_path.stat().st_size
    num_records = paired_size // PAIRED_RECORD_BYTES
    if cache_size != num_records * LEGACY_RECORD_BYTES:
        return HypothesisResult(
            title="H4: pairedデータ境界で DNN/NNUE がずれている",
            method="paired後半40Bと抽出キャッシュ(.nnue40.bin)の同一インデックス比較。",
            result=(
                f"size mismatch: paired_size={paired_size}, cache_size={cache_size}, "
                f"expected_cache={num_records * LEGACY_RECORD_BYTES}"
            ),
            verdict="supported",
        )

    idxs = _sample_indices(num_records, sample_n, seed)
    mismatches = 0
    same_first_second = 0
    with paired_path.open("rb") as fp, cache_path.open("rb") as fc:
        for idx in idxs:
            fp.seek(idx * PAIRED_RECORD_BYTES)
            rec = fp.read(PAIRED_RECORD_BYTES)
            first = rec[:LEGACY_RECORD_BYTES]
            second = rec[LEGACY_RECORD_BYTES:]

            fc.seek(idx * LEGACY_RECORD_BYTES)
            c = fc.read(LEGACY_RECORD_BYTES)

            if second != c:
                mismatches += 1
            if first == second:
                same_first_second += 1

    method = "paired後半40Bと cache の同位置40Bをランダムサンプル比較。"
    result = (
        f"sample_n={len(idxs)}, mismatches(second_vs_cache)={mismatches}, "
        f"same(first_vs_second)={same_first_second}"
    )
    verdict = "not_supported" if mismatches == 0 else "supported"
    return HypothesisResult(
        title="H4: pairedデータ境界で DNN/NNUE がずれている",
        method=method,
        result=result,
        verdict=verdict,
    )


def render_report(output_path: str, command: str, event_file: str, results: Iterable[HypothesisResult]):
    lines = []
    lines.append("# val_loss 上昇 仮説検証レポート")
    lines.append("")
    lines.append("## 実行コマンド")
    lines.append("")
    lines.append("```bash")
    lines.append(command)
    lines.append("```")
    lines.append("")
    lines.append(f"- TensorBoard event: `{event_file}`")
    lines.append("")
    lines.append("## 結果")
    lines.append("")
    for r in results:
        lines.append(f"### {r.title}")
        lines.append(f"- 検証方法: {r.method}")
        lines.append(f"- 結果: {r.result}")
        lines.append(f"- 判定: `{r.verdict}`")
        lines.append("")

    lines.append("## 判定キー")
    lines.append("")
    lines.append("- `supported`: 仮説を支持する結果")
    lines.append("- `not_supported`: 仮説を支持しない結果")
    lines.append("- `partially_supported`: 一部のみ支持")
    lines.append("- `inconclusive`: データ不足で判断不能")
    lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Validate val_loss hypotheses")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--val-bin", required=True)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--paired-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-val-positions", type=int, default=100000)
    parser.add_argument("--sample-n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ea, event_file = _load_events(args.logdir)

    results = [
        validate_hypothesis_1_val_window_shift(
            val_bin=args.val_bin,
            paired=args.paired,
            batch_size=args.batch_size,
            max_val_positions=args.max_val_positions,
        ),
        validate_hypothesis_2_logging_granularity(ea),
        validate_hypothesis_3_overfitting_signal(ea),
        validate_hypothesis_4_paired_misalignment(
            val_bin=args.val_bin,
            paired_cache_dir=args.paired_cache_dir,
            sample_n=args.sample_n,
            seed=args.seed,
        ),
    ]

    cmd = (
        "PYTHONPATH=src python -m train_nnue.validate_val_loss_hypotheses "
        f"--logdir {args.logdir} --train-bin {args.train_bin} --val-bin {args.val_bin} "
        f"{'--paired ' if args.paired else ''}"
        f"--paired-cache-dir {args.paired_cache_dir if args.paired_cache_dir else ''} "
        f"--batch-size {args.batch_size} --max-val-positions {args.max_val_positions} "
        f"--sample-n {args.sample_n} --seed {args.seed} --output {args.output}"
    ).strip()

    render_report(args.output, cmd, event_file, results)

    print(f"Wrote report: {args.output}")
    for r in results:
        print(f"[{r.verdict}] {r.title}")
        print(f"  {r.result}")


if __name__ == "__main__":
    main()
