"""
各エキスパートを独立NNUE評価関数としてnnue_ply別のlossを計算し、
ベースラインNNUEと合わせてプロットするスクリプト。

Expert Blendingモデルの各エキスパートが、どのply帯で強い/弱いかを可視化する。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -u -m train_nnue.check_loss_per_expert \
        --expert-blending-checkpoint <path_to_160.ckpt> \
        --nnue-checkpoint <path_to_83000.ckpt> \
        --val <paired_bin> \
        --feature-set HalfKP \
        --max-positions 1000000 \
        --output loss_per_expert.png
"""

import argparse
import mmap
import os
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import features as nnue_features
import nnue_dataset
from train_nnue.expert_blending_dataset import (
    PAIRED_RECORD_BYTES,
    extract_paired_nnue_bin,
)


def log(msg):
    print(msg, flush=True)


def extract_nnue_plys(paired_bin_path, max_records=None):
    """paired .bin (80B/record) から NNUE側の game_ply を抽出する。"""
    file_size = os.path.getsize(paired_bin_path)
    num_records = file_size // PAIRED_RECORD_BYTES
    if max_records is not None:
        num_records = min(num_records, max_records)

    nnue_plys = np.empty(num_records, dtype=np.uint16)
    with open(paired_bin_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for i in range(num_records):
            base = i * PAIRED_RECORD_BYTES
            nnue_plys[i] = struct.unpack_from("<H", mm, base + 76)[0]
        mm.close()
    return nnue_plys


def compute_per_record_loss(q, score, outcome, scaling, lambda_, label_smoothing_eps):
    """per-record loss (mean前) を返す。shape: (batch,)"""
    t = outcome * (1.0 - label_smoothing_eps * 2.0) + label_smoothing_eps
    p = (score / scaling).sigmoid()

    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))

    result = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    return (result - entropy).squeeze(-1)


def load_expert_as_nnue(expert_state, feature_set, device):
    """1つのエキスパート重みからNNUEモデルを構築する。

    Args:
        expert_state: dict with keys input.weight, input.bias, l1.weight, ...
        feature_set: FeatureBlock
        device: device string
    """
    from model import NNUE

    nnue = NNUE(feature_set=feature_set)
    nnue.load_state_dict(expert_state)
    nnue.to(device)
    nnue.eval()
    return nnue


def extract_expert_states(checkpoint_path):
    """Expert Blendingチェックポイントから各エキスパートのNNUE state_dictを抽出する。

    Returns:
        list[dict]: 各エキスパートのstate_dict (NNUE形式のキー)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    n_experts = state_dict["model.nnue_experts.input_weight"].shape[0]
    log(f"  n_experts={n_experts}")

    # NNUEExperts param name -> NNUE model param name
    param_map = {
        "input_weight": "input.weight",
        "input_bias": "input.bias",
        "l1_weight": "l1.weight",
        "l1_bias": "l1.bias",
        "l2_weight": "l2.weight",
        "l2_bias": "l2.bias",
        "output_weight": "output.weight",
        "output_bias": "output.bias",
    }

    expert_states = []
    for k in range(n_experts):
        sd = {}
        for expert_key, nnue_key in param_map.items():
            full_key = f"model.nnue_experts.{expert_key}"
            sd[nnue_key] = state_dict[full_key][k]
        expert_states.append(sd)

    return expert_states


def main():
    parser = argparse.ArgumentParser(
        description="Check per-expert loss by nnue_ply"
    )
    parser.add_argument(
        "--expert-blending-checkpoint", required=True,
        help="Expert Blending model .ckpt path",
    )
    parser.add_argument(
        "--nnue-checkpoint", required=True,
        help="Baseline NNUE .ckpt path",
    )
    parser.add_argument("--val", required=True, help="Paired validation data (.bin, 80B/record)")
    parser.add_argument("--feature-set", default="HalfKP")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-positions", type=int, default=1000000)
    parser.add_argument("--output", default="loss_per_expert.png")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_")
    parser.add_argument("--label-smoothing-eps", type=float, default=0.001)
    parser.add_argument("--score-scaling", type=float, default=361)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    for path in [args.expert_blending_checkpoint, args.nnue_checkpoint, args.val]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log(f"Device: {device}")

    # 1. nnue_ply抽出
    log("Extracting nnue_plys from paired bin...")
    nnue_plys = extract_nnue_plys(args.val, max_records=args.max_positions)
    log(f"  Records: {len(nnue_plys)}, nnue_ply range: [{nnue_plys.min()}, {nnue_plys.max()}]")

    # 2. モデルロード
    feature_set = nnue_features.get_feature_set_from_name(args.feature_set)

    log("Loading expert weights from checkpoint...")
    expert_states = extract_expert_states(args.expert_blending_checkpoint)
    n_experts = len(expert_states)

    models = []
    for k in range(n_experts):
        log(f"  Loading expert {k}...")
        models.append(load_expert_as_nnue(expert_states[k], feature_set, device))

    log("Loading baseline NNUE model...")
    from model import NNUE
    bl_ckpt = torch.load(args.nnue_checkpoint, map_location="cpu", weights_only=False)
    baseline = NNUE(feature_set=feature_set)
    baseline.load_state_dict(bl_ckpt["state_dict"])
    baseline.to(device)
    baseline.eval()

    # 3. NNUE側binでデータ読み込み & 全モデルのlossを同時計算
    log("Extracting NNUE-side bin...")
    nnue_bin_path = extract_paired_nnue_bin(args.val)
    log(f"  {nnue_bin_path}")

    log("Creating dataset...")
    val_dataset = nnue_dataset.SparseBatchDataset(
        args.feature_set, nnue_bin_path, args.batch_size, cyclic=False, device=device
    )

    nnue2score = 600
    scaling = args.score_scaling
    # n_experts + 1 (baseline)
    all_losses = [[] for _ in range(n_experts + 1)]
    total_positions = 0

    log(f"Computing per-record losses ({n_experts} experts + baseline)...")
    with torch.no_grad():
        for batch in val_dataset:
            us, them, white, black, outcome, score, _ = batch
            bs = us.shape[0]

            for k in range(n_experts):
                q = models[k](us, them, white, black) * nnue2score / scaling
                losses = compute_per_record_loss(
                    q, score, outcome, scaling, args.lambda_, args.label_smoothing_eps
                )
                all_losses[k].append(losses.cpu().numpy())

            # Baseline
            q_bl = baseline(us, them, white, black) * nnue2score / scaling
            bl_losses = compute_per_record_loss(
                q_bl, score, outcome, scaling, args.lambda_, args.label_smoothing_eps
            )
            all_losses[n_experts].append(bl_losses.cpu().numpy())

            total_positions += bs
            if total_positions % 100000 < args.batch_size:
                log(f"  {total_positions} positions processed...")
            if total_positions >= args.max_positions:
                break

    max_pos = args.max_positions
    for i in range(n_experts + 1):
        all_losses[i] = np.concatenate(all_losses[i])[:max_pos]
    log(f"  Evaluated {len(all_losses[0])} positions")

    # 4. nnue_ply別にbin集計 & プロット
    n = min(len(nnue_plys), len(all_losses[0]))
    nnue_plys_trimmed = nnue_plys[:n].astype(np.int32)

    bin_width = 10
    lo, hi = nnue_plys_trimmed.min(), nnue_plys_trimmed.max()
    bins = np.arange(lo, hi + bin_width + 1, bin_width)
    bin_indices = np.digitize(nnue_plys_trimmed, bins) - 1

    # Compute bin stats for each model
    bin_centers = []
    model_means = [[] for _ in range(n_experts + 1)]
    counts = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        cnt = mask.sum()
        if cnt < 10:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
        counts.append(cnt)
        for m in range(n_experts + 1):
            model_means[m].append(all_losses[m][mask].mean())

    bin_centers = np.array(bin_centers)
    counts = np.array(counts)
    for m in range(n_experts + 1):
        model_means[m] = np.array(model_means[m])

    # Plot
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for k in range(n_experts):
        ax1.plot(bin_centers, model_means[k], "-", label=f"Expert {k}",
                 color=colors[k], alpha=0.7, linewidth=1)
    ax1.plot(bin_centers, model_means[n_experts], "k-", label="Baseline NNUE",
             linewidth=2.5)

    ax1.set_ylabel("Mean Loss")
    ax1.set_title("Validation Loss by nnue_ply (per expert)")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    ax2.bar(bin_centers, counts, width=bin_width * 0.8, alpha=0.5, color="gray")
    ax2.set_xlabel("nnue_ply")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    log(f"Plot saved: {args.output}")

    # テキストテーブル
    header = f"{'nnue_ply':>8}"
    for k in range(n_experts):
        header += f" {'E'+str(k):>10}"
    header += f" {'Baseline':>10} {'count':>8}"
    log(f"\n{header}")
    log("-" * len(header))
    for i, c in enumerate(bin_centers):
        row = f"{c:8.0f}"
        for k in range(n_experts):
            row += f" {model_means[k][i]:10.6f}"
        row += f" {model_means[n_experts][i]:10.6f} {counts[i]:8d}"
        log(row)


if __name__ == "__main__":
    main()
