"""
Expert Blending モデルの expert 重み分布を分析・可視化するスクリプト。

学習済みチェックポイントを読み込み、validation データのサンプル局面に対して
expert 重みの分布、平均値、エントロピーを可視化する。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.analyze_expert_weights \
        --checkpoint logs/expert_blending_v1/lightning_logs/version_0/checkpoints/epoch=X.ckpt \
        --val ../dataset_qsearch_split/val.bin \
        --backbone-weights ../tmp/dlshogi-model/model_resnet10_swish-072 \
        --nnue-checkpoint logs/halfkp_v1/checkpoints/83000.ckpt \
        --output results/expert_weights_analysis.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import features as nnue_features
from train_nnue.expert_blending_dataset import ExpertBlendingDataset
from train_nnue.expert_blending_model import (
    create_expert_blending_model,
    detect_blend_mode_from_state_dict,
)
from train_nnue.train_expert_blending import ExpertBlendingLightningModule


def collect_expert_weights(lit_module, val_dataset, max_positions=10000, device="cpu"):
    """Validation データに対して expert 重みを収集する。

    Returns:
        gate_weights_all: (N, n_experts) numpy array
    """
    lit_module.eval()
    lit_module.to(device)

    all_weights = []
    total = 0

    with torch.no_grad():
        for batch in val_dataset:
            x1, x2, us, them, white, black, outcome, score, ply = batch
            x1, x2 = x1.to(device), x2.to(device)

            feat = lit_module.model.backbone(x1, x2)
            gate_weights = lit_module.model.adapter(feat, training=False)
            all_weights.append(gate_weights.cpu().numpy())

            total += x1.shape[0]
            if total >= max_positions:
                break

    return np.concatenate(all_weights, axis=0)[:max_positions]


def plot_expert_analysis(gate_weights, output_path):
    """Expert 重み分布の分析結果を画像として保存する。

    Args:
        gate_weights: (N, n_experts) numpy array
        output_path: 出力画像パス
    """
    n_experts = gate_weights.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Expert Blending Weight Analysis", fontsize=14)

    # (1) 各 expert の平均重みの棒グラフ
    ax = axes[0, 0]
    mean_weights = gate_weights.mean(axis=0)
    std_weights = gate_weights.std(axis=0)
    expert_labels = [f"Expert {i}" for i in range(n_experts)]
    bars = ax.bar(expert_labels, mean_weights, yerr=std_weights, capsize=5, alpha=0.7)
    ax.set_ylabel("Mean Weight")
    ax.set_title("Mean Expert Weights (+/- std)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, mean_weights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
        )

    # (2) Expert 重みのヒストグラム（各 expert ごと）
    ax = axes[0, 1]
    for i in range(n_experts):
        ax.hist(
            gate_weights[:, i],
            bins=50,
            alpha=0.5,
            label=f"Expert {i}",
            range=(0, 1),
        )
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title("Expert Weight Distribution")
    ax.legend()

    # (3) エントロピー分布
    ax = axes[1, 0]
    epsilon = 1e-12
    entropy = -(gate_weights * np.log(gate_weights + epsilon)).sum(axis=1)
    max_entropy = np.log(n_experts)
    ax.hist(entropy, bins=50, alpha=0.7, color="steelblue")
    ax.axvline(max_entropy, color="red", linestyle="--", label=f"Max entropy (ln {n_experts}={max_entropy:.2f})")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Count")
    ax.set_title(f"Gate Entropy Distribution (mean={entropy.mean():.3f})")
    ax.legend()

    # (4) Expert 支配度（各局面で最大重みの expert のヒストグラム）
    ax = axes[1, 1]
    dominant_expert = gate_weights.argmax(axis=1)
    counts = np.bincount(dominant_expert, minlength=n_experts)
    ax.bar(expert_labels, counts, alpha=0.7, color="coral")
    ax.set_ylabel("Count (dominant)")
    ax.set_title("Dominant Expert per Position")
    for i, c in enumerate(counts):
        ax.text(i, c + len(gate_weights) * 0.01, str(c), ha="center", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Analysis saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze Expert Blending weights")
    parser.add_argument("--checkpoint", required=True, help="Expert Blending .ckpt path")
    parser.add_argument("--val", required=True, help="Validation data (.bin)")
    parser.add_argument("--backbone-weights", required=True, help="dlshogi .npz weights path")
    parser.add_argument("--nnue-checkpoint", required=True, help="NNUE .ckpt for model structure")
    parser.add_argument("--feature-set", default="HalfKP")
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--adapter-hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-positions", type=int, default=10000)
    parser.add_argument("--output", default="results/expert_weights_analysis.png")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for path in [args.checkpoint, args.val, args.backbone_weights, args.nnue_checkpoint]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    feature_set = nnue_features.get_feature_set_from_name(args.feature_set)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    blend_mode = detect_blend_mode_from_state_dict(ckpt["state_dict"])

    # Build model structure
    print("Building model...")
    model = create_expert_blending_model(
        backbone_weights_path=args.backbone_weights,
        nnue_ckpt_path=args.nnue_checkpoint,
        feature_set=feature_set,
        n_experts=args.n_experts,
        adapter_hidden=args.adapter_hidden,
        blend_mode=blend_mode,
        device="cpu",
    )

    lit_module = ExpertBlendingLightningModule(model=model)

    # Load trained weights from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    lit_module.load_state_dict(ckpt["state_dict"])

    # Load validation data
    print(f"Loading validation data: {args.val}")
    val_dataset = ExpertBlendingDataset(
        args.val, args.feature_set, args.batch_size, device=device, shuffle=False,
    )

    # Collect expert weights
    print(f"Collecting expert weights (max {args.max_positions} positions)...")
    gate_weights = collect_expert_weights(
        lit_module, val_dataset, max_positions=args.max_positions, device=device,
    )
    print(f"Collected {len(gate_weights)} positions")

    # Plot analysis
    plot_expert_analysis(gate_weights, args.output)

    # Print summary statistics
    print(f"\n=== Expert Weight Summary ===")
    for i in range(args.n_experts):
        w = gate_weights[:, i]
        print(f"  Expert {i}: mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")
    epsilon = 1e-12
    entropy = -(gate_weights * np.log(gate_weights + epsilon)).sum(axis=1)
    max_entropy = np.log(args.n_experts)
    print(f"  Entropy: mean={entropy.mean():.4f}, std={entropy.std():.4f} (max={max_entropy:.4f})")


if __name__ == "__main__":
    main()
