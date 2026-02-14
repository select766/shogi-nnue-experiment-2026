"""
ベースライン（単一NNUE）の validation loss を計算するスクリプト。

Expert Blending モデルと同じ損失関数・同じデータで
単一 NNUE の validation loss を計測し、比較用ベースラインとする。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.eval_baseline_loss \
        --nnue-checkpoint logs/halfkp_v1/checkpoints/83000.ckpt \
        --val ../dataset_qsearch_split/val.bin \
        --feature-set HalfKP
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

import features as nnue_features
import nnue_dataset


def compute_baseline_loss(
    nnue_ckpt_path,
    val_bin_path,
    feature_set_name="HalfKP",
    batch_size=256,
    max_positions=100000,
    lambda_=1.0,
    label_smoothing_eps=0.001,
    score_scaling=361,
    device="cpu",
):
    """単一 NNUE モデルの validation loss を計算する。

    Args:
        nnue_ckpt_path: NNUE チェックポイントパス
        val_bin_path: validation データ (.bin) パス
        feature_set_name: NNUE 特徴セット名
        batch_size: バッチサイズ
        max_positions: 最大評価局面数
        lambda_: teacher score の重み (1.0=teacher only)
        label_smoothing_eps: ラベルスムージング
        score_scaling: スコアスケーリング定数
        device: デバイス

    Returns:
        平均 validation loss (float)
    """
    # Load NNUE model
    from model import NNUE

    feature_set = nnue_features.get_feature_set_from_name(feature_set_name)
    ckpt = torch.load(nnue_ckpt_path, map_location="cpu", weights_only=False)

    nnue = NNUE(feature_set=feature_set)
    # state_dict のキーを調整（Lightning の prefix がある場合に対応）
    state_dict = ckpt["state_dict"]
    nnue.load_state_dict(state_dict)
    nnue.to(device)
    nnue.eval()

    # Load validation data using nnue_dataset (SparseBatchProvider)
    # cyclic=False で EOF まで読む
    val_dataset = nnue_dataset.SparseBatchDataset(
        feature_set_name, val_bin_path, batch_size, cyclic=False, device=device
    )

    nnue2score = 600
    scaling = score_scaling

    total_loss = 0.0
    total_batches = 0
    total_positions = 0

    print(f"Computing baseline validation loss...")
    print(f"  Checkpoint: {nnue_ckpt_path}")
    print(f"  Validation: {val_bin_path}")
    print(f"  lambda={lambda_}, score_scaling={score_scaling}, eps={label_smoothing_eps}")

    with torch.no_grad():
        for batch in val_dataset:
            us, them, white, black, outcome, score, ply = batch

            q = nnue(us, them, white, black) * nnue2score / scaling
            t = outcome * (1.0 - label_smoothing_eps * 2.0) + label_smoothing_eps
            p = (score / scaling).sigmoid()

            epsilon = 1e-12
            teacher_entropy = -(
                p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log()
            )
            outcome_entropy = -(
                t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log()
            )
            teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
            outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))

            result = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
            entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
            loss = result.mean() - entropy.mean()

            total_loss += loss.item()
            total_batches += 1
            total_positions += us.shape[0]

            if total_positions >= max_positions:
                break

    avg_loss = total_loss / total_batches
    print(f"\n=== Baseline NNUE Validation Loss ===")
    print(f"  Positions evaluated: {total_positions}")
    print(f"  Batches: {total_batches}")
    print(f"  Average loss: {avg_loss:.6f}")

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline NNUE validation loss")
    parser.add_argument(
        "--nnue-checkpoint", required=True, help="NNUE .ckpt path"
    )
    parser.add_argument("--val", required=True, help="Validation data (.bin)")
    parser.add_argument("--feature-set", default="HalfKP", help="NNUE feature set name")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-positions", type=int, default=100000)
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_")
    parser.add_argument("--label-smoothing-eps", type=float, default=0.001)
    parser.add_argument("--score-scaling", type=float, default=361)
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    args = parser.parse_args()

    for path in [args.nnue_checkpoint, args.val]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    compute_baseline_loss(
        nnue_ckpt_path=args.nnue_checkpoint,
        val_bin_path=args.val,
        feature_set_name=args.feature_set,
        batch_size=args.batch_size,
        max_positions=args.max_positions,
        lambda_=args.lambda_,
        label_smoothing_eps=args.label_smoothing_eps,
        score_scaling=args.score_scaling,
        device=device,
    )


if __name__ == "__main__":
    main()
