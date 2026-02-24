"""
game_plyの差(nnue_ply - dnn_ply)によるvalidation lossの違いを検証するスクリプト。

Expert Blendingモデルとベースライン単一NNUEのper-record lossを
delta = nnue_ply - dnn_ply でbin分割して比較プロットする。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.check_loss_per_gameply \
        --expert-blending-checkpoint <path_to_160.ckpt> \
        --nnue-checkpoint logs/halfkp_v1/checkpoints/83000.ckpt \
        --val-dir ../dataset/split_v1_paired/val1 \
        --feature-set HalfKP \
        --output loss_per_gameply_delta.png
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
from train_nnue.expert_blending_dataset import ExpertBlendingDataset
from train_nnue.expert_blending_model import ExpertBlendingModel

RECORD_BYTES = 40
GAME_PLY_OFFSET = 36


def log(msg):
    print(msg, flush=True)


# --- game_ply extraction ---

def extract_game_plys(val_dir, max_records=None):
    """split dir の dnn.bin / nnue.bin から game_ply を抽出する。"""
    dnn_path = os.path.join(val_dir, "dnn.bin")
    nnue_path = os.path.join(val_dir, "nnue.bin")
    dnn_size = os.path.getsize(dnn_path)
    nnue_size = os.path.getsize(nnue_path)
    if dnn_size != nnue_size:
        raise ValueError(f"size mismatch: {dnn_path}={dnn_size}, {nnue_path}={nnue_size}")
    num_records = dnn_size // RECORD_BYTES
    if max_records is not None:
        num_records = min(num_records, max_records)

    dnn_plys = np.empty(num_records, dtype=np.uint16)
    nnue_plys = np.empty(num_records, dtype=np.uint16)

    with open(dnn_path, "rb") as fd, open(nnue_path, "rb") as fn:
        md = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
        mn = mmap.mmap(fn.fileno(), 0, access=mmap.ACCESS_READ)
        for i in range(num_records):
            base = i * RECORD_BYTES
            dnn_plys[i] = struct.unpack_from("<H", md, base + GAME_PLY_OFFSET)[0]
            nnue_plys[i] = struct.unpack_from("<H", mn, base + GAME_PLY_OFFSET)[0]
        md.close()
        mn.close()

    return dnn_plys, nnue_plys


# --- per-record loss computation ---

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


def _detect_backbone_type(state_dict):
    """state_dictからbackbone_typeを自動判定する。"""
    if "model.adapter.fc1.weight" in state_dict:
        return "dnn"
    if "model.backbone.input.weight" in state_dict:
        return "nnue"
    raise ValueError("Cannot detect backbone type from checkpoint state_dict")


def load_expert_blending_model(checkpoint_path, device):
    """Expert Blendingチェックポイントからモデルを復元する。DNN/NNUEバックボーン自動判定。"""
    from train_nnue.expert_blending_model import DNNAdapter, DNNBackbone, NNUEBackbone, NNUEExperts
    from dlshogi.network.policy_value_network_resnet10_swish import (
        PolicyValueNetwork as DlshogiPVNet,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    backbone_type = _detect_backbone_type(state_dict)
    n_experts = state_dict["model.nnue_experts.input_weight"].shape[0]
    num_features = state_dict["model.nnue_experts.input_weight"].shape[2]

    if backbone_type == "dnn":
        adapter_hidden = state_dict["model.adapter.fc1.weight"].shape[0]
        pv_net = DlshogiPVNet()
        backbone = DNNBackbone(pv_net)
        adapter = DNNAdapter(192, hidden_dim=adapter_hidden, n_experts=n_experts)
    else:
        bb_num_features = state_dict["model.backbone.input.weight"].shape[1]
        backbone = NNUEBackbone(bb_num_features, n_experts=n_experts)
        adapter = None

    nnue_experts = NNUEExperts(n_experts, num_features)
    model = ExpertBlendingModel(backbone, adapter, nnue_experts, backbone_type=backbone_type)

    model_state = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, backbone_type


def load_baseline_nnue(nnue_ckpt_path, feature_set_name, device):
    """ベースラインNNUEモデルをロードする。"""
    from model import NNUE

    feature_set = nnue_features.get_feature_set_from_name(feature_set_name)
    ckpt = torch.load(nnue_ckpt_path, map_location="cpu", weights_only=False)
    nnue = NNUE(feature_set=feature_set)
    nnue.load_state_dict(ckpt["state_dict"])
    nnue.to(device)
    nnue.eval()
    return nnue


# --- bin集計 & プロット ---

def _bin_and_aggregate(keys, eb_losses, bl_losses, bin_width, min_count=10):
    """keysでbin分割し、各binの平均lossとカウントを返す。"""
    n = min(len(keys), len(eb_losses), len(bl_losses))
    keys = keys[:n]
    eb_losses = eb_losses[:n]
    bl_losses = bl_losses[:n]

    lo, hi = keys.min(), keys.max()
    bins = np.arange(lo, hi + bin_width + 1, bin_width)
    bin_indices = np.digitize(keys, bins) - 1

    bin_centers, eb_means, bl_means, counts = [], [], [], []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        cnt = mask.sum()
        if cnt < min_count:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
        eb_means.append(eb_losses[mask].mean())
        bl_means.append(bl_losses[mask].mean())
        counts.append(cnt)

    return np.array(bin_centers), np.array(eb_means), np.array(bl_means), np.array(counts)


def _plot_loss_chart(bin_centers, eb_means, bl_means, counts, xlabel, title, output_path, bin_width=1):
    """2段プロット（上: loss曲線、下: ヒストグラム）を保存する。"""
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(bin_centers, eb_means, "o-", label="Expert Blending", markersize=3)
    ax1.plot(bin_centers, bl_means, "s-", label="Baseline NNUE", markersize=3)
    ax1.set_ylabel("Mean Loss")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(bin_centers, counts, width=bin_width * 0.8, alpha=0.5, color="gray")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log(f"Plot saved: {output_path}")


def print_table(header, bin_centers, eb_means, bl_means, counts):
    log(f"\n{header:>8} {'EB_loss':>10} {'BL_loss':>10} {'count':>8}")
    log("-" * 40)
    for c, e, b, ct in zip(bin_centers, eb_means, bl_means, counts):
        log(f"{c:8.1f} {e:10.6f} {b:10.6f} {ct:8d}")


def main():
    parser = argparse.ArgumentParser(
        description="Check validation loss per game_ply delta"
    )
    parser.add_argument(
        "--expert-blending-checkpoint", required=True,
        help="Expert Blending model .ckpt path",
    )
    parser.add_argument(
        "--nnue-checkpoint", required=True,
        help="Baseline NNUE .ckpt path",
    )
    parser.add_argument(
        "--val-dir",
        required=True,
        help="Validation split directory containing dnn.bin and nnue.bin",
    )
    parser.add_argument("--feature-set", default="HalfKP")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-positions", type=int, default=100000)
    parser.add_argument("--output", default="loss_per_gameply_delta.png")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_")
    parser.add_argument("--label-smoothing-eps", type=float, default=0.001)
    parser.add_argument("--score-scaling", type=float, default=361)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    for path in [args.expert_blending_checkpoint, args.nnue_checkpoint, args.val_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log(f"Device: {device}")

    # 1. game_ply抽出
    log("Extracting game_plys from split directory...")
    dnn_plys, nnue_plys = extract_game_plys(args.val_dir, max_records=args.max_positions)
    deltas = nnue_plys.astype(np.int32) - dnn_plys.astype(np.int32)
    log(f"  Records: {len(deltas)}, delta range: [{deltas.min()}, {deltas.max()}]")
    log(f"  nnue_ply range: [{nnue_plys.min()}, {nnue_plys.max()}]")

    # 2. モデルロード
    log("Loading Expert Blending model...")
    eb_model, backbone_type = load_expert_blending_model(args.expert_blending_checkpoint, device)
    log(f"  backbone_type: {backbone_type}")
    log("Loading baseline NNUE model...")
    baseline_nnue = load_baseline_nnue(args.nnue_checkpoint, args.feature_set, device)

    # 3. 単一データセットで両モデルのper-record lossを同時計算
    #    SparseBatchProviderを1つだけ使用（複数Provider同時使用でのデッドロック回避）
    log("Creating dataset...")
    dataset = ExpertBlendingDataset(
        args.val_dir, args.feature_set, args.batch_size, device=device, shuffle=False,
        backbone_type=backbone_type,
    )

    nnue2score = 600
    scaling = args.score_scaling
    eb_all_losses = []
    bl_all_losses = []
    total_positions = 0

    log("Computing per-record losses (both models in single pass)...")
    with torch.no_grad():
        for batch in dataset:
            if backbone_type == "nnue":
                us_bb, them_bb, white_bb, black_bb, us, them, white, black, outcome, score, _ = batch
                q_eb = eb_model(us_bb, them_bb, white_bb, black_bb, us, them, white, black, training=False) * nnue2score / scaling
            else:
                x1, x2, us, them, white, black, outcome, score, _ = batch
                q_eb = eb_model(x1, x2, us, them, white, black, training=False) * nnue2score / scaling

            # Expert Blending
            eb_losses = compute_per_record_loss(
                q_eb, score, outcome, scaling, args.lambda_, args.label_smoothing_eps
            )
            eb_all_losses.append(eb_losses.cpu().numpy())

            # Baseline NNUE (同じNNUE features を使用)
            q_bl = baseline_nnue(us, them, white, black) * nnue2score / scaling
            bl_losses = compute_per_record_loss(
                q_bl, score, outcome, scaling, args.lambda_, args.label_smoothing_eps
            )
            bl_all_losses.append(bl_losses.cpu().numpy())

            total_positions += us.shape[0]
            if total_positions >= args.max_positions:
                break

    eb_all_losses = np.concatenate(eb_all_losses)[:args.max_positions]
    bl_all_losses = np.concatenate(bl_all_losses)[:args.max_positions]
    log(f"  Evaluated {len(eb_all_losses)} positions")

    # 4. bin集計 & プロット
    output_base, output_ext = os.path.splitext(args.output)

    # Plot 1: delta (nnue_ply - dnn_ply) 別
    bc, em, bm, ct = _bin_and_aggregate(deltas, eb_all_losses, bl_all_losses, bin_width=1)
    delta_path = f"{output_base}_delta{output_ext}"
    _plot_loss_chart(bc, em, bm, ct, "delta (nnue_ply - dnn_ply)",
                     "Validation Loss by game_ply delta (nnue_ply - dnn_ply)", delta_path)
    print_table("delta", bc, em, bm, ct)

    # Plot 2: nnue_ply 別 (bin_width=10 for smoother)
    bc2, em2, bm2, ct2 = _bin_and_aggregate(
        nnue_plys[:len(eb_all_losses)].astype(np.int32),
        eb_all_losses, bl_all_losses, bin_width=10,
    )
    ply_path = f"{output_base}_nnue_ply{output_ext}"
    _plot_loss_chart(bc2, em2, bm2, ct2, "nnue_ply",
                     "Validation Loss by nnue_ply", ply_path, bin_width=10)
    print_table("nnue_ply", bc2, em2, bm2, ct2)


if __name__ == "__main__":
    main()
