"""Expert Blending モデルのエキスパート可視化スクリプト。

各エキスパートが担当する局面の手数分布と、代表局面 (argmax_i p(i, j) の上位)
を画像化して 1 枚の HTML にまとめる。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.visualize_experts \\
        --checkpoint ../logs/expert_blending_8experts_v4_paired_uniform50_noise0_lambda05/checkpoints/180.ckpt \\
        --val ../dataset/split_v1_paired_uniform_50/val1 \\
        --backbone-weights ../tmp/dlshogi-model/model_resnet10_swish-072 \\
        --nnue-checkpoint logs/halfkp_v1/checkpoints/83000.ckpt \\
        --output-dir ../results/visualize_experts_8experts_lambda05_180
"""

import argparse
import html
import mmap
import os

import cshogi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import features as nnue_features
from train_nnue.expert_blending_dataset import (
    LEGACY_RECORD_BYTES,
    PSFEN_BYTES,
    ExpertBlendingDataset,
)
from train_nnue.expert_blending_model import (
    create_expert_blending_model,
    detect_blend_mode_from_state_dict,
)
from train_nnue.train_expert_blending import ExpertBlendingLightningModule


def collect_data(lit_module, val_dataset, dnn_bin_path, max_positions, device):
    """Validation データから gate weights / ply / PackedSfen を収集する。

    ExpertBlendingDataset (shuffle=False, backbone_type='dnn') は
    dnn.bin / nnue.bin を先頭から逐次読みする。dnn.bin の record 順は
    バッチ生成順と一致するため、本関数も同じ順序で PackedSfen バイト列を
    取り出して整合させる。

    Returns:
        weights: (N, n_experts) numpy array (softmax 後)
        plies: (N,) int32 numpy array (m(i))
        scores: (N,) float32 numpy array (教師評価値, NNUE と同じ単位)
        psfens: list[bytes] (32B each, dnn.bin 側の PackedSfen)
    """
    lit_module.eval()
    lit_module.to(device)

    weights_chunks = []
    plies_chunks = []
    scores_chunks = []
    psfens = []

    f = open(dnn_bin_path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    num_records = len(mm) // LEGACY_RECORD_BYTES
    record_pos = 0
    total = 0

    try:
        with torch.no_grad():
            for batch in val_dataset:
                x1, x2, us, them, white, black, outcome, score, ply = batch
                x1 = x1.to(device)
                x2 = x2.to(device)

                feat = lit_module.model.backbone(x1, x2)
                gate_weights = lit_module.model.adapter(feat, training=False)

                bs = x1.shape[0]
                weights_chunks.append(gate_weights.detach().cpu().numpy())
                plies_chunks.append(ply.detach().cpu().numpy().astype(np.int32).reshape(-1))
                scores_chunks.append(score.detach().cpu().numpy().astype(np.float32).reshape(-1))

                for _ in range(bs):
                    if record_pos >= num_records:
                        record_pos = 0
                    offset = record_pos * LEGACY_RECORD_BYTES
                    psfens.append(bytes(mm[offset : offset + PSFEN_BYTES]))
                    record_pos += 1

                total += bs
                if total >= max_positions:
                    break
    finally:
        mm.close()
        f.close()

    weights = np.concatenate(weights_chunks, axis=0)[:max_positions]
    plies = np.concatenate(plies_chunks, axis=0)[:max_positions]
    scores = np.concatenate(scores_chunks, axis=0)[:max_positions]
    psfens = psfens[:max_positions]
    return weights, plies, scores, psfens


def render_position_svg(psfen_bytes):
    """PackedSfen バイト列を cshogi.Board.to_svg() で SVG 化する。"""
    board = cshogi.Board()
    psfen_arr = np.frombuffer(psfen_bytes, dtype=np.uint8).copy()
    board.set_psfen(psfen_arr)
    return str(board.to_svg(scale=0.7))


def write_histogram(plies_for_expert, expert_id, total, max_ply, out_path):
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    bins = np.arange(0, max_ply + 6, 4)
    ax.hist(plies_for_expert, bins=bins, color="steelblue", alpha=0.85)
    ax.set_xlabel("Ply (move number)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Expert {expert_id}: ply distribution (n={len(plies_for_expert)} / {total})"
    )
    ax.set_xlim(0, max_ply + 4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_score_histogram(scores_for_expert, expert_id, total, score_bins, out_path):
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.hist(scores_for_expert, bins=score_bins, color="darkorange", alpha=0.85)
    ax.set_xlabel("Teacher score")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Expert {expert_id}: score distribution (n={len(scores_for_expert)} / {total})"
    )
    ax.set_xlim(score_bins[0], score_bins[-1])
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def select_representative_indices(col, dominant, expert_id, top_k):
    """Expert j の代表局面 index を「担当局面の p 中央値近傍」から選ぶ。

    1. argmax_{j'} p(i, j') = j を満たす i に絞る (dominant filter)
    2. p(i, j) の昇順で並べ、中央付近の top_k 個を選ぶ

    こうすることで、gate が極端に反応する adversarial な局面ではなく、
    担当エキスパートが「ふつうに選んでいる」局面を抽出できる。
    """
    indices = np.where(dominant == expert_id)[0]
    if len(indices) == 0:
        return np.array([], dtype=int)
    sorted_idx = indices[np.argsort(col[indices])]
    n = len(sorted_idx)
    if n <= top_k:
        return sorted_idx
    center = n // 2
    start = max(0, center - top_k // 2)
    end = start + top_k
    if end > n:
        end = n
        start = end - top_k
    return sorted_idx[start:end]


def write_html(output_dir, html_path, n_experts, dominant, weights, plies, scores, psfens, top_k, meta):
    rows = []
    for j in range(n_experts):
        col = weights[:, j]
        chosen = select_representative_indices(col, dominant, j, top_k)
        cells = []
        for idx in chosen:
            svg = render_position_svg(psfens[int(idx)])
            svg_name = f"expert_{j}_top_{len(cells)}.svg"
            with open(os.path.join(output_dir, svg_name), "w", encoding="utf-8") as fp:
                fp.write(svg)
            cells.append(
                (
                    svg_name,
                    float(col[int(idx)]),
                    int(plies[int(idx)]),
                    float(scores[int(idx)]),
                )
            )
        rows.append((j, int((dominant == j).sum()), cells))

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
        f.write("<title>Expert visualization</title>\n")
        f.write(
            "<style>\n"
            "body{font-family:sans-serif;margin:16px}\n"
            "table{border-collapse:collapse}\n"
            "td,th{border:1px solid #888;padding:6px;vertical-align:top}\n"
            ".pos{display:inline-block;text-align:center;margin:2px;font-size:11px}\n"
            ".pos object{display:block;width:170px;height:170px}\n"
            "img.hist{width:520px}\n"
            "</style>\n"
        )
        f.write("</head><body>\n")
        f.write("<h1>Expert visualization</h1>\n")
        f.write("<ul>\n")
        for k, v in meta.items():
            f.write(f"<li>{html.escape(k)}: {html.escape(str(v))}</li>\n")
        f.write("</ul>\n")
        f.write("<table>\n")
        f.write(
            "<tr><th>Expert</th><th>Ply histogram</th>"
            "<th>Score histogram</th>"
            f"<th>Representative positions (median-p of dominant, n={top_k})</th></tr>\n"
        )
        for j, n_assigned, cells in rows:
            f.write("<tr>\n")
            f.write(f"<td>Expert {j}<br>assigned n={n_assigned}</td>\n")
            f.write(
                f"<td><img class='hist' src='hist_expert_{j}.png'></td>\n"
            )
            f.write(
                f"<td><img class='hist' src='hist_score_expert_{j}.png'></td>\n"
            )
            f.write("<td>")
            for svg_name, p, ply, score in cells:
                f.write(
                    "<div class='pos'>"
                    f"<object data='{svg_name}' type='image/svg+xml'></object>"
                    f"<div>p={p:.3f}<br>ply={ply}<br>score={score:.0f}</div>"
                    "</div>\n"
                )
            f.write("</td>\n</tr>\n")
        f.write("</table>\n")
        f.write("</body></html>\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize expert specialization")
    parser.add_argument("--checkpoint", required=True, help="Expert Blending .ckpt path")
    parser.add_argument("--val", required=True, help="Validation directory (dnn.bin + nnue.bin)")
    parser.add_argument("--backbone-weights", required=True, help="dlshogi .npz weights path")
    parser.add_argument("--nnue-checkpoint", required=True, help="NNUE .ckpt for model structure")
    parser.add_argument("--feature-set", default="HalfKP")
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--adapter-hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-positions", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for path in [args.checkpoint, args.backbone_weights, args.nnue_checkpoint]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    if not os.path.isdir(args.val):
        raise FileNotFoundError(args.val)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feature_set = nnue_features.get_feature_set_from_name(args.feature_set)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    blend_mode = detect_blend_mode_from_state_dict(ckpt["state_dict"])
    print(f"blend_mode={blend_mode}")

    print("Building model ...")
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
    lit_module.load_state_dict(ckpt["state_dict"])

    print(f"Loading validation data: {args.val}")
    val_dataset = ExpertBlendingDataset(
        args.val,
        args.feature_set,
        args.batch_size,
        device=device,
        shuffle=False,
        backbone_type="dnn",
    )

    print(f"Collecting up to {args.max_positions} positions ...")
    weights, plies, scores, psfens = collect_data(
        lit_module,
        val_dataset,
        val_dataset.dnn_bin_path,
        max_positions=args.max_positions,
        device=device,
    )
    n = len(weights)
    n_experts = weights.shape[1]
    print(f"Collected {n} positions, {n_experts} experts")

    dominant = weights.argmax(axis=1)
    max_ply = int(plies.max()) if len(plies) else 256

    score_clip = float(np.percentile(np.abs(scores), 99)) if len(scores) else 1000.0
    score_clip = max(score_clip, 1.0)
    score_bins = np.linspace(-score_clip, score_clip, 51)

    print("Drawing per-expert ply / score histograms ...")
    for j in range(n_experts):
        mask = dominant == j
        write_histogram(
            plies_for_expert=plies[mask],
            expert_id=j,
            total=n,
            max_ply=max_ply,
            out_path=os.path.join(args.output_dir, f"hist_expert_{j}.png"),
        )
        write_score_histogram(
            scores_for_expert=np.clip(scores[mask], score_bins[0], score_bins[-1]),
            expert_id=j,
            total=n,
            score_bins=score_bins,
            out_path=os.path.join(args.output_dir, f"hist_score_expert_{j}.png"),
        )

    print("Rendering representative positions and HTML ...")
    html_path = os.path.join(args.output_dir, "index.html")
    meta = {
        "checkpoint": args.checkpoint,
        "validation_dir": args.val,
        "feature_set": args.feature_set,
        "n_experts": n_experts,
        "positions_used": n,
        "top_k": args.top_k,
        "blend_mode": blend_mode,
    }
    write_html(
        output_dir=args.output_dir,
        html_path=html_path,
        n_experts=n_experts,
        dominant=dominant,
        weights=weights,
        plies=plies,
        scores=scores,
        psfens=psfens,
        top_k=args.top_k,
        meta=meta,
    )
    print(f"Wrote: {html_path}")

    print("\n=== Per-expert summary ===")
    counts = np.bincount(dominant, minlength=n_experts)
    for j in range(n_experts):
        col = weights[:, j]
        mask = dominant == j
        ply_assigned = plies[mask]
        score_assigned = scores[mask]
        ply_mean = float(ply_assigned.mean()) if len(ply_assigned) else float("nan")
        score_mean = float(score_assigned.mean()) if len(score_assigned) else float("nan")
        score_std = float(score_assigned.std()) if len(score_assigned) else float("nan")
        print(
            f"  Expert {j}: assigned={counts[j]:5d}  "
            f"mean_p={col.mean():.4f}  max_p={col.max():.4f}  "
            f"mean_ply={ply_mean:.1f}  "
            f"mean_score={score_mean:.0f}  std_score={score_std:.0f}"
        )


if __name__ == "__main__":
    main()
