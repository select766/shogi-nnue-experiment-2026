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
import json
import math
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


# ── 将棋特徴量・Lift 分析 ─────────────────────────────────────

_ROOK_PIECES_BLACK = frozenset({cshogi.BROOK, cshogi.BPROM_ROOK})
_ROOK_PIECES_WHITE = frozenset({cshogi.WROOK, cshogi.WPROM_ROOK})

# 特徴名 → 表示ラベル
_FEATURE_LABELS = {
    "b_ibisha":       "先手居飛車",
    "b_furibisha":    "先手振り飛車",
    "b_chubi":        "先手中飛車",
    "w_ibisha":       "後手居飛車",
    "w_furibisha":    "後手振り飛車",
    "w_chubi":        "後手中飛車",
    "ai_furibisha":   "相振り飛車",
    "taiko_form":     "対抗形",
    "b_irikoma":      "先手入玉",
    "w_irikoma":      "後手入玉",
    "irikoma_any":    "入玉あり",
    "b_anaguma_hint": "先手穴熊的",
    "w_anaguma_hint": "後手穴熊的",
    "score_winning":  "勝勢局面(|s|>2000)",
    "score_even":     "互角局面(|s|≤500)",
    "early_game":     "序盤(≤40手)",
    "mid_game":       "中盤(41-100手)",
    "late_game":      "終盤(>100手)",
}

# ラベル自動生成の優先順位（意味的に上位の特徴から並べる）
_LABEL_PRIORITY = [
    "irikoma_any",
    "ai_furibisha",
    "taiko_form",
    "b_furibisha",
    "w_furibisha",
    "b_chubi",
    "w_chubi",
    "b_anaguma_hint",
    "w_anaguma_hint",
    "early_game",
    "late_game",
    "score_winning",
    "score_even",
]

_LIFT_LABEL_THRESHOLD = 1.5  # CI 下限がこの値を超えたら有意と判定


def _rook_file_shogi(board, color):
    """指定色の飛車/竜の将棋の筋番号 (1–9) を返す。盤上になければ None。

    cshogi の make_file(sq) は 0-indexed で 0=1筋, 8=9筋。
    先手(BLACK)の居飛車は 1–4 筋、振り飛車は 6–9 筋、中飛車は 5 筋。
    後手(WHITE)は左右が逆になる（居飛車 6–9 筋、振り飛車 1–4 筋）。
    """
    targets = _ROOK_PIECES_BLACK if color == cshogi.BLACK else _ROOK_PIECES_WHITE
    for sq in range(81):
        if board.piece(sq) in targets:
            return cshogi.make_file(sq) + 1  # 1–9
    return None


def extract_shogi_features(psfen_bytes):
    """PackedSfen バイト列から将棋特徴量 (bool の dict) を抽出する。"""
    board = cshogi.Board()
    board.set_psfen(np.frombuffer(psfen_bytes, dtype=np.uint8).copy())

    bf = _rook_file_shogi(board, cshogi.BLACK)
    wf = _rook_file_shogi(board, cshogi.WHITE)

    b_ibisha = bf is not None and bf <= 4
    b_chubi  = bf is not None and bf == 5
    b_furi   = bf is not None and bf >= 6
    w_ibisha = wf is not None and wf >= 6
    w_chubi  = wf is not None and wf == 5
    w_furi   = wf is not None and wf <= 4

    bk_sq   = board.king_square(cshogi.BLACK)
    wk_sq   = board.king_square(cshogi.WHITE)
    # make_rank: 0=1段目(後手陣), 8=9段目(先手陣)
    bk_rank = cshogi.make_rank(bk_sq)
    wk_rank = cshogi.make_rank(wk_sq)
    bk_file = cshogi.make_file(bk_sq) + 1  # 1–9
    wk_file = cshogi.make_file(wk_sq) + 1

    # 先手(BLACK)入玉: 玉が 1–3 段目 (rank 0–2)
    b_irikoma = bk_rank <= 2
    # 後手(WHITE)入玉: 玉が 7–9 段目 (rank 6–8)
    w_irikoma = wk_rank >= 6

    # 穴熊ヒント: 玉が端筋 (1–2 or 8–9 筋) かつ自陣の奥に引いている
    b_anaguma = (bk_file <= 2 or bk_file >= 8) and bk_rank >= 7
    w_anaguma = (wk_file <= 2 or wk_file >= 8) and wk_rank <= 1

    return {
        "b_ibisha":       bool(b_ibisha),
        "b_furibisha":    bool(b_furi),
        "b_chubi":        bool(b_chubi),
        "w_ibisha":       bool(w_ibisha),
        "w_furibisha":    bool(w_furi),
        "w_chubi":        bool(w_chubi),
        "ai_furibisha":   bool(b_furi and w_furi),
        "taiko_form":     bool((b_furi and w_ibisha) or (b_ibisha and w_furi)),
        "b_irikoma":      bool(b_irikoma),
        "w_irikoma":      bool(w_irikoma),
        "irikoma_any":    bool(b_irikoma or w_irikoma),
        "b_anaguma_hint": bool(b_anaguma),
        "w_anaguma_hint": bool(w_anaguma),
    }


def extract_all_features(psfens, plies, scores):
    """全局面から将棋特徴量を抽出し、feature_name -> bool 配列の dict を返す。"""
    n = len(psfens)
    feat = {name: np.zeros(n, dtype=bool) for name in _FEATURE_LABELS}

    for i, psfen in enumerate(psfens):
        for name, val in extract_shogi_features(psfen).items():
            feat[name][i] = val

    feat["score_winning"] = np.abs(scores) > 2000
    feat["score_even"]    = np.abs(scores) <= 500
    feat["early_game"]    = plies <= 40
    feat["mid_game"]      = (plies > 40) & (plies <= 100)
    feat["late_game"]     = plies > 100
    return feat


def _wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI (下限, 上限) を返す。"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_lift_table(features, dominant, n_experts):
    """各エキスパートの特徴量 lift を計算する。

    Returns:
        dict: expert_id -> [(fname, lift, lift_ci_lo, lift_ci_hi, rate_j, rate_all), ...]
              |log(lift)| の降順でソート済み
    """
    n = len(dominant)
    result = {}
    for j in range(n_experts):
        mask_j = dominant == j
        n_j = int(mask_j.sum())
        rows = []
        for fname, fvals in features.items():
            k_all = int(fvals.sum())
            rate_all = k_all / n if n > 0 else 0.0
            if rate_all == 0.0:
                continue
            k_j = int((fvals & mask_j).sum())
            rate_j = k_j / n_j if n_j > 0 else 0.0
            lift = rate_j / rate_all
            ci_lo_r, ci_hi_r = _wilson_ci(k_j, n_j)
            rows.append((
                fname, lift,
                ci_lo_r / rate_all,  # lift CI 下限
                ci_hi_r / rate_all,  # lift CI 上限
                rate_j, rate_all,
            ))
        rows.sort(key=lambda x: abs(math.log(max(x[1], 1e-9))), reverse=True)
        result[j] = rows
    return result


def auto_label_expert(lift_rows):
    """lift CI 下限が閾値を超える上位特徴でエキスパートのラベル文字列を生成する。"""
    lift_dict = {fname: (lift, lo) for fname, lift, lo, *_ in lift_rows}
    labels = []
    for fname in _LABEL_PRIORITY:
        if fname not in lift_dict:
            continue
        lift, lo = lift_dict[fname]
        if lo >= _LIFT_LABEL_THRESHOLD:
            display = _FEATURE_LABELS.get(fname, fname)
            labels.append(f"{display}({lift:.1f}×)")
        if len(labels) >= 2:
            break
    return " / ".join(labels) if labels else "—"


def _lift_details_html(lift_rows, top_n=8):
    """lift 詳細を HTML <details> 要素として返す。"""
    rows_html = []
    for fname, lift, lo, hi, rate_j, rate_all in lift_rows[:top_n]:
        label = html.escape(_FEATURE_LABELS.get(fname, fname))
        direction = "▲" if lift >= 1.0 else "▼"
        color = "#c00" if lift >= _LIFT_LABEL_THRESHOLD else ("#060" if lift <= 1 / _LIFT_LABEL_THRESHOLD else "#000")
        rows_html.append(
            f"<tr>"
            f"<td>{label}</td>"
            f"<td style='color:{color}'>{direction}{lift:.2f}</td>"
            f"<td>[{lo:.2f},{hi:.2f}]</td>"
            f"<td>{rate_j*100:.1f}%</td>"
            f"<td>{rate_all*100:.1f}%</td>"
            f"</tr>"
        )
    if not rows_html:
        return ""
    return (
        "<details style='margin-top:4px'>"
        "<summary style='cursor:pointer;font-size:11px'>Lift 詳細</summary>"
        "<table style='font-size:11px;border-collapse:collapse;margin-top:2px'>"
        "<tr style='background:#eee'><th>特徴</th><th>lift</th>"
        "<th>95%CI</th><th>expert率</th><th>全体率</th></tr>"
        + "".join(rows_html)
        + "</table></details>"
    )


def _sample_evenly(indices, n):
    """indices から n 個を均等サンプリングする。"""
    if len(indices) == 0:
        return np.array([], dtype=int)
    if len(indices) <= n:
        return indices
    step = len(indices) / n
    return indices[[int(i * step) for i in range(n)]]


def write_feature_check_html(output_dir, features, psfens, plies, scores, n_per_class=10):
    """各特徴量の True/False 局面サンプルを一覧表示する HTML を生成する。

    目視で特徴量抽出の正確さを確認するためのデバッグ用出力。
    SVG は feat_check_{連番}.svg として保存し、HTML から参照する。
    """
    html_path = os.path.join(output_dir, "feature_check.html")
    svg_counter = [0]

    def render_and_save(idx):
        svg = render_position_svg(psfens[int(idx)])
        svg_name = f"feat_check_{svg_counter[0]}.svg"
        svg_counter[0] += 1
        with open(os.path.join(output_dir, svg_name), "w", encoding="utf-8") as fp:
            fp.write(svg)
        return svg_name

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
        f.write("<title>Feature check</title>\n")
        f.write(
            "<style>\n"
            "body{font-family:sans-serif;margin:16px}\n"
            "h2{margin-top:28px;border-bottom:2px solid #888;padding-bottom:4px}\n"
            "table{border-collapse:collapse;width:100%}\n"
            "td,th{border:1px solid #888;padding:6px;vertical-align:top}\n"
            ".pos{display:inline-block;text-align:center;margin:3px;font-size:11px}\n"
            ".pos object{display:block;width:150px;height:150px}\n"
            ".true-col{background:#efffef}\n"
            ".false-col{background:#fff0f0}\n"
            ".rate{color:#555;font-size:13px;margin-left:8px}\n"
            "</style>\n"
        )
        f.write("</head><body>\n")
        f.write("<h1>特徴量検証 (Feature Check)</h1>\n")
        f.write(
            f"<p>各特徴が <b>True</b>（緑）/ <b>False</b>（赤）となる局面を"
            f"それぞれ最大 {n_per_class} 個、均等サンプリングして表示します。</p>\n"
        )

        for fname, feat_arr in features.items():
            label = _FEATURE_LABELS.get(fname, fname)
            true_idx = np.where(feat_arr)[0]
            false_idx = np.where(~feat_arr)[0]
            true_sample = _sample_evenly(true_idx, n_per_class)
            false_sample = _sample_evenly(false_idx, n_per_class)
            rate = float(feat_arr.mean()) * 100

            f.write(
                f"<h2>{html.escape(label)}"
                f"<span class='rate'>({html.escape(fname)}) "
                f"True: {len(true_idx)} 局面 / {rate:.1f}%</span></h2>\n"
            )
            f.write("<table><tr>\n")
            f.write(f"<th class='true-col'>True ({len(true_sample)} 局面を表示)</th>\n")
            f.write(f"<th class='false-col'>False ({len(false_sample)} 局面を表示)</th>\n")
            f.write("</tr><tr>\n")

            for cls, sample, css in [("True", true_sample, "true-col"), ("False", false_sample, "false-col")]:
                f.write(f"<td class='{css}'>")
                for idx in sample:
                    svg_name = render_and_save(idx)
                    f.write(
                        "<div class='pos'>"
                        f"<object data='{svg_name}' type='image/svg+xml'></object>"
                        f"<div>ply={int(plies[idx])}<br>score={float(scores[idx]):.0f}</div>"
                        "</div>"
                    )
                f.write("</td>\n")

            f.write("</tr></table>\n")

    print(f"Wrote: {html_path}")
    return html_path


def write_lift_json(output_dir, lift_table, dominant, plies, scores, n_experts, all_features, meta):
    """エキスパート分析結果を機械可読な JSON で保存する。"""
    n = len(dominant)
    counts = np.bincount(dominant, minlength=n_experts)
    experts = []
    for j in range(n_experts):
        mask = dominant == j
        ply_j = plies[mask]
        score_j = scores[mask]
        rows = lift_table.get(j, [])
        experts.append({
            "id": j,
            "n_assigned": int(counts[j]),
            "auto_label": auto_label_expert(rows),
            "mean_ply":   round(float(ply_j.mean()),   2) if len(ply_j) else None,
            "std_ply":    round(float(ply_j.std()),    2) if len(ply_j) else None,
            "mean_score": round(float(score_j.mean()), 2) if len(score_j) else None,
            "std_score":  round(float(score_j.std()),  2) if len(score_j) else None,
            "lift_table": [
                {
                    "feature":    fname,
                    "label":      _FEATURE_LABELS.get(fname, fname),
                    "lift":       round(lift, 4),
                    "lift_ci_lo": round(lo,   4),
                    "lift_ci_hi": round(hi,   4),
                    "rate_expert":round(rate_j,   4),
                    "rate_all":   round(rate_all, 4),
                }
                for fname, lift, lo, hi, rate_j, rate_all in rows
            ],
        })

    feature_rates = {
        fname: {
            "label": _FEATURE_LABELS.get(fname, fname),
            "rate":  round(float(fvals.mean()), 4),
            "count": int(fvals.sum()),
            "total": n,
        }
        for fname, fvals in all_features.items()
    }

    data = {
        "meta": {k: str(v) for k, v in meta.items()},
        "n_positions": n,
        "experts": experts,
        "feature_rates": feature_rates,
    }

    json_path = os.path.join(output_dir, "lift_analysis.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print(f"Wrote: {json_path}")
    return json_path


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


def write_html(output_dir, html_path, n_experts, dominant, weights, plies, scores, psfens, top_k, meta, lift_table=None):
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
            if lift_table and j in lift_table:
                label = html.escape(auto_label_expert(lift_table[j]))
                lift_html = _lift_details_html(lift_table[j])
                f.write(
                    f"<td>Expert {j}<br>assigned n={n_assigned}<br>"
                    f"<strong>{label}</strong>{lift_html}</td>\n"
                )
            else:
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

    print("Extracting shogi features for lift analysis ...")
    all_features = extract_all_features(psfens, plies, scores)
    lift_table = compute_lift_table(all_features, dominant, n_experts)
    print("  Done.")

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
        lift_table=lift_table,
    )
    print(f"Wrote: {html_path}")

    print("Writing machine-readable JSON ...")
    write_lift_json(
        output_dir=args.output_dir,
        lift_table=lift_table,
        dominant=dominant,
        plies=plies,
        scores=scores,
        n_experts=n_experts,
        all_features=all_features,
        meta=meta,
    )

    print("Writing feature check HTML ...")
    write_feature_check_html(
        output_dir=args.output_dir,
        features=all_features,
        psfens=psfens,
        plies=plies,
        scores=scores,
        n_per_class=10,
    )

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
