"""
Expert Blending checkpoint → やねうら王ロード形式 への変換ツール。

出力ディレクトリには 3 ファイルを書き出す:

  backbone.onnx
      DNNBackbone (dlshogi ResNet10) + DNNAdapter (FC→softmax) を結合した ONNX。
      入力:
          input1: float32 (N, FEATURES1_NUM, 9, 9)
          input2: float32 (N, FEATURES2_NUM, 9, 9)
      出力:
          gate:   float32 (N, n_experts)   # softmax 済み

  head.bin
      NNUEExperts の重みを「事前量子化＋やねうら王のメモリ順」で書き出した
      バイナリ。各 expert は次の順で連続配置される (E 個分先頭から):
          ft_bias    : int16[L1]
          ft_weight  : int16[F][L1]              ← (F, L1) 順 (= memcpy 順)
          fc1_bias   : int32[L2]
          fc1_weight : int8[L2][padded(2*L1)]
          fc2_bias   : int32[L3]
          fc2_weight : int8[L3][padded(L2)]
          out_bias   : int32[1]
          out_weight : int8[1][padded(L3)]
      blend_mode == "residual" の場合は、E 個の expert の後に同じレイアウトで
      base_* (1 セット) を追加する。

  head.json
      ファイル全体のメタ情報 (json)。やねうら王側はこちらを最初に読む。

設計指針:
  - 量子化は export 時に確定 (= scripts/serialize.py / blend_and_export.py と同じ式)。
  - 各 expert を独立に量子化するため、ブレンド後再量子化する場合と僅かに値が異なる
    可能性があるが、gate (sum=1) で重み付き平均する範囲では誤差は数 LSB 以内に
    収まる。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_nnue.expert_blending_model import (
    DNNAdapter,
    DNNBackbone,
    NNUEExperts,
    detect_blend_mode_from_state_dict,
    load_backbone,
)
from train_nnue.blend_and_export import (
    L1, L2, L3,
    FT_SCALE,
    FC_BIAS_SCALE,
    FC_WEIGHT_SCALE,
    OUTPUT_BIAS_SCALE,
    OUTPUT_WEIGHT_SCALE,
    ACTIVATION_SCALE,
    coalesce_ft_weights,
)


HEAD_VERSION = 1

# head.bin の先頭に置く固定長バイナリヘッダ (C++ 側がパース)。
# 128 バイト固定。すべて little-endian。
HEAD_BIN_HEADER_SIZE = 128
HEAD_BIN_MAGIC = b"EBHEAD01"  # 8 bytes (matches version=1)


def _build_head_bin_header(
    *,
    n_experts: int,
    num_features: int,
    blend_mode: str,
    features1_num: int,
    features2_num: int,
    breakdown: dict,
) -> bytes:
    """head.bin の先頭 128 バイトを組み立てる。"""
    blend_mode_int = {"weighted": 0, "residual": 1}[blend_mode]
    fields = [
        HEAD_BIN_MAGIC,                                # 8B  magic
        struct.pack("<I", HEAD_VERSION),               # 4B  version
        struct.pack("<I", n_experts),                  # 4B
        struct.pack("<I", num_features),               # 4B
        struct.pack("<I", L1),                         # 4B
        struct.pack("<I", L2),                         # 4B
        struct.pack("<I", L3),                         # 4B
        struct.pack("<I", blend_mode_int),             # 4B
        struct.pack("<I", features1_num),              # 4B
        struct.pack("<I", features2_num),              # 4B
        struct.pack("<I", breakdown["ft_bias_bytes"]),       # 4B
        struct.pack("<I", breakdown["ft_weight_bytes"]),     # 4B
        struct.pack("<I", breakdown["fc1_bias_bytes"]),      # 4B
        struct.pack("<I", breakdown["fc1_weight_bytes"]),    # 4B
        struct.pack("<I", breakdown["fc2_bias_bytes"]),      # 4B
        struct.pack("<I", breakdown["fc2_weight_bytes"]),    # 4B
        struct.pack("<I", breakdown["out_bias_bytes"]),      # 4B
        struct.pack("<I", breakdown["out_weight_bytes"]),    # 4B
        struct.pack("<I", breakdown["total_bytes"]),         # 4B  expert_total_bytes
    ]
    blob = b"".join(fields)
    assert len(blob) <= HEAD_BIN_HEADER_SIZE, (
        f"header overflow: {len(blob)} > {HEAD_BIN_HEADER_SIZE}"
    )
    return blob + b"\x00" * (HEAD_BIN_HEADER_SIZE - len(blob))


class GatingNetwork(nn.Module):
    """ONNX export 用の薄いラッパ: backbone + adapter を直列に呼び、softmax 済み
    gate weights を返すだけ。"""

    def __init__(self, backbone: DNNBackbone, adapter: DNNAdapter):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def forward(self, x1, x2):
        feat = self.backbone(x1, x2)
        gate = self.adapter(feat, training=False)
        return gate


def _resolve_features_constants(feature_set):
    """nnue-pytorch features.py から num_features を取り、コアレッセ後の F を返す。"""
    return feature_set.num_real_features


def _resolve_dlshogi_feature_dims():
    """dlshogi の FEATURES1_NUM / FEATURES2_NUM を返す (cppshogi 不要)。"""
    from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
    return FEATURES1_NUM, FEATURES2_NUM


def _padded(in_dim: int, multiple: int = 32) -> int:
    return ((in_dim + multiple - 1) // multiple) * multiple


def _quantize_ft_bias(bias_t: torch.Tensor) -> np.ndarray:
    """(L1,) float → int16."""
    arr = bias_t.detach().to("cpu").mul(FT_SCALE).round().to(torch.int16).numpy()
    return arr


def _quantize_ft_weight(weight_t: torch.Tensor, feature_set) -> np.ndarray:
    """(L1, num_features) float → int16, やねうら王のメモリ順 (F, L1)。"""
    coalesced = coalesce_ft_weights(weight_t.detach().to("cpu"), feature_set)  # (L1, F)
    transposed = coalesced.transpose(0, 1).contiguous()                        # (F, L1)
    return transposed.mul(FT_SCALE).round().to(torch.int16).numpy()


def _quantize_fc_layer(weight_t: torch.Tensor, bias_t: torch.Tensor, *, is_output: bool):
    """1 つの FC 層を (bias int32, weight int8 padded) に量子化。

    Returns: (bias_int32: (out,), weight_int8: (out, padded_in))
    """
    bias_scale = OUTPUT_BIAS_SCALE if is_output else FC_BIAS_SCALE
    weight_scale = bias_scale / ACTIVATION_SCALE
    max_weight = 127.0 / weight_scale

    w = weight_t.detach().to("cpu")
    b = bias_t.detach().to("cpu")

    q_bias = b.mul(bias_scale).round().to(torch.int32).numpy()  # (out,)
    q_weight = (
        w.clamp(-max_weight, max_weight).mul(weight_scale).round().to(torch.int8).numpy()
    )  # (out, in)

    out, in_dim = q_weight.shape
    pad = _padded(in_dim) - in_dim
    if pad > 0:
        q_weight = np.concatenate(
            [q_weight, np.zeros((out, pad), dtype=np.int8)], axis=1
        )
    return q_bias, q_weight


def _expert_blob(experts: NNUEExperts, feature_set, k: int, *, use_base: bool = False) -> bytes:
    """単一 expert (k 番目) の量子化済みバイト列を返す。use_base=True の場合は
    base_* を量子化して返す (residual モード用)。"""
    buf = bytearray()

    if use_base:
        ft_bias = experts.base_input_bias.data
        ft_weight = experts.base_input_weight.data
        l1_w = experts.base_l1_weight.data
        l1_b = experts.base_l1_bias.data
        l2_w = experts.base_l2_weight.data
        l2_b = experts.base_l2_bias.data
        out_w = experts.base_output_weight.data
        out_b = experts.base_output_bias.data
    else:
        ft_bias = experts.input_bias.data[k]
        ft_weight = experts.input_weight.data[k]
        l1_w = experts.l1_weight.data[k]
        l1_b = experts.l1_bias.data[k]
        l2_w = experts.l2_weight.data[k]
        l2_b = experts.l2_bias.data[k]
        out_w = experts.output_weight.data[k]
        out_b = experts.output_bias.data[k]

    buf.extend(_quantize_ft_bias(ft_bias).tobytes())
    buf.extend(_quantize_ft_weight(ft_weight, feature_set).tobytes())

    for w, b, is_out in (
        (l1_w, l1_b, False),
        (l2_w, l2_b, False),
        (out_w, out_b, True),
    ):
        q_bias, q_weight = _quantize_fc_layer(w, b, is_output=is_out)
        buf.extend(q_bias.tobytes())
        buf.extend(q_weight.tobytes())

    return bytes(buf)


def _per_expert_byte_breakdown(num_features: int) -> dict:
    """1 expert あたりの各セクションのバイト数 (head.json に書く)。"""
    F = num_features
    fc_in = 2 * L1
    fc_w_l1 = L2 * _padded(fc_in)
    fc_w_l2 = L3 * _padded(L2)
    fc_w_out = 1 * _padded(L3)
    return {
        "ft_bias_bytes": L1 * 2,
        "ft_weight_bytes": F * L1 * 2,
        "fc1_bias_bytes": L2 * 4,
        "fc1_weight_bytes": fc_w_l1,
        "fc2_bias_bytes": L3 * 4,
        "fc2_weight_bytes": fc_w_l2,
        "out_bias_bytes": 1 * 4,
        "out_weight_bytes": fc_w_out,
        "total_bytes": (
            L1 * 2
            + F * L1 * 2
            + L2 * 4 + fc_w_l1
            + L3 * 4 + fc_w_l2
            + 1 * 4 + fc_w_out
        ),
    }


def _load_checkpoint(checkpoint_path, n_experts):
    """checkpoint から (experts, adapter, blend_mode, num_features, hidden_dim) を返す。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    blend_mode = detect_blend_mode_from_state_dict(state_dict)

    if state_dict["model.nnue_experts.input_bias"].shape[0] != n_experts:
        raise ValueError(
            f"Checkpoint n_experts mismatch: "
            f"checkpoint={state_dict['model.nnue_experts.input_bias'].shape[0]}, "
            f"arg={n_experts}"
        )

    num_features = state_dict["model.nnue_experts.input_weight"].shape[2]

    experts = NNUEExperts(n_experts, num_features, blend_mode=blend_mode)
    expert_state = {}
    prefix = "model.nnue_experts."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            expert_state[k[len(prefix):]] = v
    experts.load_state_dict(expert_state)
    experts.eval()

    adapter_fc1_weight = state_dict["model.adapter.fc1.weight"]
    hidden_dim = adapter_fc1_weight.shape[0]
    in_channels = adapter_fc1_weight.shape[1]
    adapter = DNNAdapter(
        in_channels=in_channels, hidden_dim=hidden_dim, n_experts=n_experts
    )
    adapter_state = {}
    prefix = "model.adapter."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            adapter_state[k[len(prefix):]] = v
    adapter.load_state_dict(adapter_state)
    adapter.eval()

    return experts, adapter, blend_mode, num_features, hidden_dim


def export_backbone_onnx(
    backbone: DNNBackbone,
    adapter: DNNAdapter,
    output_path: Path,
    *,
    features1_num: int,
    features2_num: int,
):
    """gating network を ONNX で書き出す。"""
    model = GatingNetwork(backbone, adapter)
    model.eval()

    dummy_x1 = torch.zeros(1, features1_num, 9, 9, dtype=torch.float32)
    dummy_x2 = torch.zeros(1, features2_num, 9, 9, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_x1, dummy_x2),
        str(output_path),
        input_names=["input1", "input2"],
        output_names=["gate"],
        dynamic_axes={
            "input1": {0: "batch_size"},
            "input2": {0: "batch_size"},
            "gate": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )


def write_head_bin(
    experts: NNUEExperts,
    feature_set,
    output_path: Path,
    *,
    blend_mode: str,
    features1_num: int,
    features2_num: int,
):
    """先頭に 128B バイナリヘッダ + E 個の expert (+residual の場合 base) を書き出す。"""
    n_experts = experts.n_experts
    num_features = feature_set.num_real_features
    breakdown = _per_expert_byte_breakdown(num_features)

    header = _build_head_bin_header(
        n_experts=n_experts,
        num_features=num_features,
        blend_mode=blend_mode,
        features1_num=features1_num,
        features2_num=features2_num,
        breakdown=breakdown,
    )

    with open(output_path, "wb") as f:
        f.write(header)
        for k in range(n_experts):
            f.write(_expert_blob(experts, feature_set, k))
        if blend_mode == "residual":
            f.write(_expert_blob(experts, feature_set, k=0, use_base=True))


def write_head_json(
    output_path: Path,
    *,
    n_experts: int,
    num_features: int,
    blend_mode: str,
    feature_set_name: str,
    features1_num: int,
    features2_num: int,
    adapter_hidden_dim: int,
    backbone_weights_basename: str,
):
    """head.json を書き出す。"""
    breakdown = _per_expert_byte_breakdown(num_features)
    meta = {
        "version": HEAD_VERSION,
        "format": "expert-blending-yaneuraou-v1",
        "n_experts": n_experts,
        "num_features": num_features,           # coalesce 後 (real features)
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "blend_mode": blend_mode,
        "feature_set": feature_set_name,
        "input_features1_channels": features1_num,
        "input_features2_channels": features2_num,
        "adapter_hidden_dim": adapter_hidden_dim,
        "backbone_onnx": "backbone.onnx",
        "head_bin": "head.bin",
        "source_backbone_weights": backbone_weights_basename,
        "quantization": {
            "ft_scale": FT_SCALE,
            "fc_bias_scale": FC_BIAS_SCALE,
            "fc_weight_scale": FC_WEIGHT_SCALE,
            "output_bias_scale": OUTPUT_BIAS_SCALE,
            "output_weight_scale": OUTPUT_WEIGHT_SCALE,
            "activation_scale": ACTIVATION_SCALE,
        },
        "expert_byte_layout": breakdown,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Expert Blending checkpoint → やねうら王 ロード形式 (dir) 変換"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Expert Blending Lightning checkpoint (.ckpt)")
    parser.add_argument("--backbone-weights", required=True,
                        help="dlshogi .npz 重みパス (DNN backbone 用)")
    parser.add_argument("--features", default="HalfKP",
                        help="NNUE feature set 名 (default: HalfKP)")
    parser.add_argument("--n-experts", type=int, required=True,
                        help="expert 数")
    parser.add_argument("--output-dir", required=True,
                        help="出力ディレクトリ (作成される)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # nnue-pytorch features を解決
    from train_nnue.dnn_inference_server import load_nnue_feature_set
    feature_set = load_nnue_feature_set(args.features)
    if feature_set.num_virtual_features != 0:
        # FastBlendingPacker と同じ理由で factorized は今は対象外。
        raise NotImplementedError(
            "factorized 特徴量 (HalfKP^ など) は未対応。HalfKP のみ対応。"
        )
    num_features_real = feature_set.num_real_features
    print(f"feature_set={args.features} num_real_features={num_features_real}")

    features1_num, features2_num = _resolve_dlshogi_feature_dims()
    print(f"dlshogi FEATURES1_NUM={features1_num} FEATURES2_NUM={features2_num}")

    # checkpoint
    print(f"loading checkpoint: {args.checkpoint}")
    experts, adapter, blend_mode, num_features_ckpt, hidden_dim = _load_checkpoint(
        args.checkpoint, args.n_experts
    )
    if num_features_ckpt != num_features_real:
        raise ValueError(
            f"checkpoint num_features={num_features_ckpt} != "
            f"feature_set.num_real_features={num_features_real}"
        )
    print(f"  blend_mode={blend_mode} hidden_dim={hidden_dim}")

    # backbone
    print(f"loading backbone weights: {args.backbone_weights}")
    backbone = load_backbone(args.backbone_weights)
    backbone.eval()

    # ONNX export
    onnx_path = out_dir / "backbone.onnx"
    print(f"exporting ONNX: {onnx_path}")
    export_backbone_onnx(
        backbone, adapter, onnx_path,
        features1_num=features1_num, features2_num=features2_num,
    )

    # head.bin
    head_bin = out_dir / "head.bin"
    print(f"writing head.bin: {head_bin}")
    write_head_bin(
        experts, feature_set, head_bin,
        blend_mode=blend_mode,
        features1_num=features1_num,
        features2_num=features2_num,
    )
    head_size = head_bin.stat().st_size
    print(f"  head.bin size = {head_size:,} bytes "
          f"({head_size / (1024*1024):.1f} MiB)")

    # head.json
    head_json = out_dir / "head.json"
    print(f"writing head.json: {head_json}")
    write_head_json(
        head_json,
        n_experts=args.n_experts,
        num_features=num_features_real,
        blend_mode=blend_mode,
        feature_set_name=args.features,
        features1_num=features1_num,
        features2_num=features2_num,
        adapter_hidden_dim=hidden_dim,
        backbone_weights_basename=Path(args.backbone_weights).name,
    )

    # サイズ整合性チェック (ヘッダ込み)
    breakdown = _per_expert_byte_breakdown(num_features_real)
    expected = HEAD_BIN_HEADER_SIZE + breakdown["total_bytes"] * args.n_experts
    if blend_mode == "residual":
        expected += breakdown["total_bytes"]
    if head_size != expected:
        raise RuntimeError(
            f"head.bin size mismatch: actual={head_size} expected={expected}"
        )

    print("done.")


if __name__ == "__main__":
    main()
