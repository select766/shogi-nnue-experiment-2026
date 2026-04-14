"""
Expert Blending: 重みブレンド＆量子化バイナリ出力。

gate_weights に基づいて N_EXPERTS 個の NNUE 重みを加重平均し、
やねうら王が直接メモリにロードできる量子化バイナリを出力する。

量子化ロジックは nnue-pytorch/serialize.py の NNUEWriter と同等。
"""

import struct

import torch
import numpy as np


# NNUE architecture constants (halfkp_256x2-32-32)
L1 = 256  # half dimensions (feature transformer output per side)
L2 = 32   # hidden layer 1
L3 = 32   # hidden layer 2

# Quantization constants (from serialize.py and nnue_common.h)
FT_SCALE = 127  # feature transformer quantization scale
WEIGHT_SCALE_BITS = 6
ACTIVATION_SCALE = 127.0
FC_BIAS_SCALE = (1 << WEIGHT_SCALE_BITS) * ACTIVATION_SCALE  # = 8128
FC_WEIGHT_SCALE = FC_BIAS_SCALE / ACTIVATION_SCALE  # = 64.0
OUTPUT_BIAS_SCALE = 9600.0  # kPonanzaConstant(600) * FV_SCALE(16)
OUTPUT_WEIGHT_SCALE = OUTPUT_BIAS_SCALE / ACTIVATION_SCALE  # ≈ 75.59


def blend_expert_weights(nnue_experts, gate_weights):
    """gate_weights で各 expert の重みを加重平均し、単一 NNUE の重み辞書を返す。

    Args:
        nnue_experts: NNUEExperts モジュール
        gate_weights: (n_experts,) テンソル。総和=1。

    Returns:
        dict: {
            'input_weight': (L1, num_features),
            'input_bias': (L1,),
            'l1_weight': (L2, 2*L1),
            'l1_bias': (L2,),
            'l2_weight': (L3, L2),
            'l2_bias': (L3,),
            'output_weight': (1, L3),
            'output_bias': (1,),
        }
    """
    gate_weights = gate_weights.to(nnue_experts.input_weight.device)

    blended = {}
    param_names = [
        'input_weight', 'input_bias',
        'l1_weight', 'l1_bias',
        'l2_weight', 'l2_bias',
        'output_weight', 'output_bias',
    ]
    for name in param_names:
        # param shape: (n_experts, *param_shape)
        param = getattr(nnue_experts, name).data
        # gate_weights を param の次元に合わせて reshape
        # gate_weights: (n_experts,) -> (n_experts, 1, 1, ...) for broadcasting
        w = gate_weights
        for _ in range(param.dim() - 1):
            w = w.unsqueeze(-1)
        blended[name] = (param * w).sum(dim=0)
        base_name = f"base_{name}"
        if hasattr(nnue_experts, base_name):
            blended[name] = blended[name] + getattr(nnue_experts, base_name).data

    return blended


def coalesce_ft_weights(weight, feature_set):
    """Feature transformer の重みを coalesce する（virtual features → real features に畳み込み）。

    HalfKP (非factorized) の場合は virtual features がないので、そのまま返す。
    HalfKP^ (factorized) の場合は torch.index_add で高速に処理する。

    Args:
        weight: (L1, num_features) — num_features は real + virtual
        feature_set: FeatureBlock インスタンス

    Returns:
        (L1, num_real_features) テンソル
    """
    if feature_set.num_virtual_features == 0:
        # HalfKP (non-factorized): no coalescing needed
        return weight

    # Factorized case: use get_virtual_to_real_features_gather_indices
    indices = feature_set.get_virtual_to_real_features_gather_indices()
    num_real = feature_set.num_real_features
    weight_coalesced = weight.new_zeros((weight.shape[0], num_real))

    # Build mapping: for each virtual feature, find which real feature it maps to
    # indices[i_real] = list of virtual feature indices that map to i_real
    # Use index_add for vectorized accumulation
    for i_real, i_virtuals in enumerate(indices):
        for i_virtual in i_virtuals:
            weight_coalesced[:, i_real] += weight[:, i_virtual]

    return weight_coalesced


def quantize_and_pack(blended, feature_set):
    """ブレンド済み重みを量子化し、ヘッダなしの生バイナリ (little-endian) として返す。

    出力フォーマット:
        [FT bias: 256 × int16]
        [FT weight: num_real_features × 256 × int16]  ← [features][half_dim] 順
        [L1 bias: 32 × int32]
        [L1 weight: 32 × padded_512 × int8]  ← [out][padded_in] 順
        [L2 bias: 32 × int32]
        [L2 weight: 32 × 32 × int8]
        [Output bias: 1 × int32]
        [Output weight: 1 × 32 × int8]

    Args:
        blended: blend_expert_weights() の出力
        feature_set: FeatureBlock インスタンス

    Returns:
        bytes: 量子化済みバイナリデータ
    """
    # numpy 変換はCPUテンソルのみ対応のため、デバイス非依存で量子化できるよう
    # ここで一度CPUへ集約する。
    blended = {k: v.detach().to("cpu") for k, v in blended.items()}

    buf = bytearray()

    # --- Feature Transformer ---
    # Bias: int16, scale=127
    ft_bias = blended['input_bias'].mul(FT_SCALE).round().to(torch.int16)
    buf.extend(ft_bias.flatten().numpy().tobytes())

    # Weight: coalesce then int16, scale=127
    ft_weight = coalesce_ft_weights(blended['input_weight'], feature_set)
    ft_weight = ft_weight.mul(FT_SCALE).round().to(torch.int16)
    # Transpose: PyTorch stores as [256][num_features], C++ expects [num_features][256]
    buf.extend(ft_weight.transpose(0, 1).contiguous().flatten().numpy().tobytes())

    # --- FC Layers ---
    def pack_fc_layer(weight, bias, is_output=False):
        if is_output:
            bias_scale = OUTPUT_BIAS_SCALE
        else:
            bias_scale = FC_BIAS_SCALE
        weight_scale = bias_scale / ACTIVATION_SCALE
        max_weight = 127.0 / weight_scale

        # Bias: int32
        q_bias = bias.mul(bias_scale).round().to(torch.int32)
        buf.extend(q_bias.flatten().numpy().tobytes())

        # Weight: int8, with padding to multiple of 32
        q_weight = weight.clamp(-max_weight, max_weight).mul(weight_scale).round().to(torch.int8)
        num_input = q_weight.shape[1]
        if num_input % 32 != 0:
            padded = num_input + 32 - (num_input % 32)
            new_w = torch.zeros(q_weight.shape[0], padded, dtype=torch.int8)
            new_w[:, :num_input] = q_weight
            q_weight = new_w
        # Stored as [outputs][padded_inputs]
        buf.extend(q_weight.flatten().numpy().tobytes())

    pack_fc_layer(blended['l1_weight'], blended['l1_bias'])
    pack_fc_layer(blended['l2_weight'], blended['l2_bias'])
    pack_fc_layer(blended['output_weight'], blended['output_bias'], is_output=True)

    return bytes(buf)


def write_nnue_file(blended, feature_set, output_path):
    """ブレンド済み重みを完全な .nnue ファイルとして書き出す（テスト用）。

    serialize.py の NNUEWriter と同じヘッダフォーマットを使用する。

    Args:
        blended: blend_expert_weights() の出力
        feature_set: FeatureBlock インスタンス
        output_path: 出力ファイルパス
    """
    VERSION = 0x7AF32F16

    buf = bytearray()

    def int32(v):
        buf.extend(struct.pack("<I", v & 0xFFFFFFFF))

    # --- FC hash calculation ---
    # InputSlice hash
    prev_hash = 0xEC42E90D
    prev_hash ^= (L1 * 2)

    # Layer specs: [(out_features, is_output), ...]
    layers = [(L2, False), (L3, False), (1, True)]
    fc_hash = prev_hash
    for out_features, is_output in layers:
        layer_hash = 0xCC03DAE4
        layer_hash = (layer_hash + out_features) & 0xFFFFFFFF
        layer_hash ^= fc_hash >> 1
        layer_hash ^= (fc_hash << 31) & 0xFFFFFFFF
        if out_features != 1:
            # ClippedReLU hash
            layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
        fc_hash = layer_hash

    # --- Header ---
    ft_hash = feature_set.hash ^ (L1 * 2)
    int32(VERSION)
    int32(fc_hash ^ ft_hash)  # network hash
    description = b"Features=HalfKP(Friend)[125388->256x2],"
    description += b"Network=AffineTransform[1<-256](ClippedReLU[256](AffineTransform[256<-256]"
    description += b"(ClippedReLU[256](AffineTransform[256<-512](InputSlice[512(0:512)])))))"
    int32(len(description))
    buf.extend(description)

    # --- Feature Transformer ---
    int32(ft_hash)

    # Pack raw weights (same as quantize_and_pack but with header)
    raw = quantize_and_pack(blended, feature_set)

    # Split raw into FT part and FC part, inserting FC hash in between
    # FT part: bias + weight
    num_real_features = feature_set.num_real_features
    ft_bias_size = L1 * 2  # int16
    ft_weight_size = num_real_features * L1 * 2  # int16
    ft_total = ft_bias_size + ft_weight_size

    buf.extend(raw[:ft_total])

    # --- FC layers ---
    int32(fc_hash)
    buf.extend(raw[ft_total:])

    with open(output_path, 'wb') as f:
        f.write(buf)


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.insert(0, '.')
    import features as nnue_features
    from train_nnue.train_expert_blending import ExpertBlendingLightningModule

    parser = argparse.ArgumentParser(
        description="Export blended NNUE weights from Expert Blending checkpoint")
    parser.add_argument("--checkpoint", required=True,
                        help="Expert Blending Lightning checkpoint (.ckpt)")
    parser.add_argument("--features", default="HalfKP",
                        help="Feature set name (default: HalfKP)")
    parser.add_argument("--output", required=True,
                        help="Output .nnue file path")
    parser.add_argument("--gate-weights", default=None,
                        help="Comma-separated gate weights (default: uniform)")
    args = parser.parse_args()

    feature_set = nnue_features.get_feature_set_from_name(args.features)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    # Extract NNUEExperts from state_dict
    from train_nnue.expert_blending_model import (
        NNUEExperts,
        detect_blend_mode_from_state_dict,
    )
    # Determine n_experts from checkpoint
    n_experts = state_dict['model.nnue_experts.input_bias'].shape[0]
    num_features = state_dict['model.nnue_experts.input_weight'].shape[2]
    blend_mode = detect_blend_mode_from_state_dict(state_dict)
    print(f"n_experts={n_experts}, num_features={num_features}, blend_mode={blend_mode}")

    experts = NNUEExperts(n_experts, num_features, blend_mode=blend_mode)
    expert_state = {}
    prefix = 'model.nnue_experts.'
    for k, v in state_dict.items():
        if k.startswith(prefix):
            expert_state[k[len(prefix):]] = v
    experts.load_state_dict(expert_state)

    # Gate weights
    if args.gate_weights:
        gw = torch.tensor([float(x) for x in args.gate_weights.split(',')])
    else:
        gw = torch.ones(n_experts) / n_experts
    print(f"Gate weights: {gw.tolist()}")

    # Blend and export
    blended = blend_expert_weights(experts, gw)
    write_nnue_file(blended, feature_set, args.output)
    print(f"Written: {args.output}")

    # Report sizes
    import os
    size = os.path.getsize(args.output)
    print(f"File size: {size:,} bytes ({size / 1024 / 1024:.1f} MB)")
