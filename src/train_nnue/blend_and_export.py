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
        # ナイーブな broadcast 乗算 + sum は (n_experts, *) の中間テンソルを
        # 物理的に確保するため、input_weight (約 1 GB) で大きなコストになる。
        # (n_experts,) と (n_experts, prod) の matmul に落とすと BLAS GEMV を
        # 利用でき、中間メモリも要らない。
        param_flat = param.reshape(param.shape[0], -1)
        blended_flat = torch.matmul(gate_weights, param_flat)
        blended[name] = blended_flat.reshape(param.shape[1:])
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


def _pack_fc_layer(buf, weight, bias, is_output=False):
    """FC 層 (bias int32 + weight int8 padded) を buf に追記する。"""
    if is_output:
        bias_scale = OUTPUT_BIAS_SCALE
    else:
        bias_scale = FC_BIAS_SCALE
    weight_scale = bias_scale / ACTIVATION_SCALE
    max_weight = 127.0 / weight_scale

    q_bias = bias.mul(bias_scale).round().to(torch.int32)
    buf.extend(q_bias.flatten().numpy().tobytes())

    q_weight = weight.clamp(-max_weight, max_weight).mul(weight_scale).round().to(torch.int8)
    num_input = q_weight.shape[1]
    if num_input % 32 != 0:
        padded = num_input + 32 - (num_input % 32)
        new_w = torch.zeros(q_weight.shape[0], padded, dtype=torch.int8)
        new_w[:, :num_input] = q_weight
        q_weight = new_w
    buf.extend(q_weight.flatten().numpy().tobytes())


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
    _pack_fc_layer(buf, blended['l1_weight'], blended['l1_bias'])
    _pack_fc_layer(buf, blended['l2_weight'], blended['l2_bias'])
    _pack_fc_layer(buf, blended['output_weight'], blended['output_bias'], is_output=True)

    return bytes(buf)


class FastBlendingPacker:
    """毎 go の blend+pack を高速化するための事前計算キャッシュ。

    入力 weight (FT) は約 64MB (32M int16) を占めるため、毎回の
      - FT_SCALE 倍 (mul) … float ops 32M
      - transpose(0,1).contiguous() … 64MB のメモリ shuffle
    が支配的。これを起動時に一回だけ済ませておくと、毎回の合成は
      - matmul で C++ メモリ順の (num_features, L1) を直接得る
      - round → int16 → tobytes
    だけになる。

    制限: 現状は HalfKP 非 factorized (num_virtual_features == 0) のみ対応。
    factorized の場合は coalesce が必要なので fallback パスを使う。
    """

    def __init__(self, nnue_experts, feature_set):
        if feature_set.num_virtual_features != 0:
            raise NotImplementedError(
                "FastBlendingPacker currently supports only non-factorized features"
            )
        self.feature_set = feature_set
        self.experts = nnue_experts

        # nnue_experts.input_weight: (E, L1, F)
        # → (E, F, L1) に転置 + × FT_SCALE してキャッシュ。
        iw = nnue_experts.input_weight.data.detach().to("cpu")
        ft_w_scaled = (iw.permute(0, 2, 1).contiguous().mul(FT_SCALE))
        E, F, L1_ = ft_w_scaled.shape
        self._E = E
        self._F = F
        self._L1 = L1_
        self.ft_w_flat = ft_w_scaled.reshape(E, F * L1_)

        # input_bias: (E, L1) は × FT_SCALE のみ
        self.ft_b_scaled = nnue_experts.input_bias.data.detach().to("cpu").mul(FT_SCALE)

        # residual mode: base 重みも事前にスケール済みキャッシュ
        bw = getattr(nnue_experts, "base_input_weight", None)
        if bw is not None:
            base_w = bw.data.detach().to("cpu")
            # (L1, F) → (F, L1) 転置 + × FT_SCALE
            self.ft_w_base_scaled = base_w.t().contiguous().mul(FT_SCALE)
            self._ft_w_base_scaled_np = self.ft_w_base_scaled.numpy()
        else:
            self.ft_w_base_scaled = None
            self._ft_w_base_scaled_np = None
        bb = getattr(nnue_experts, "base_input_bias", None)
        if bb is not None:
            self.ft_b_base_scaled = bb.data.detach().to("cpu").mul(FT_SCALE)
        else:
            self.ft_b_base_scaled = None

        # FT weight 用の事前確保バッファ。matmul((1,E),(E,F*L1),out=...) は
        # 出力 shape (1, F*L1) を要求するので、そのまま (1, F*L1) で確保し
        # (F, L1) view を numpy 経由で持つ。
        self._ft_w_buf_f32 = torch.empty((1, F * L1_), dtype=torch.float32)
        self._ft_w_buf_f32_view = self._ft_w_buf_f32.view(F, L1_)
        self._ft_w_buf_f32_np = self._ft_w_buf_f32_view.numpy()
        self._ft_w_buf_i16 = torch.empty((F, L1_), dtype=torch.int16)
        self._ft_w_buf_i16_np = self._ft_w_buf_i16.numpy()

        # 払い出すバイト数 (FT_bias int16 + FT_weight int16 + FC packed)
        # FT 部分のサイズは確定。FC は _pack_fc_layer の paddding 仕様から計算する。
        ft_bytes = (L1_ + F * L1_) * 2
        # L1 (32 out × padded(2*L1=512) input) + L2 (32×32) + output (1×32)
        L2 = nnue_experts.l1_weight.shape[1]
        L3 = nnue_experts.l2_weight.shape[1]
        l1_in = nnue_experts.l1_weight.shape[2]
        l2_in = nnue_experts.l2_weight.shape[2]
        out_in = nnue_experts.output_weight.shape[2]
        def fc_bytes(out_dim, in_dim):
            padded = ((in_dim + 31) // 32) * 32
            return out_dim * 4 + out_dim * padded  # bias int32 + weight int8(padded)
        self.payload_size = (
            ft_bytes
            + fc_bytes(L2, l1_in)
            + fc_bytes(L3, l2_in)
            + fc_bytes(1, out_in)
        )

    def write_to_stream(self, gate_weights, stream):
        """ブレンド済み量子化重みを `stream` (file-like) に直接書き出す。

        64 MB クラスの中間 ``bytes`` オブジェクトを作らないことが目的。
        """
        gate = gate_weights.detach().to(self.ft_w_flat.device).to(torch.float32)

        # --- FT bias (small) ---
        ft_b = torch.matmul(gate, self.ft_b_scaled)
        if self.ft_b_base_scaled is not None:
            ft_b = ft_b + self.ft_b_base_scaled
        ft_b_i16 = ft_b.round().to(torch.int16).numpy()
        stream.write(memoryview(ft_b_i16).cast("B"))

        # --- FT weight (大): 事前確保バッファに matmul、numpy で round/cast、
        # memoryview を直接 stream に渡して bytes 割当を回避 ---
        gate_2d = gate.unsqueeze(0)  # (1, E)
        torch.matmul(gate_2d, self.ft_w_flat, out=self._ft_w_buf_f32)
        f32_np = self._ft_w_buf_f32_np  # (F, L1) の numpy view
        if self._ft_w_base_scaled_np is not None:
            np.add(f32_np, self._ft_w_base_scaled_np, out=f32_np)
        np.rint(f32_np, out=f32_np)
        np.copyto(self._ft_w_buf_i16_np, f32_np, casting="unsafe")
        stream.write(memoryview(self._ft_w_buf_i16_np).cast("B"))

        # --- FC 層: 小サイズ。bytearray にまとめてから書く ---
        fc_buf = bytearray()
        fc_blended = {}
        for name in ("l1_weight", "l1_bias", "l2_weight", "l2_bias",
                     "output_weight", "output_bias"):
            param = getattr(self.experts, name).data
            flat = param.reshape(param.shape[0], -1)
            blended = torch.matmul(gate, flat).reshape(param.shape[1:])
            base = getattr(self.experts, f"base_{name}", None)
            if base is not None:
                blended = blended + base.data
            fc_blended[name] = blended.detach().to("cpu")
        _pack_fc_layer(fc_buf, fc_blended["l1_weight"], fc_blended["l1_bias"])
        _pack_fc_layer(fc_buf, fc_blended["l2_weight"], fc_blended["l2_bias"])
        _pack_fc_layer(fc_buf, fc_blended["output_weight"], fc_blended["output_bias"],
                       is_output=True)
        stream.write(fc_buf)

    def blend_and_pack(self, gate_weights):
        """互換用: bytes を返すバージョン。テスト用途や 1 回限りの呼び出し向け。"""
        import io
        bio = io.BytesIO()
        self.write_to_stream(gate_weights, bio)
        return bio.getvalue()


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
