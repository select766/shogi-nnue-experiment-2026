"""
DNN推論サーバー: やねうら王の子プロセスとして常駐し、
SFEN を受け取ってブレンド済み量子化 NNUE 重みを返す。

プロトコル (stdin/stdout pipe):
    [C++ → Python] SFEN文字列 + "\\n"
    [Python → C++] 4バイト: データサイズ(uint32 LE) + 重みバイナリ(固定サイズ)

起動時に "ready\\n" を stdout に出力して準備完了を通知する。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.dnn_inference_server \\
        --checkpoint <expert_blending.ckpt> \\
        --backbone-weights <dlshogi_model.npz> \\
        --features HalfKP \\
        --n-experts 8
"""

import argparse
import struct
import sys
import time

import numpy as np
import torch
import cshogi

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
import dlshogi.cppshogi as dcppshogi

from train_nnue.expert_blending_model import (
    DNNBackbone,
    DNNAdapter,
    NNUEExperts,
    ExpertBlendingModel,
    load_backbone,
)
from train_nnue.blend_and_export import blend_expert_weights, quantize_and_pack
from train_nnue.verify_dlshogi import encode_position


def load_model_from_checkpoint(checkpoint_path, backbone_weights_path,
                               feature_set, n_experts, device='cpu'):
    """チェックポイントから ExpertBlendingModel をロードする。

    Args:
        checkpoint_path: Expert Blending Lightning .ckpt パス
        backbone_weights_path: dlshogi .npz 重みパス
        feature_set: FeatureBlock インスタンス
        n_experts: expert 数
        device: デバイス

    Returns:
        (backbone, adapter, nnue_experts, feature_set)
    """
    # Load backbone
    backbone = load_backbone(backbone_weights_path, device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']

    # Reconstruct adapter
    # Determine hidden_dim from checkpoint
    adapter_fc1_weight = state_dict['model.adapter.fc1.weight']
    hidden_dim = adapter_fc1_weight.shape[0]
    in_channels = adapter_fc1_weight.shape[1]
    adapter = DNNAdapter(in_channels=in_channels, hidden_dim=hidden_dim, n_experts=n_experts)
    adapter_state = {}
    prefix = 'model.adapter.'
    for k, v in state_dict.items():
        if k.startswith(prefix):
            adapter_state[k[len(prefix):]] = v
    adapter.load_state_dict(adapter_state)
    adapter.to(device)

    # Reconstruct nnue_experts
    num_features = state_dict['model.nnue_experts.input_weight'].shape[2]
    nnue_experts = NNUEExperts(n_experts, num_features)
    expert_state = {}
    prefix = 'model.nnue_experts.'
    for k, v in state_dict.items():
        if k.startswith(prefix):
            expert_state[k[len(prefix):]] = v
    nnue_experts.load_state_dict(expert_state)
    nnue_experts.to(device)

    return backbone, adapter, nnue_experts


def infer_gate_weights(backbone, adapter, board, device='cpu'):
    """局面から gate_weights を推論する。

    Args:
        backbone: DNNBackbone
        adapter: DNNAdapter
        board: cshogi.Board

    Returns:
        gate_weights: (n_experts,) テンソル
    """
    features1, features2 = encode_position(board)
    x1 = torch.from_numpy(features1).to(device)
    x2 = torch.from_numpy(features2).to(device)

    with torch.no_grad():
        feat = backbone(x1, x2)
        gate_weights = adapter(feat, training=False)

    return gate_weights[0]  # (n_experts,)


def main():
    parser = argparse.ArgumentParser(description="DNN inference server for Expert Blending")
    parser.add_argument("--checkpoint", required=True,
                        help="Expert Blending Lightning checkpoint (.ckpt)")
    parser.add_argument("--backbone-weights", required=True,
                        help="dlshogi .npz weights path")
    parser.add_argument("--features", default="HalfKP",
                        help="Feature set name (default: HalfKP)")
    parser.add_argument("--n-experts", type=int, required=True,
                        help="Number of NNUE experts")
    parser.add_argument("--device", default="cpu",
                        help="Device (cpu/cuda)")
    parser.add_argument("--log", default=None,
                        help="Log file path for debug output (default: stderr)")
    args = parser.parse_args()

    # Setup logging to stderr or file
    if args.log:
        log_f = open(args.log, 'w')
    else:
        log_f = sys.stderr

    def log(msg):
        print(msg, file=log_f, flush=True)

    log(f"Loading model...")
    log(f"  checkpoint: {args.checkpoint}")
    log(f"  backbone: {args.backbone_weights}")
    log(f"  features: {args.features}")
    log(f"  n_experts: {args.n_experts}")
    log(f"  device: {args.device}")

    # Import features from nnue-pytorch
    sys.path.insert(0, '.')
    import features as nnue_features
    feature_set = nnue_features.get_feature_set_from_name(args.features)

    backbone, adapter, nnue_experts = load_model_from_checkpoint(
        args.checkpoint, args.backbone_weights,
        feature_set, args.n_experts, args.device
    )
    backbone.eval()
    adapter.eval()
    nnue_experts.eval()

    log(f"Model loaded. num_features={feature_set.num_features}, "
        f"num_real_features={feature_set.num_real_features}")

    board = cshogi.Board()

    # Signal ready
    sys.stdout.buffer.write(b"ready\n")
    sys.stdout.buffer.flush()
    log("ready")

    # Main loop
    request_count = 0
    while True:
        try:
            line = sys.stdin.readline()
        except EOFError:
            break
        if not line:
            break

        sfen = line.strip()
        if not sfen:
            continue

        t0 = time.time()

        # Set position
        board.set_sfen(sfen)

        # Infer gate weights
        gate_weights = infer_gate_weights(backbone, adapter, board, args.device)

        t1 = time.time()

        # Blend and quantize
        blended = blend_expert_weights(nnue_experts, gate_weights)
        weight_bytes = quantize_and_pack(blended, feature_set)

        t2 = time.time()

        # Send response: size (uint32 LE) + data
        size = len(weight_bytes)
        sys.stdout.buffer.write(struct.pack("<I", size))
        sys.stdout.buffer.write(weight_bytes)
        sys.stdout.buffer.flush()

        request_count += 1
        log(f"[{request_count}] sfen={sfen[:40]}... "
            f"gate={[f'{g:.3f}' for g in gate_weights.tolist()]} "
            f"infer={t1-t0:.3f}s blend={t2-t1:.3f}s size={size}")

    log(f"Server shutting down. Total requests: {request_count}")
    if args.log:
        log_f.close()


if __name__ == "__main__":
    main()
