"""
DNN推論サーバー: やねうら王の子プロセスとして常駐し、
SFEN を受け取ってブレンド済み量子化 NNUE 重みを返す。

プロトコル (stdin/stdout pipe):
    [C++ → Python] SFEN文字列 + "\\n"
    [Python → C++] 32バイト: 合成比率(float32 LE x 8)
                     + 4バイト: データサイズ(uint32 LE)
                     + 重みバイナリ(固定サイズ)

起動時に "ready\\n" を stdout に出力して準備完了を通知する。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.dnn_inference_server \\
        --checkpoint <expert_blending.ckpt> \\
        --backbone-type dnn \\
        --backbone-weights <dlshogi_model.npz> \\
        --features HalfKP \\
        --n-experts 8
"""

import argparse
import contextlib
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cshogi

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
import dlshogi.cppshogi as dcppshogi

from train_nnue.expert_blending_model import (
    DNNBackbone,
    DNNAdapter,
    NNUEBackbone,
    NNUEExperts,
    ExpertBlendingModel,
    detect_blend_mode_from_state_dict,
    load_backbone,
)
from train_nnue.blend_and_export import (
    FastBlendingPacker,
    blend_expert_weights,
    quantize_and_pack,
)
from train_nnue.verify_dlshogi import encode_position


def get_nnue_pytorch_dir(require_nnue_dataset=False):
    """nnue-pytorch ディレクトリを解決する。"""
    candidates = []

    # Optional explicit override.
    nnue_dir_env = os.environ.get("NNUE_PYTORCH_DIR")
    if nnue_dir_env:
        candidates.append(Path(nnue_dir_env))

    # Repository-local default: <repo>/nnue-pytorch
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "nnue-pytorch")

    # Backward compatibility: current working directory (when cd nnue-pytorch)
    candidates.append(Path.cwd())

    for d in candidates:
        features_py = d / "features.py"
        nnue_dataset_py = d / "nnue_dataset.py"
        if not features_py.is_file():
            continue
        if require_nnue_dataset and not nnue_dataset_py.is_file():
            continue
        return d

    raise ModuleNotFoundError(
        "Could not find nnue-pytorch directory with required files. "
        "Set NNUE_PYTORCH_DIR or run from a workspace containing nnue-pytorch."
    )


def load_nnue_feature_set(feature_name):
    """nnue-pytorch/features.py から feature set を解決する。"""
    nnue_dir = get_nnue_pytorch_dir(require_nnue_dataset=False)
    sys.path.insert(0, str(nnue_dir))
    import features as nnue_features
    return nnue_features.get_feature_set_from_name(feature_name)


def load_make_sparse_batch_from_fens(nnue_dir=None):
    """nnue-pytorch/nnue_dataset.py から make_sparse_batch_from_fens を解決する。"""
    if nnue_dir is None:
        nnue_dir = get_nnue_pytorch_dir(require_nnue_dataset=True)
    sys.path.insert(0, str(nnue_dir))
    from nnue_dataset import make_sparse_batch_from_fens
    return make_sparse_batch_from_fens


def load_model_from_checkpoint(
    checkpoint_path,
    backbone_weights_path,
    feature_set,
    n_experts,
    device='cpu',
    backbone_type='dnn',
):
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
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict']
    blend_mode = detect_blend_mode_from_state_dict(state_dict)

    backbone_type = backbone_type.lower()
    if backbone_type == "nnue":
        backbone_num_features = state_dict['model.backbone.input.weight'].shape[1]
        backbone_n_experts = state_dict['model.backbone.output.weight'].shape[0]
        if backbone_n_experts != n_experts:
            raise ValueError(
                f"Checkpoint n_experts mismatch: checkpoint={backbone_n_experts}, "
                f"arg={n_experts}"
            )
        backbone = NNUEBackbone(backbone_num_features, n_experts=backbone_n_experts)
        backbone_state = {}
        prefix = 'model.backbone.'
        for k, v in state_dict.items():
            if k.startswith(prefix):
                backbone_state[k[len(prefix):]] = v
        backbone.load_state_dict(backbone_state)
        backbone.to(device)
        adapter = None
    elif backbone_type == "dnn":
        if not backbone_weights_path:
            raise ValueError("backbone_weights_path is required when backbone_type='dnn'")
        backbone = load_backbone(backbone_weights_path, device)

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
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")

    # Reconstruct nnue_experts
    num_features = state_dict['model.nnue_experts.input_weight'].shape[2]
    nnue_experts = NNUEExperts(n_experts, num_features, blend_mode=blend_mode)
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


def infer_gate_weights_nnue(
    backbone,
    board,
    feature_set,
    make_sparse_batch_from_fens,
    device='cpu',
):
    """局面から NNUEBackbone で gate_weights を推論する。"""
    sfen = board.sfen()
    with suppress_stdout_fd():
        batch = make_sparse_batch_from_fens(feature_set, [sfen], [0], [0], [0])
        us, them, white, black, outcome, score, ply = batch.contents.get_tensors(device)
    with torch.no_grad():
        gate_weights = backbone(us, them, white, black, training=False)
    return gate_weights[0]  # (n_experts,)


@contextlib.contextmanager
def suppress_stdout_fd():
    """C/C++拡張を含む stdout 汚染を防ぐため、プロセスstdoutを一時的に捨てる。"""
    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)


def main():
    parser = argparse.ArgumentParser(description="DNN inference server for Expert Blending")
    parser.add_argument("--checkpoint", required=True,
                        help="Expert Blending Lightning checkpoint (.ckpt)")
    parser.add_argument(
        "--backbone-type",
        default="dnn",
        choices=["dnn", "nnue"],
        help="Backbone type: dnn or nnue",
    )
    parser.add_argument("--backbone-weights", required=False,
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
    if args.backbone_type == "dnn" and not args.backbone_weights:
        parser.error("--backbone-weights is required when --backbone-type dnn")

    # Setup logging to stderr or file
    if args.log:
        log_f = open(args.log, 'w')
    else:
        log_f = sys.stderr

    def log(msg):
        print(msg, file=log_f, flush=True)

    log(f"Loading model...")
    log(f"  checkpoint: {args.checkpoint}")
    log(f"  backbone_type: {args.backbone_type}")
    log(f"  backbone: {args.backbone_weights}")
    log(f"  features: {args.features}")
    log(f"  n_experts: {args.n_experts}")
    log(f"  device: {args.device}")

    feature_set = load_nnue_feature_set(args.features)
    make_sparse_batch_from_fens = None
    if args.backbone_type == "nnue":
        nnue_dir = get_nnue_pytorch_dir(require_nnue_dataset=True)
        # nnue_dataset.py は shared library を cwd 相対で探索するため、
        # nnueモードでは作業ディレクトリを nnue-pytorch に固定する。
        os.chdir(nnue_dir)
        log(f"  nnue_pytorch_dir: {nnue_dir}")
        make_sparse_batch_from_fens = load_make_sparse_batch_from_fens(nnue_dir)

    backbone, adapter, nnue_experts = load_model_from_checkpoint(
        args.checkpoint, args.backbone_weights,
        feature_set, args.n_experts, args.device, args.backbone_type
    )
    backbone.eval()
    if adapter is not None:
        adapter.eval()
    nnue_experts.eval()

    log(f"Model loaded. num_features={feature_set.num_features}, "
        f"num_real_features={feature_set.num_real_features}")

    # 事前計算した転置 + スケール済み input_weight を持つ高速合成機を構築。
    # factorized HalfKP の場合は coalesce が必要なので fallback。
    try:
        packer = FastBlendingPacker(nnue_experts, feature_set)
        log("FastBlendingPacker enabled (FT input_weight pre-transposed/pre-scaled)")
    except NotImplementedError as e:
        packer = None
        log(f"FastBlendingPacker disabled: {e}")

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
        if args.backbone_type == "nnue":
            gate_weights = infer_gate_weights_nnue(
                backbone,
                board,
                feature_set,
                make_sparse_batch_from_fens,
                args.device,
            )
        else:
            gate_weights = infer_gate_weights(backbone, adapter, board, args.device)

        t1 = time.time()

        # Blend and quantize
        if packer is not None:
            weight_bytes = packer.blend_and_pack(gate_weights)
        else:
            blended = blend_expert_weights(nnue_experts, gate_weights)
            weight_bytes = quantize_and_pack(blended, feature_set)

        t2 = time.time()

        gate_weights_list = gate_weights.detach().cpu().tolist()
        if len(gate_weights_list) != 8:
            raise RuntimeError(
                f"Protocol requires exactly 8 experts, but got {len(gate_weights_list)}"
            )

        # Send response:
        # [8 x float32 blending weights] + [uint32 size] + [payload bytes]
        size = len(weight_bytes)
        sys.stdout.buffer.write(struct.pack("<8f", *gate_weights_list))
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
