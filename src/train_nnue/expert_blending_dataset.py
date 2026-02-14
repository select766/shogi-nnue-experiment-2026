"""
Dataset for Expert Blending model training.

Reads packed SFEN .bin files (40 bytes/record) and generates both
NNUE sparse features and dlshogi dense features for the same positions.

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.expert_blending_dataset \
        --bin data/val.bin --feature-set "HalfKP" --batch-size 256
"""

import mmap
import os
import struct
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

import cshogi
import dlshogi.cppshogi as dcppshogi
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

import nnue_dataset
import features as nnue_features

# Packed SFEN record layout (40 bytes)
RECORD_BYTES = 40
# offset 0: HCP (32 bytes), offset 32: score (int16), offset 34: move (uint16),
# offset 36: gamePly (uint16), offset 38: game_result (int8), offset 39: padding (uint8)

# HCPE dtype for cppshogi.hcpe_decode_with_value
HCPE_DTYPE = np.dtype([
    ('hcp', np.uint8, 32),
    ('eval', np.int16),
    ('bestMove16', np.uint16),
    ('gameResult', np.uint8),
    ('padding', np.uint8),
])


class ExpertBlendingDataset(IterableDataset):
    """Packed SFEN .bin から NNUE + DNN 両方の特徴量をバッチ生成する。"""

    def __init__(self, bin_path, feature_set_name, batch_size, device='cpu', shuffle=True):
        super().__init__()
        self.bin_path = bin_path
        self.feature_set_name = feature_set_name
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.num_records = os.path.getsize(bin_path) // RECORD_BYTES
        self.feature_set = nnue_features.get_feature_set_from_name(feature_set_name)

    def __iter__(self):
        return _ExpertBlendingIterator(self)


class _ExpertBlendingIterator:
    """Stateful iterator for ExpertBlendingDataset."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = dataset.batch_size
        self.device = dataset.device
        self.feature_set = dataset.feature_set
        self.num_records = dataset.num_records

        # Open file and create mmap
        self.file = open(dataset.bin_path, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        # Create shuffled index
        self.indices = np.arange(self.num_records)
        if dataset.shuffle:
            np.random.shuffle(self.indices)
        self.pos = 0

        # Reusable board for HCP -> SFEN conversion
        self.board = cshogi.Board()

    def __iter__(self):
        return self

    def __next__(self):
        # Collect valid records until we have a full batch (or exhaust data)
        hcp_list = []
        score_list = []
        ply_list = []
        result_list = []

        while len(hcp_list) < self.batch_size and self.pos < self.num_records:
            # Grab a chunk of indices to scan
            chunk_end = min(self.pos + self.batch_size - len(hcp_list), self.num_records)
            chunk_indices = self.indices[self.pos:chunk_end]
            self.pos = chunk_end

            for idx in chunk_indices:
                offset = int(idx) * RECORD_BYTES
                record = self.mm[offset:offset + RECORD_BYTES]
                hcp = np.frombuffer(record[:32], dtype=np.uint8).copy()
                # Validate HCP by attempting to decode
                try:
                    self.board.set_hcp(hcp)
                except RuntimeError:
                    continue
                hcp_list.append(hcp)
                score_list.append(struct.unpack_from('<h', record, 32)[0])
                ply_list.append(struct.unpack_from('<H', record, 36)[0])
                result_list.append(struct.unpack_from('<b', record, 38)[0])
                if len(hcp_list) >= self.batch_size:
                    break

        if not hcp_list:
            self._close()
            raise StopIteration

        actual_batch_size = len(hcp_list)
        hcps = np.array(hcp_list)
        scores = np.array(score_list, dtype=np.int16)
        plies = np.array(ply_list, dtype=np.uint16)
        results = np.array(result_list, dtype=np.int8)

        # --- dlshogi dense features ---
        hcpe_batch = np.zeros(actual_batch_size, dtype=HCPE_DTYPE)
        hcpe_batch['hcp'] = hcps

        features1 = np.zeros((actual_batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32)
        features2 = np.zeros((actual_batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)
        move_dummy = np.zeros(actual_batch_size, dtype=np.int64)
        result_dummy = np.zeros(actual_batch_size, dtype=np.float32)
        value_dummy = np.zeros(actual_batch_size, dtype=np.float32)

        dcppshogi.hcpe_decode_with_value(
            hcpe_batch.view(np.uint8).reshape(actual_batch_size, -1),
            features1, features2, move_dummy, result_dummy, value_dummy
        )

        x1 = torch.from_numpy(features1).to(device=self.device)
        x2 = torch.from_numpy(features2).to(device=self.device)

        # --- NNUE sparse features ---
        fens = []
        for i in range(actual_batch_size):
            self.board.set_hcp(hcps[i])
            fens.append(self.board.sfen())

        scores_int = scores.astype(np.int32).tolist()
        plies_int = plies.astype(np.int32).tolist()
        # Convert game_result: packed SFEN uses 1=win,-1=loss,0=draw
        # nnue_dataset expects same convention
        results_int = results.astype(np.int32).tolist()

        sparse_batch = nnue_dataset.make_sparse_batch_from_fens(
            self.feature_set, fens, scores_int, plies_int, results_int
        )
        us, them, white, black, outcome, score, ply = sparse_batch.contents.get_tensors(self.device)
        nnue_dataset.destroy_sparse_batch(sparse_batch)

        return x1, x2, us, them, white, black, outcome, score, ply

    def _close(self):
        if hasattr(self, 'mm') and self.mm is not None:
            self.mm.close()
            self.mm = None
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        self._close()


class FixedNumBatchesDataset(Dataset):
    """Wraps an IterableDataset to yield a fixed number of batches per epoch."""

    def __init__(self, dataset, num_batches):
        super().__init__()
        self.dataset = dataset
        self.iter = iter(self.dataset)
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return next(self.iter)


def create_data_loaders(train_bin, val_bin, feature_set_name, batch_size, device, epoch_size):
    """学習/検証用DataLoaderのペアを返す。

    Args:
        train_bin: 学習用 packed SFEN .bin パス
        val_bin: 検証用 packed SFEN .bin パス
        feature_set_name: NNUE特徴セット名 (例: "HalfKP")
        batch_size: バッチサイズ
        device: デバイス文字列 (例: "cuda:0")
        epoch_size: 1エポックあたりの局面数

    Returns:
        (train_loader, val_loader) のタプル
    """
    train_dataset = ExpertBlendingDataset(
        train_bin, feature_set_name, batch_size, device=device, shuffle=True
    )
    val_dataset = ExpertBlendingDataset(
        val_bin, feature_set_name, batch_size, device=device, shuffle=False
    )

    num_train_batches = (epoch_size + batch_size - 1) // batch_size
    val_records = os.path.getsize(val_bin) // RECORD_BYTES
    num_val_batches = (min(val_records, 1_000_000) + batch_size - 1) // batch_size

    train_loader = DataLoader(
        FixedNumBatchesDataset(train_dataset, num_train_batches),
        batch_size=None,
        batch_sampler=None,
    )
    val_loader = DataLoader(
        FixedNumBatchesDataset(val_dataset, num_val_batches),
        batch_size=None,
        batch_sampler=None,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ExpertBlendingDataset verification")
    parser.add_argument("--bin", required=True, help="Path to packed SFEN .bin file")
    parser.add_argument("--feature-set", default="HalfKP", help="NNUE feature set name")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to test")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"=== ExpertBlendingDataset Verification ===")
    print(f"File: {args.bin}")
    print(f"Feature set: {args.feature_set}")
    print(f"Batch size: {args.batch_size}")

    dataset = ExpertBlendingDataset(
        args.bin, args.feature_set, args.batch_size, device=args.device, shuffle=True
    )
    print(f"Total records: {dataset.num_records}")

    # --- Shape and value range test ---
    print(f"\n--- Shape & Value Range Test ({args.num_batches} batches) ---")
    total_positions = 0
    t0 = time.time()
    for i, batch in enumerate(dataset):
        if i >= args.num_batches:
            break
        x1, x2, us, them, white, black, outcome, score, ply = batch
        bs = x1.shape[0]
        total_positions += bs

        print(f"\nBatch {i}: {bs} positions")
        print(f"  x1:      {x1.shape} dtype={x1.dtype} range=[{x1.min():.1f}, {x1.max():.1f}]")
        print(f"  x2:      {x2.shape} dtype={x2.dtype} range=[{x2.min():.1f}, {x2.max():.1f}]")
        print(f"  us:      {us.shape} dtype={us.dtype}")
        print(f"  them:    {them.shape} dtype={them.dtype}")
        print(f"  white:   {white.shape} (sparse)")
        print(f"  black:   {black.shape} (sparse)")
        print(f"  outcome: {outcome.shape} range=[{outcome.min():.2f}, {outcome.max():.2f}]")
        print(f"  score:   {score.shape} range=[{score.min():.1f}, {score.max():.1f}]")
        print(f"  ply:     {ply.shape} range=[{ply.min():.0f}, {ply.max():.0f}]")

    elapsed = time.time() - t0
    print(f"\n--- Performance ---")
    print(f"  {total_positions} positions in {elapsed:.2f}s")
    print(f"  {total_positions / elapsed:.0f} positions/sec")

    # --- Correspondence test: verify NNUE and DNN features come from same position ---
    print(f"\n--- Correspondence Test ---")
    dataset_check = ExpertBlendingDataset(
        args.bin, args.feature_set, 4, device='cpu', shuffle=False
    )
    board = cshogi.Board()
    for batch in dataset_check:
        x1, x2, us, them, white, black, outcome, score, ply = batch
        bs = x1.shape[0]
        # Re-read the first few records directly for comparison
        with open(args.bin, 'rb') as f:
            for j in range(bs):
                f.seek(j * RECORD_BYTES)
                hcp = np.frombuffer(f.read(32), dtype=np.uint8).copy()
                board.set_hcp(hcp)
                sfen = board.sfen()

                # Verify dlshogi features by re-encoding single position
                hcpe_single = np.zeros(1, dtype=HCPE_DTYPE)
                hcpe_single[0]['hcp'] = hcp
                f1 = np.zeros((1, FEATURES1_NUM, 9, 9), dtype=np.float32)
                f2 = np.zeros((1, FEATURES2_NUM, 9, 9), dtype=np.float32)
                mv = np.zeros(1, dtype=np.int64)
                res = np.zeros(1, dtype=np.float32)
                val = np.zeros(1, dtype=np.float32)
                dcppshogi.hcpe_decode_with_value(
                    hcpe_single.view(np.uint8).reshape(1, -1), f1, f2, mv, res, val
                )

                x1_match = np.allclose(x1[j].numpy(), f1[0])
                x2_match = np.allclose(x2[j].numpy(), f2[0])
                print(f"  Position {j}: {sfen}")
                print(f"    x1 match: {x1_match}, x2 match: {x2_match}")
                if not x1_match or not x2_match:
                    print("    WARNING: feature mismatch!")
        break  # Only check first batch

    # --- Model forward test ---
    print(f"\n--- Model Forward Test ---")
    try:
        from train_nnue.expert_blending_model import (
            ExpertBlendingModel, DNNBackbone, DNNAdapter, NNUEExperts,
        )
        from dlshogi.network.policy_value_network_resnet10_swish import PolicyValueNetwork as DlshogiPVNet

        # Construct model components with random weights
        pv_net = DlshogiPVNet()
        backbone = DNNBackbone(pv_net)
        n_experts = 4
        # DNNBackbone output channels = 192 (see backbone forward: u21 shape)
        adapter = DNNAdapter(192, n_experts=n_experts)
        nnue_experts = NNUEExperts(n_experts, dataset.feature_set.num_features)
        model = ExpertBlendingModel(backbone, adapter, nnue_experts)
        model.eval()

        dataset_fwd = ExpertBlendingDataset(
            args.bin, args.feature_set, 8, device='cpu', shuffle=False
        )
        for batch in dataset_fwd:
            x1, x2, us, them, white, black, outcome, score, ply = batch
            with torch.no_grad():
                value = model(x1, x2, us, them, white, black, training=False)
            print(f"  Forward pass OK: input batch={x1.shape[0]}, output={value.shape}")
            break
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Model forward test skipped: {e}")

    print("\n=== Done ===")
