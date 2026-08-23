"""Root-grouped DNN/NNUE batches with one router decision shared by K leaves."""

import json
import mmap
import os

import cshogi
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

import dlshogi.cppshogi as dcppshogi
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

from train_nnue.expert_blending_dataset import (
    HCPE_DTYPE,
    LEGACY_RECORD_BYTES,
    PSFEN_BYTES,
    FixedNumBatchesDataset,
    _create_sparse_batch_provider,
)


class RootGroupedDataset(IterableDataset):
    def __init__(
        self,
        directory,
        feature_set_name,
        root_batch_size,
        *,
        device="cpu",
        teacher_cache_path=None,
        start_root=0,
    ):
        super().__init__()
        self.directory = os.path.abspath(directory)
        self.feature_set_name = feature_set_name
        self.root_batch_size = int(root_batch_size)
        self.device = device
        self.start_root = int(start_root)
        with open(os.path.join(self.directory, "metadata.json")) as file:
            self.metadata = json.load(file)
        if self.metadata.get("format") != "root-grouped-paired-v1":
            raise ValueError("unsupported root-grouped data format")
        self.group_size = int(self.metadata["group_size"])
        self.num_roots = int(self.metadata["num_roots"])
        if self.root_batch_size <= 0 or self.group_size <= 1:
            raise ValueError("root batch size must be positive and group size must exceed 1")
        if self.start_root < 0 or self.start_root % self.root_batch_size != 0:
            raise ValueError("start_root must be a non-negative root-batch multiple")
        if self.start_root >= self.num_roots:
            raise ValueError("start_root must be smaller than num_roots")
        self.roots_path = os.path.join(self.directory, "roots.bin")
        self.leaves_path = os.path.join(self.directory, "leaves.bin")
        expected_roots = self.num_roots * LEGACY_RECORD_BYTES
        expected_leaves = expected_roots * self.group_size
        if os.path.getsize(self.roots_path) != expected_roots:
            raise ValueError("roots.bin size does not match metadata")
        if os.path.getsize(self.leaves_path) != expected_leaves:
            raise ValueError("leaves.bin size does not match metadata")
        self.teacher_cache = None
        if teacher_cache_path:
            self.teacher_cache = np.load(teacher_cache_path, mmap_mode="r")
            if (
                self.teacher_cache.ndim != 2
                or self.teacher_cache.shape[0] <= self.start_root
            ):
                raise ValueError("teacher cache does not cover start_root")

    def __iter__(self):
        return _RootGroupedIterator(self)


class _RootGroupedIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.root_batch_size = dataset.root_batch_size
        self.group_size = dataset.group_size
        self.num_roots = dataset.num_roots
        self.device = dataset.device
        self.root_position = dataset.start_root
        self.teacher_cache = dataset.teacher_cache
        self.leaf_provider = _create_sparse_batch_provider(
            dataset.feature_set_name,
            dataset.leaves_path,
            self.root_batch_size * self.group_size,
            dataset.device,
        )
        for _ in range(dataset.start_root // dataset.root_batch_size):
            next(self.leaf_provider)
        self.root_file = open(dataset.roots_path, "rb")
        self.root_mmap = mmap.mmap(
            self.root_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.board = cshogi.Board()

    def __iter__(self):
        return self

    def __next__(self):
        us, them, white, black, outcome, score, ply = next(self.leaf_provider)
        leaf_count = int(us.shape[0])
        if leaf_count != self.root_batch_size * self.group_size:
            raise RuntimeError("native leaf provider returned an unexpected batch size")
        root_indices = (
            np.arange(self.root_position, self.root_position + self.root_batch_size)
            % self.num_roots
        )
        hcps = np.zeros((self.root_batch_size, 32), dtype=np.uint8)
        hcp_tmp = np.zeros(1, dtype=cshogi.dtypeHcp)
        psfen_tmp = np.zeros(1, dtype=cshogi.PackedSfen)
        for batch_index, root_index in enumerate(root_indices):
            offset = int(root_index) * LEGACY_RECORD_BYTES
            psfen_tmp[0]["sfen"] = np.frombuffer(
                self.root_mmap[offset : offset + PSFEN_BYTES], dtype=np.uint8
            )
            self.board.set_psfen(psfen_tmp)
            self.board.to_hcp(hcp_tmp)
            hcps[batch_index] = hcp_tmp[0]
        hcpe = np.zeros(self.root_batch_size, dtype=HCPE_DTYPE)
        hcpe["hcp"] = hcps
        features1 = np.zeros(
            (self.root_batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32
        )
        features2 = np.zeros(
            (self.root_batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32
        )
        move = np.zeros(self.root_batch_size, dtype=np.int64)
        result = np.zeros(self.root_batch_size, dtype=np.float32)
        value = np.zeros(self.root_batch_size, dtype=np.float32)
        dcppshogi.hcpe_decode_with_value(
            hcpe.view(np.uint8).reshape(self.root_batch_size, -1),
            features1,
            features2,
            move,
            result,
            value,
        )
        x1 = torch.from_numpy(features1).to(self.device)
        x2 = torch.from_numpy(features2).to(self.device)
        teacher = None
        if self.teacher_cache is not None:
            if int(root_indices.max()) >= self.teacher_cache.shape[0]:
                raise IndexError("teacher cache exhausted for the requested root batch")
            teacher = torch.from_numpy(
                np.asarray(self.teacher_cache[root_indices], dtype=np.float32).copy()
            ).to(self.device)
        self.root_position = (
            self.root_position + self.root_batch_size
        ) % self.num_roots
        batch = (x1, x2, us, them, white, black, outcome, score, ply)
        return batch + ((teacher,) if teacher is not None else ())

    def __del__(self):
        if getattr(self, "root_mmap", None) is not None:
            self.root_mmap.close()
            self.root_mmap = None
        if getattr(self, "root_file", None) is not None:
            self.root_file.close()
            self.root_file = None


def create_root_grouped_loaders(
    train_directory,
    val_directory,
    feature_set_name,
    root_batch_size,
    device,
    epoch_roots,
    max_val_roots,
    *,
    train_shuffle_buffer_size=0,
    seed=42,
    train_teacher_cache=None,
    val_teacher_cache=None,
):
    train = RootGroupedDataset(
        train_directory,
        feature_set_name,
        root_batch_size,
        device=device,
        teacher_cache_path=train_teacher_cache,
    )
    val = RootGroupedDataset(
        val_directory,
        feature_set_name,
        root_batch_size,
        device=device,
        teacher_cache_path=val_teacher_cache,
    )
    if train.group_size != val.group_size:
        raise ValueError("train and validation group sizes must match")
    train_batches = min(train.num_roots, epoch_roots) // root_batch_size
    val_batches = min(val.num_roots, max_val_roots) // root_batch_size
    if train_batches == 0 or val_batches == 0:
        raise ValueError("root counts must cover at least one complete batch")
    train_loader = DataLoader(
        FixedNumBatchesDataset(
            train,
            train_batches,
            reset_on_epoch_start=False,
            shuffle_buffer_size=train_shuffle_buffer_size,
            seed=seed,
        ),
        batch_size=None,
        batch_sampler=None,
    )
    val_loader = DataLoader(
        FixedNumBatchesDataset(
            val, val_batches, reset_on_epoch_start=True, seed=seed
        ),
        batch_size=None,
        batch_sampler=None,
    )
    return train_loader, val_loader, train.group_size
