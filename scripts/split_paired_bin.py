#!/usr/bin/env python3
"""Split paired 80-byte records into dnn.bin and nnue.bin (40 bytes each)."""

import argparse
import os
from pathlib import Path

import numpy as np

PAIRED_RECORD_BYTES = 80
RECORD_BYTES = 40
NNUE_OFFSET = 40


def split_paired_bin(input_path: Path, output_dir: Path, chunk_records: int = 262_144) -> tuple[Path, Path]:
    src_size = input_path.stat().st_size
    if src_size % PAIRED_RECORD_BYTES != 0:
        raise ValueError(
            f"Input size must be a multiple of {PAIRED_RECORD_BYTES} bytes: {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dnn_path = output_dir / "dnn.bin"
    nnue_path = output_dir / "nnue.bin"
    dnn_tmp = output_dir / f"dnn.bin.tmp.{os.getpid()}"
    nnue_tmp = output_dir / f"nnue.bin.tmp.{os.getpid()}"

    with input_path.open("rb") as src, dnn_tmp.open("wb") as dnn_out, nnue_tmp.open("wb") as nnue_out:
        while True:
            block = src.read(chunk_records * PAIRED_RECORD_BYTES)
            if not block:
                break
            if len(block) % PAIRED_RECORD_BYTES != 0:
                raise ValueError(f"Corrupted paired block in {input_path}")
            arr = np.frombuffer(block, dtype=np.uint8).reshape(-1, PAIRED_RECORD_BYTES)
            dnn_out.write(arr[:, :RECORD_BYTES].tobytes())
            nnue_out.write(arr[:, NNUE_OFFSET:NNUE_OFFSET + RECORD_BYTES].tobytes())

        dnn_out.flush()
        os.fsync(dnn_out.fileno())
        nnue_out.flush()
        os.fsync(nnue_out.fileno())

    os.replace(dnn_tmp, dnn_path)
    os.replace(nnue_tmp, nnue_path)
    return dnn_path, nnue_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split paired 80B records into dnn.bin/nnue.bin")
    parser.add_argument("--input", required=True, help="Path to paired .bin file (80B/record)")
    parser.add_argument("--output-dir", required=True, help="Output directory for dnn.bin/nnue.bin")
    parser.add_argument("--chunk-records", type=int, default=262_144, help="Records processed per chunk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    dnn_path, nnue_path = split_paired_bin(input_path, output_dir, chunk_records=args.chunk_records)
    records = dnn_path.stat().st_size // RECORD_BYTES
    print(f"Split complete: records={records}")
    print(f"  dnn : {dnn_path} ({dnn_path.stat().st_size} bytes)")
    print(f"  nnue: {nnue_path} ({nnue_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
