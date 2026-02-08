"""Shuffle and split PackedSfenValue dataset into train/val."""
import numpy as np
import glob
import os
import sys

RECORD_SIZE = 40  # bytes per PackedSfenValue
VAL_RATIO = 0.02  # 2% for validation
SEED = 42


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "subset_tanuki-.nnue-pytorch-2024-07-30.1"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset_shuffled"

    bin_files = sorted(glob.glob(os.path.join(input_dir, "*.bin")))
    print(f"Found {len(bin_files)} .bin files")

    # Read all files into a single buffer
    buffers = []
    total_records = 0
    for f in bin_files:
        size = os.path.getsize(f)
        n_records = size // RECORD_SIZE
        total_records += n_records
        print(f"  {os.path.basename(f)}: {n_records} records")
        buffers.append(np.fromfile(f, dtype=np.dtype((np.void, RECORD_SIZE))))

    print(f"Total records: {total_records}")

    # Concatenate
    print("Concatenating...")
    data = np.concatenate(buffers)
    del buffers

    # Shuffle
    print("Shuffling...")
    rng = np.random.default_rng(SEED)
    rng.shuffle(data)

    # Split
    val_count = int(len(data) * VAL_RATIO)
    train_count = len(data) - val_count
    print(f"Train: {train_count}, Val: {val_count}")

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    print(f"Writing {train_path}...")
    data[:train_count].tofile(train_path)
    print(f"Writing {val_path}...")
    data[train_count:].tofile(val_path)
    print("Done!")


if __name__ == "__main__":
    main()
