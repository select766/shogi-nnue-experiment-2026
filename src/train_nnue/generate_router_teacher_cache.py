"""Generate aligned per-expert loss caches for router distillation."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import features as nnue_features
from train_nnue.diagnose_gate import (
    infer_model_configuration,
    per_record_loss,
)
from train_nnue.expert_blending_dataset import (
    LEGACY_RECORD_BYTES,
    _create_sparse_batch_provider,
)
from train_nnue.expert_blending_model import create_expert_blending_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument("--backbone-weights", required=True)
    parser.add_argument("--nnue-checkpoint", required=True)
    parser.add_argument("--feature-set", default="HalfKP")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-positions", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("teacher cache generation requires CUDA")
    if args.batch_size <= 0 or args.max_positions <= 0:
        parser.error("batch size and max positions must be positive")

    nnue_bin_path = os.path.abspath(os.path.join(args.bin_dir, "nnue.bin"))
    record_count = os.path.getsize(nnue_bin_path) // LEGACY_RECORD_BYTES
    if args.max_positions > record_count:
        parser.error("max positions exceeds the input record count")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configuration = infer_model_configuration(checkpoint["state_dict"])
    hyperparameters = checkpoint.get("hyper_parameters", {})
    objective = {
        "lambda": float(hyperparameters.get("lambda_", 1.0)),
        "label_smoothing_eps": float(
            hyperparameters.get("label_smoothing_eps", 0.0)
        ),
        "score_scaling": float(hyperparameters.get("score_scaling", 361.0)),
    }
    feature_set = nnue_features.get_feature_set_from_name(args.feature_set)
    model = create_expert_blending_model(
        backbone_weights_path=args.backbone_weights,
        nnue_ckpt_path=args.nnue_checkpoint,
        feature_set=feature_set,
        n_experts=configuration["n_experts"],
        adapter_hidden=configuration["adapter_hidden"] or 128,
        backbone_type=configuration["backbone_type"],
        blend_mode=configuration["blend_mode"],
        device="cpu",
    )
    model_state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state)
    experts = model.nnue_experts.to(args.device).eval()
    n_experts = experts.n_experts
    del model_state, checkpoint, model

    provider = _create_sparse_batch_provider(
        args.feature_set,
        nnue_bin_path,
        args.batch_size,
        args.device,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(f"{output_path}.tmp.npy")
    cache = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(args.max_positions, n_experts),
    )

    processed = 0
    with torch.inference_mode():
        while processed < args.max_positions:
            us, them, white, black, outcome, score, _ = next(provider)
            if white.is_sparse:
                white = white.to_dense()
            if black.is_sparse:
                black = black.to_dense()
            expert_values = torch.cat(
                [
                    experts.forward_expert(index, us, them, white, black)
                    for index in range(n_experts)
                ],
                dim=1,
            )
            losses = torch.stack(
                [
                    per_record_loss(
                        expert_values[:, index : index + 1],
                        score,
                        outcome,
                        objective["score_scaling"],
                        objective["lambda"],
                        objective["label_smoothing_eps"],
                    )
                    for index in range(n_experts)
                ],
                dim=1,
            )
            take = min(int(losses.shape[0]), args.max_positions - processed)
            cache[processed : processed + take] = losses[:take].cpu().numpy()
            processed += take
            if processed % 10000 < take:
                print(f"Progress: {processed}/{args.max_positions}", flush=True)

    cache.flush()
    del cache
    os.replace(temporary_path, output_path)
    metadata = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_size": os.path.getsize(args.checkpoint),
        "bin_dir": os.path.abspath(args.bin_dir),
        "nnue_bin_size": os.path.getsize(nnue_bin_path),
        "positions": args.max_positions,
        "n_experts": n_experts,
        "dtype": "float32",
        "objective": objective,
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    print(f"Wrote cache: {output_path}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
