"""Optimize per-position gate weights against the actual blended NNUE loss."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import features as nnue_features
from train_nnue.diagnose_gate import infer_model_configuration, per_record_loss
from train_nnue.expert_blending_dataset import ExpertBlendingDataset
from train_nnue.expert_blending_model import create_expert_blending_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument("--expert-loss-cache", required=True)
    parser.add_argument("--backbone-weights", required=True)
    parser.add_argument("--nnue-checkpoint", required=True)
    parser.add_argument("--max-positions", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--prior-kl", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("optimized gate teacher generation requires CUDA")
    if args.steps <= 0 or args.lr <= 0.0 or args.prior_kl < 0.0:
        parser.error("steps/lr must be positive and prior-kl non-negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configuration = infer_model_configuration(checkpoint["state_dict"])
    hyperparameters = checkpoint.get("hyper_parameters", {})
    feature_set = nnue_features.get_feature_set_from_name("HalfKP")
    model = create_expert_blending_model(
        backbone_weights_path=args.backbone_weights,
        nnue_ckpt_path=args.nnue_checkpoint,
        feature_set=feature_set,
        n_experts=configuration["n_experts"],
        adapter_hidden=configuration["adapter_hidden"],
        backbone_type=configuration["backbone_type"],
        blend_mode=configuration["blend_mode"],
        device="cpu",
    )
    model.load_state_dict(
        {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
    )
    model.requires_grad_(False)
    model.to(args.device).eval()
    n_experts = model.nnue_experts.n_experts
    del checkpoint

    dataset = ExpertBlendingDataset(
        args.bin_dir,
        "HalfKP",
        args.batch_size,
        device=args.device,
        shuffle=False,
        teacher_cache_path=args.expert_loss_cache,
    )
    if args.max_positions > min(dataset.num_records, dataset.teacher_losses.shape[0]):
        parser.error("max positions exceeds dataset or expert-loss cache")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(f"{output_path}.tmp.npy")
    output = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(args.max_positions, 2 * n_experts),
    )
    processed = 0
    initial_loss_sum = 0.0
    optimized_loss_sum = 0.0
    entropy_sum = 0.0
    iterator = iter(dataset)
    while processed < args.max_positions:
        x1, x2, us, them, white, black, outcome, score, _, expert_losses = next(
            iterator
        )
        if white.is_sparse:
            white = white.to_dense()
        if black.is_sparse:
            black = black.to_dense()
        with torch.no_grad():
            features = model.backbone(x1, x2)
            initial_weights = model.adapter(features, training=False)
            initial_value = model.nnue_experts(
                initial_weights, us, them, white, black
            )
            initial_losses = per_record_loss(
                initial_value,
                score,
                outcome,
                float(hyperparameters.get("score_scaling", 361.0)),
                float(hyperparameters.get("lambda_", 1.0)),
                float(hyperparameters.get("label_smoothing_eps", 0.0)),
            )

        logits = initial_weights.clamp_min(1e-12).log().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=args.lr)
        for _ in range(args.steps):
            optimizer.zero_grad()
            weights = torch.softmax(logits, dim=-1)
            raw_value = model.nnue_experts(weights, us, them, white, black)
            task_losses = per_record_loss(
                raw_value,
                score,
                outcome,
                float(hyperparameters.get("score_scaling", 361.0)),
                float(hyperparameters.get("lambda_", 1.0)),
                float(hyperparameters.get("label_smoothing_eps", 0.0)),
            )
            prior_kl = (
                weights
                * (
                    (weights + 1e-12).log()
                    - (initial_weights + 1e-12).log()
                )
            ).sum(dim=-1)
            (task_losses + args.prior_kl * prior_kl).mean().backward()
            optimizer.step()

        with torch.no_grad():
            optimized_weights = torch.softmax(logits, dim=-1)
            optimized_value = model.nnue_experts(
                optimized_weights, us, them, white, black
            )
            optimized_losses = per_record_loss(
                optimized_value,
                score,
                outcome,
                float(hyperparameters.get("score_scaling", 361.0)),
                float(hyperparameters.get("lambda_", 1.0)),
                float(hyperparameters.get("label_smoothing_eps", 0.0)),
            )
            entropy = -(
                optimized_weights * (optimized_weights + 1e-12).log()
            ).sum(dim=-1)

        take = min(int(expert_losses.shape[0]), args.max_positions - processed)
        output[processed : processed + take, :n_experts] = (
            expert_losses[:take].cpu().numpy()
        )
        output[processed : processed + take, n_experts:] = (
            optimized_weights[:take].cpu().numpy()
        )
        initial_loss_sum += float(initial_losses[:take].sum().cpu())
        optimized_loss_sum += float(optimized_losses[:take].sum().cpu())
        entropy_sum += float(entropy[:take].sum().cpu())
        processed += take
        if processed % 10000 < take:
            print(f"Progress: {processed}/{args.max_positions}", flush=True)

    output.flush()
    del output
    os.replace(temporary_path, output_path)
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "bin_dir": os.path.abspath(args.bin_dir),
        "expert_loss_cache": os.path.abspath(args.expert_loss_cache),
        "positions": args.max_positions,
        "n_experts": n_experts,
        "steps": args.steps,
        "lr": args.lr,
        "prior_kl": args.prior_kl,
        "initial_mean_loss": initial_loss_sum / processed,
        "optimized_mean_loss": optimized_loss_sum / processed,
        "optimized_mean_entropy": entropy_sum / processed,
    }
    with output_path.with_suffix(".json").open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
