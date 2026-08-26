"""Compare assigned-phase and average single-expert loss on paired roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_nnue.compare_root_grouped_checkpoints import load_model
from train_nnue.diagnose_gate import per_record_loss
from train_nnue.evaluate_root_grouped_teachers import bootstrap_interval
from train_nnue.root_grouped_dataset import RootGroupedDataset


def evaluate(path, args):
    model, hparams, auxiliary_dim = load_model(
        path, args.backbone_weights, args.nnue_checkpoint, args.device
    )
    if auxiliary_dim:
        raise ValueError("phase specialization comparison does not accept auxiliary adapters")
    dataset = RootGroupedDataset(
        args.data,
        "HalfKP",
        args.root_batch_size,
        device=args.device,
        expert_role_cache_path=args.role_cache,
    )
    assigned_losses = []
    average_losses = []
    roles_output = []
    processed = 0
    iterator = iter(dataset)
    with torch.inference_mode():
        while processed < args.max_roots:
            batch = next(iterator)
            x1, x2, us, them, white, black, outcome, score, _ = batch[:9]
            roles = batch[9]
            roots = int(x1.shape[0])
            take = min(roots, args.max_roots - processed)
            expert_values = torch.cat(
                [
                    model.nnue_experts.forward_expert(
                        index, us, them, white, black
                    )
                    for index in range(model.nnue_experts.n_experts)
                ],
                dim=1,
            )
            expert_losses = torch.stack(
                [
                    per_record_loss(
                        expert_values[:, index : index + 1],
                        score,
                        outcome,
                        float(hparams.get("score_scaling", 361.0)),
                        float(hparams.get("lambda_", 1.0)),
                        float(hparams.get("label_smoothing_eps", 0.0)),
                    )
                    for index in range(model.nnue_experts.n_experts)
                ],
                dim=1,
            ).reshape(roots, dataset.group_size, -1).mean(dim=1)
            row_indices = torch.arange(roots, device=args.device)
            assigned_losses.append(
                expert_losses[row_indices, roles][:take].cpu().numpy()
            )
            average_losses.append(expert_losses.mean(dim=1)[:take].cpu().numpy())
            roles_output.append(roles[:take].cpu().numpy())
            processed += take
    del model
    torch.cuda.empty_cache()
    return {
        "assigned": np.concatenate(assigned_losses),
        "average": np.concatenate(average_losses),
        "roles": np.concatenate(roles_output),
    }


def summary(values, roles):
    return {
        "assigned_phase_expert_loss": float(values["assigned"].mean()),
        "average_single_expert_loss": float(values["average"].mean()),
        "assigned_loss_by_role": {
            str(role): float(values["assigned"][roles == role].mean())
            for role in range(8)
            if np.any(roles == role)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument(
        "--reference",
        help="Optional initial model used for the average single-expert quality delta",
    )
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--role-cache", required=True)
    parser.add_argument("--backbone-weights", required=True)
    parser.add_argument("--nnue-checkpoint", required=True)
    parser.add_argument("--max-roots", type=int, required=True)
    parser.add_argument("--root-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("phase specialization comparison requires CUDA")
    control = evaluate(args.control, args)
    reference = evaluate(args.reference, args) if args.reference else control
    if not np.array_equal(reference["roles"], control["roles"]):
        raise RuntimeError("role order changed between control and reference evaluations")
    output = {
        "format": "phase-specialization-comparison-v1",
        "roots": args.max_roots,
        "control": {"checkpoint": str(Path(args.control).resolve()), **summary(control, control["roles"])},
        "reference": {
            "checkpoint": str(Path(args.reference or args.control).resolve()),
            **summary(reference, reference["roles"]),
        },
        "candidates": [],
    }
    for path in args.candidate:
        candidate = evaluate(path, args)
        if not np.array_equal(candidate["roles"], control["roles"]):
            raise RuntimeError("role order changed between model evaluations")
        assigned_delta = candidate["assigned"] - control["assigned"]
        average_delta = candidate["average"] - control["average"]
        average_reference_delta = candidate["average"] - reference["average"]
        output["candidates"].append(
            {
                "checkpoint": str(Path(path).resolve()),
                **summary(candidate, candidate["roles"]),
                "assigned_phase_loss_delta": {
                    "mean": float(assigned_delta.mean()),
                    "bootstrap_95_interval": bootstrap_interval(assigned_delta, args.bootstrap_seed),
                },
                "average_single_expert_loss_delta": {
                    "mean": float(average_delta.mean()),
                    "bootstrap_95_interval": bootstrap_interval(average_delta, args.bootstrap_seed),
                },
                "average_single_expert_loss_delta_vs_reference": {
                    "mean": float(average_reference_delta.mean()),
                    "bootstrap_95_interval": bootstrap_interval(
                        average_reference_delta, args.bootstrap_seed
                    ),
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
