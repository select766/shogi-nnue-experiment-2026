"""
Verify AobaZero PyTorch model against game records.

Evaluates:
1. Move agreement: Does the model's top policy move match the game record?
2. Outcome accuracy: Does the model's value prediction agree with the game result?

Acceptance criteria (Step 1-1):
- Move agreement > 30%
- Outcome accuracy > 70%

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.verify_aobazero \
        --weights ../aobazero/weights/aobazero_w4636.pt \
        --dataset ../data/accuracy_eval/test.jsonl \
        --num-positions 200
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import cshogi

from train_nnue.aobazero_model import AobaZeroNet
from train_nnue.aobazero_features import encode_features


def move_to_policy_index(move: int, turn: int) -> int:
    """Convert a cshogi move to a dlshogi-style policy index (0-2186).

    The policy output is (27, 9, 9) = 2187, viewed from the current player's
    perspective. For gote, the board is flipped 180 degrees.

    Policy layout (27 channels):
      ch  0- 9: non-promote, directions 0-9
      ch 10-19: promote, directions 0-9
      ch 20-26: drops, piece types 0-6 (pawn..rook)

    Directions (dx = from_x - to_x, dy = from_y - to_y, in player's coords):
      0: dx>0, dy==0  (right-to-left)
      1: dx>0, dy>0   (diagonal)
      2: dx==0, dy>0  (up-to-down)
      3: dx<0, dy>0   (diagonal)
      4: dx<0, dy==0  (left-to-right)
      5: dx<0, dy<0   (diagonal)
      6: dx==0, dy<0  (down-to-up)
      7: dx>0, dy<0   (diagonal)
      8: knight (dy=-2, dx=-1)
      9: knight (dy=-2, dx=+1)

    (y, x) in the policy = destination square in the player's coordinate system.

    Reference: aobazero/repo/src/usi-engine/bona/yss_net.cpp get_dlshogi_policy_id()
    """
    is_drop = cshogi.move_is_drop(move)
    to_sq = cshogi.move_to(move)
    flip = (turn == cshogi.WHITE)

    # Convert to AobaZero (rank, file) coordinates
    # cshogi: sq = file_index * 9 + rank_index
    to_rank = to_sq % 9
    to_file = to_sq // 9
    if flip:
        to_rank = 8 - to_rank
        to_file = 8 - to_file

    if is_drop:
        # Drop piece type: pawn=1, lance=2, ..., rook=7 in cshogi hand piece
        drop_piece = cshogi.move_drop_hand_piece(move)
        # Map to 0-6: pawn=0, lance=1, knight=2, silver=3, gold=4, bishop=5, rook=6
        dir_idx = drop_piece - 1
        ch = 20 + dir_idx
    else:
        from_sq = cshogi.move_from(move)
        is_promote = cshogi.move_is_promotion(move)

        from_rank = from_sq % 9
        from_file = from_sq // 9
        if flip:
            from_rank = 8 - from_rank
            from_file = 8 - from_file

        # Direction: d_file = from_file - to_file, d_rank = from_rank - to_rank
        # Maps to AobaZero's dx (=d_file), dy (=d_rank) in get_dlshogi_policy_id
        # YSS z = (rank+1)*16 + (file+1), so az_x = file, az_y = rank
        # dx = bz_x - az_x = from_file - to_file = d_file
        # dy = bz_y - az_y = from_rank - to_rank = d_rank
        d_file = from_file - to_file
        d_rank = from_rank - to_rank

        # Determine direction
        # Knight: az == bz - 0x21 → d_rank=+2, d_file=+1 → dir=8
        #         az == bz - 0x1f → d_rank=+2, d_file=-1 → dir=9
        if d_file == 1 and d_rank == 2:
            dir_idx = 8  # knight (sente: forward-left)
        elif d_file == -1 and d_rank == 2:
            dir_idx = 9  # knight (sente: forward-right)
        elif d_file > 0 and d_rank == 0:
            dir_idx = 0
        elif d_file > 0 and d_rank > 0:
            dir_idx = 1
        elif d_file == 0 and d_rank > 0:
            dir_idx = 2
        elif d_file < 0 and d_rank > 0:
            dir_idx = 3
        elif d_file < 0 and d_rank == 0:
            dir_idx = 4
        elif d_file < 0 and d_rank < 0:
            dir_idx = 5
        elif d_file == 0 and d_rank < 0:
            dir_idx = 6
        elif d_file > 0 and d_rank < 0:
            dir_idx = 7
        else:
            raise ValueError(f"Invalid move: d_file={d_file}, d_rank={d_rank}, move={move}")

        ch = dir_idx + (10 if is_promote else 0)

    # Policy index: channel-first flattening of (27, 9, 9)
    # AobaZero: z81 = (rank) * 9 + (file), same as tensor[ch][rank][file]
    return ch * 81 + to_rank * 9 + to_file


def get_best_legal_move(policy_logits: np.ndarray, board: cshogi.Board) -> tuple:
    """Find the legal move with the highest policy logit.

    Returns:
        (best_move_usi, best_logit, best_index)
    """
    turn = board.turn
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None, -float('inf'), -1

    best_move = None
    best_logit = -float('inf')
    best_idx = -1

    for move in legal_moves:
        try:
            idx = move_to_policy_index(move, turn)
            if idx < 0 or idx >= 2187:
                continue
            logit = policy_logits[idx]
            if logit > best_logit:
                best_logit = logit
                best_move = move
                best_idx = idx
        except (ValueError, IndexError):
            continue

    if best_move is not None:
        return cshogi.move_to_usi(best_move), best_logit, best_idx
    return None, -float('inf'), -1


def evaluate_dataset(model: AobaZeroNet, dataset: list, device: torch.device,
                     num_positions: int = 0, verbose: bool = False):
    """Evaluate model on a dataset of positions.

    Args:
        model: AobaZeroNet model
        dataset: List of dicts with keys: sfen, bestmove, turn, gameResult
        device: torch device
        num_positions: Max positions to evaluate (0 = all)
        verbose: Print per-position details
    """
    model.eval()

    if num_positions > 0:
        dataset = dataset[:num_positions]

    total = len(dataset)
    move_matches = 0
    outcome_correct = 0
    outcome_total = 0

    board = cshogi.Board()
    start_time = time.time()

    for i, record in enumerate(dataset):
        sfen = record["sfen"]
        expected_move = record["bestmove"]
        turn = record.get("turn", 0)
        game_result = record.get("gameResult", 0)

        board.set_sfen(sfen)
        ply = record.get("ply", 0)

        # Encode features
        features = encode_features(board, ply=ply)
        x = torch.from_numpy(features).unsqueeze(0).to(device)

        # Run model
        with torch.no_grad():
            policy_logits, value = model(x)

        policy_np = policy_logits[0].cpu().numpy()
        value_scalar = value[0, 0].item()

        # Move agreement
        predicted_move, _, _ = get_best_legal_move(policy_np, board)
        is_match = (predicted_move == expected_move)
        if is_match:
            move_matches += 1

        # Outcome accuracy
        # AobaZero value: +1 = current player wins
        # gameResult: 1 = sente(black) wins, 2 = gote(white) wins
        if game_result in (1, 2):
            outcome_total += 1
            current_wins = (game_result == 1 and turn == 0) or \
                           (game_result == 2 and turn == 1)
            predicted_current_wins = value_scalar > 0
            if current_wins == predicted_current_wins:
                outcome_correct += 1

        if verbose and i < 20:
            print(f"  [{i}] expected={expected_move} predicted={predicted_move} "
                  f"match={is_match} value={value_scalar:.3f} result={game_result}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{total} ({rate:.1f} pos/s)", file=sys.stderr)

    elapsed = time.time() - start_time
    move_accuracy = move_matches / total if total > 0 else 0
    outcome_accuracy = outcome_correct / outcome_total if outcome_total > 0 else 0

    print(f"\n=== Results ===")
    print(f"Positions evaluated: {total}")
    print(f"Time: {elapsed:.1f}s ({total/elapsed:.1f} pos/s)")
    print(f"Move agreement: {move_matches}/{total} = {move_accuracy:.1%}")
    print(f"Outcome accuracy: {outcome_correct}/{outcome_total} = {outcome_accuracy:.1%}")
    print(f"\nAcceptance criteria:")
    print(f"  Move agreement > 30%: {'PASS' if move_accuracy > 0.30 else 'FAIL'} ({move_accuracy:.1%})")
    print(f"  Outcome accuracy > 70%: {'PASS' if outcome_accuracy > 0.70 else 'FAIL'} ({outcome_accuracy:.1%})")

    return {
        "total": total,
        "move_matches": move_matches,
        "move_accuracy": move_accuracy,
        "outcome_correct": outcome_correct,
        "outcome_total": outcome_total,
        "outcome_accuracy": outcome_accuracy,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify AobaZero PyTorch model")
    parser.add_argument("--weights", required=True, help="Path to PyTorch weights (.pt)")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--num-positions", type=int, default=200,
                        help="Number of positions to evaluate (0=all)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--verbose", action="store_true", help="Print per-position details")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.weights}...", file=sys.stderr)
    model = AobaZeroNet()
    state_dict = torch.load(args.weights, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(args.device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)", file=sys.stderr)

    # Load dataset
    print(f"Loading dataset from {args.dataset}...", file=sys.stderr)
    with open(args.dataset) as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(dataset)} positions", file=sys.stderr)

    # Evaluate
    device = torch.device(args.device)
    results = evaluate_dataset(model, dataset, device,
                               num_positions=args.num_positions,
                               verbose=args.verbose)

    # Save results
    output_path = args.weights.replace('.pt', '_verification.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
