"""
Verify dlshogi PyTorch model against game records.

Evaluates:
1. Move agreement: Does the model's top policy move match the game record?
2. Outcome accuracy: Does the model's value prediction agree with the game result?

Acceptance criteria:
- Move agreement > 30%
- Outcome accuracy > 70%

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.verify_dlshogi \
        --weights ../tmp/dlshogi-model/model_resnet10_swish-072 \
        --dataset ../data/accuracy_eval/test.jsonl \
        --num-positions 200 --verbose
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import cshogi

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.network.policy_value_network_resnet10_swish import PolicyValueNetwork
from dlshogi import serializers
import dlshogi.cppshogi as dcppshogi


def encode_position(board: cshogi.Board):
    """Encode a board position into dlshogi features using cppshogi.

    Uses HCPE round-trip: board -> HCP -> hcpe_decode_with_value.

    Returns:
        features1: (1, FEATURES1_NUM, 9, 9) float32
        features2: (1, FEATURES2_NUM, 9, 9) float32
    """
    hcp = np.zeros(32, dtype=np.uint8)
    board.to_hcp(hcp)

    # Construct HCPE struct (38 bytes)
    hcpe_dtype = np.dtype([
        ('hcp', np.uint8, 32),
        ('eval', np.int16),
        ('bestMove16', np.uint16),
        ('gameResult', np.uint8),
        ('padding', np.uint8),
    ])
    hcpe = np.zeros(1, dtype=hcpe_dtype)
    hcpe[0]['hcp'] = hcp

    features1 = np.zeros((1, FEATURES1_NUM, 9, 9), dtype=np.float32)
    features2 = np.zeros((1, FEATURES2_NUM, 9, 9), dtype=np.float32)
    move = np.zeros(1, dtype=np.int64)
    result = np.zeros(1, dtype=np.float32)
    value = np.zeros(1, dtype=np.float32)

    dcppshogi.hcpe_decode_with_value(
        hcpe.view(np.uint8).reshape(1, -1),
        features1, features2, move, result, value
    )
    return features1, features2


def make_move_label(move: int, turn: int) -> int:
    """Convert a cshogi move to a dlshogi policy index (0-2186).

    Matches dlshogi C++ make_move_label() in cppshogi/cppshogi.cpp.
    Policy: 27 * 81 = 2187 flat index.
    Index = 81 * move_direction + to_sq (file-major: sq = file*9 + rank).

    Directions (dir_x = from_file - to_file, dir_y = to_rank - from_rank):
      0: UP         (dy<0, dx==0)
      1: UP_LEFT    (dy<0, dx<0)
      2: UP_RIGHT   (dy<0, dx>0)
      3: LEFT       (dy==0, dx<0)
      4: RIGHT      (dy==0, dx>0)
      5: DOWN       (dy>0, dx==0)
      6: DOWN_LEFT  (dy>0, dx<0)
      7: DOWN_RIGHT (dy>0, dx>0)
      8: UP2_LEFT   (dy==-2, dx==-1)
      9: UP2_RIGHT  (dy==-2, dx==1)
      10-19: promote versions of 0-9
      20-26: drops (pawn..rook)
    """
    is_drop = cshogi.move_is_drop(move)
    to_sq = cshogi.move_to(move)

    if is_drop:
        if turn == cshogi.WHITE:
            to_sq = 80 - to_sq
        drop_piece = cshogi.move_drop_hand_piece(move)
        # drop_piece: 0-based (HPAWN=0..HROOK=6)
        direction = 20 + drop_piece
        return 81 * direction + to_sq

    from_sq = cshogi.move_from(move)
    is_promote = cshogi.move_is_promotion(move)

    if turn == cshogi.WHITE:
        to_sq = 80 - to_sq
        from_sq = 80 - from_sq

    to_file = to_sq // 9
    to_rank = to_sq % 9
    from_file = from_sq // 9
    from_rank = from_sq % 9

    dir_x = from_file - to_file
    dir_y = to_rank - from_rank

    if dir_y < 0 and dir_x == 0:
        direction = 0  # UP
    elif dir_y == -2 and dir_x == -1:
        direction = 8  # UP2_LEFT
    elif dir_y == -2 and dir_x == 1:
        direction = 9  # UP2_RIGHT
    elif dir_y < 0 and dir_x < 0:
        direction = 1  # UP_LEFT
    elif dir_y < 0 and dir_x > 0:
        direction = 2  # UP_RIGHT
    elif dir_y == 0 and dir_x < 0:
        direction = 3  # LEFT
    elif dir_y == 0 and dir_x > 0:
        direction = 4  # RIGHT
    elif dir_y > 0 and dir_x == 0:
        direction = 5  # DOWN
    elif dir_y > 0 and dir_x < 0:
        direction = 6  # DOWN_LEFT
    else:
        direction = 7  # DOWN_RIGHT

    if is_promote:
        direction += 10

    return 81 * direction + to_sq


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
            idx = make_move_label(move, turn)
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


def evaluate_dataset(model: PolicyValueNetwork, dataset: list, device: torch.device,
                     num_positions: int = 0, verbose: bool = False):
    """Evaluate model on a dataset of positions."""
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

        # Encode features via cppshogi
        features1, features2 = encode_position(board)
        x1 = torch.from_numpy(features1).to(device)
        x2 = torch.from_numpy(features2).to(device)

        # Run model
        with torch.no_grad():
            policy_logits, value_logits = model(x1, x2)

        policy_np = policy_logits[0].cpu().numpy()
        # dlshogi value: sigmoid output, 0.5 = even, >0.5 = current player advantage
        value_scalar = torch.sigmoid(value_logits[0, 0]).item()

        # Move agreement
        predicted_move, _, _ = get_best_legal_move(policy_np, board)
        is_match = (predicted_move == expected_move)
        if is_match:
            move_matches += 1

        # Outcome accuracy
        # gameResult: 1 = sente(black) wins, 2 = gote(white) wins
        if game_result in (1, 2):
            outcome_total += 1
            current_wins = (game_result == 1 and turn == 0) or \
                           (game_result == 2 and turn == 1)
            predicted_current_wins = value_scalar > 0.5
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
    parser = argparse.ArgumentParser(description="Verify dlshogi PyTorch model")
    parser.add_argument("--weights", required=True, help="Path to dlshogi NPZ weights")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--num-positions", type=int, default=200,
                        help="Number of positions to evaluate (0=all)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--verbose", action="store_true", help="Print per-position details")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.weights}...", file=sys.stderr)
    model = PolicyValueNetwork()
    serializers.load_npz(args.weights, model)
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
    output_path = args.weights + '_verification.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
