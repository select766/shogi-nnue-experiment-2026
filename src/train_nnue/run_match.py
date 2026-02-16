"""
対局スクリプト: USIエンジン同士の対局を自動実行する。

cshogiのEngineクラスを使い、先後入れ替えて指定回数対局し、
勝敗・Elo計算を出力する。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.run_match \
        --engine1 "../bin/YaneuraOu-expert-blending" \
        --engine1-options "DNNServerCmd=python -m train_nnue.dnn_inference_server ..." \
        --engine2 "../bin/YaneuraOu-by-gcc" \
        --games 100 \
        --byoyomi 3000
"""

import argparse
import math
import sys
import time

import cshogi
from cshogi.usi import Engine


def play_game(engine1, engine2, byoyomi=3000, max_moves=512):
    """1局対局する。

    Args:
        engine1: 先手エンジン
        engine2: 後手エンジン
        byoyomi: 秒読み (ミリ秒)
        max_moves: 最大手数

    Returns:
        result: 1=先手勝ち, 0=引き分け, -1=後手勝ち
        moves: 手数
    """
    board = cshogi.Board()
    engines = [engine1, engine2]

    for engine in engines:
        engine.position(sfen=cshogi.STARTING_SFEN)

    move_count = 0
    while move_count < max_moves:
        turn = board.turn  # 0=BLACK(先手), 1=WHITE(後手)
        engine = engines[turn]

        # Set position
        if move_count == 0:
            engine.position(sfen=board.sfen())
        else:
            engine.position(sfen=board.sfen())

        # Go
        bestmove, _ = engine.go(byoyomi=byoyomi)

        if bestmove is None or bestmove == 'resign':
            # Current player resigns -> opponent wins
            return -1 if turn == 0 else 1, move_count
        if bestmove == 'win':
            # Declare win
            return 1 if turn == 0 else -1, move_count

        # Apply move
        move = board.move_from_usi(bestmove)
        if move is None or move == 0:
            # Invalid move -> current player loses
            return -1 if turn == 0 else 1, move_count

        board.push(move)
        move_count += 1

        # Check game end
        if board.is_game_over():
            # Current side has no legal moves = loses
            return -1 if board.turn == 0 else 1, move_count

        # Repetition check
        rep = board.is_draw()
        if rep == cshogi.REPETITION_DRAW:
            return 0, move_count
        elif rep == cshogi.REPETITION_WIN:
            return 1 if board.turn == 0 else -1, move_count
        elif rep == cshogi.REPETITION_LOSE:
            return -1 if board.turn == 0 else 1, move_count

    # Max moves reached -> draw
    return 0, move_count


def elo_diff(wins, losses, draws):
    """勝敗からEloレーティング差を推定する。"""
    total = wins + losses + draws
    if total == 0:
        return 0.0, 0.0
    score = (wins + draws * 0.5) / total
    if score <= 0 or score >= 1:
        return float('inf') if score >= 1 else float('-inf'), 0.0

    elo = -400.0 * math.log10(1.0 / score - 1.0)

    # Standard error estimation
    w = wins / total
    l = losses / total
    d = draws / total
    var = (w * (1 - score) ** 2 + l * score ** 2 + d * (0.5 - score) ** 2) / total
    if var <= 0:
        return elo, 0.0
    se = math.sqrt(var)
    elo_se = 400.0 * se / (math.log(10) * score * (1 - score))
    return elo, elo_se


def main():
    parser = argparse.ArgumentParser(description="USI engine match")
    parser.add_argument("--engine1", required=True, help="Engine 1 path")
    parser.add_argument("--engine1-options", default="",
                        help="Engine 1 options (key=value, comma-separated)")
    parser.add_argument("--engine2", required=True, help="Engine 2 path")
    parser.add_argument("--engine2-options", default="",
                        help="Engine 2 options (key=value, comma-separated)")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--byoyomi", type=int, default=3000, help="Byoyomi in ms")
    parser.add_argument("--max-moves", type=int, default=512, help="Max moves per game")
    args = parser.parse_args()

    def parse_options(opt_str):
        if not opt_str:
            return {}
        opts = {}
        for kv in opt_str.split(','):
            if '=' in kv:
                k, v = kv.split('=', 1)
                opts[k.strip()] = v.strip()
        return opts

    engine1_opts = parse_options(args.engine1_options)
    engine2_opts = parse_options(args.engine2_options)

    print(f"Engine 1: {args.engine1}")
    print(f"Engine 1 options: {engine1_opts}")
    print(f"Engine 2: {args.engine2}")
    print(f"Engine 2 options: {engine2_opts}")
    print(f"Games: {args.games}, Byoyomi: {args.byoyomi}ms")
    print()

    # Initialize engines
    engine1 = Engine(args.engine1)
    engine2 = Engine(args.engine2)

    for k, v in engine1_opts.items():
        engine1.setoption(k, v)
    for k, v in engine2_opts.items():
        engine2.setoption(k, v)

    engine1.isready()
    engine2.isready()

    # Play games (alternate colors)
    e1_wins = 0
    e1_losses = 0
    draws = 0
    start_time = time.time()

    for i in range(args.games):
        if i % 2 == 0:
            # Engine1 = BLACK (先手), Engine2 = WHITE (後手)
            result, moves = play_game(engine1, engine2, args.byoyomi, args.max_moves)
            if result > 0:
                e1_wins += 1
                outcome = "E1 win (B)"
            elif result < 0:
                e1_losses += 1
                outcome = "E2 win (W)"
            else:
                draws += 1
                outcome = "Draw"
        else:
            # Engine1 = WHITE (後手), Engine2 = BLACK (先手)
            result, moves = play_game(engine2, engine1, args.byoyomi, args.max_moves)
            if result > 0:
                e1_losses += 1
                outcome = "E2 win (B)"
            elif result < 0:
                e1_wins += 1
                outcome = "E1 win (W)"
            else:
                draws += 1
                outcome = "Draw"

        elapsed = time.time() - start_time
        elo, elo_se = elo_diff(e1_wins, e1_losses, draws)
        print(f"[{i+1}/{args.games}] {outcome} ({moves} moves) | "
              f"E1: {e1_wins}W {e1_losses}L {draws}D | "
              f"Elo: {elo:+.1f} ±{elo_se:.1f} | "
              f"{elapsed:.0f}s", flush=True)

    # Final results
    total = e1_wins + e1_losses + draws
    elo, elo_se = elo_diff(e1_wins, e1_losses, draws)
    print(f"\n=== Final Results ===")
    print(f"Engine 1: {args.engine1}")
    print(f"Engine 2: {args.engine2}")
    print(f"Games: {total}")
    print(f"Engine 1: {e1_wins}W {e1_losses}L {draws}D "
          f"({e1_wins/total*100:.1f}% / {e1_losses/total*100:.1f}% / {draws/total*100:.1f}%)")
    print(f"Elo difference: {elo:+.1f} ±{elo_se:.1f}")

    engine1.quit()
    engine2.quit()


if __name__ == "__main__":
    main()
