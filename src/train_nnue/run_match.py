"""
対局スクリプト: USIエンジン同士の対局を自動実行する。

cshogiのEngineクラスを使い、先後入れ替えて指定回数対局し、
勝敗・Elo計算を出力する。

Use scripts/eval_match.sh as the canonical entry point.
"""

import argparse
import json
import math
import os
import sys
import time

import cshogi
from cshogi.usi import Engine

from train_nnue.eval_accuracy import resolve_engine_option_paths, resolve_path


def play_game(engine1, engine2, go_params, start_sfen=cshogi.STARTING_SFEN, max_moves=512):
    """1局対局する。

    Args:
        engine1: 先手エンジン
        engine2: 後手エンジン
        go_params: Engine.goへ渡す固定探索条件
        start_sfen: 開始局面
        max_moves: 最大手数

    Returns:
        result: 1=先手勝ち, 0=引き分け, -1=後手勝ち
        moves: 手数
    """
    board = cshogi.Board(start_sfen)
    engines = [engine1, engine2]

    for engine in engines:
        engine.position(sfen=f"sfen {start_sfen}")

    move_count = 0
    while move_count < max_moves:
        turn = board.turn  # 0=BLACK(先手), 1=WHITE(後手)
        engine = engines[turn]

        # Set position
        if move_count == 0:
            engine.position(sfen=f"sfen {board.sfen()}")
        else:
            engine.position(sfen=f"sfen {board.sfen()}")

        # Go
        bestmove, _ = engine.go(**go_params)

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


def parse_options(opt_str):
    if not opt_str:
        return {}
    opts = {}
    for kv in opt_str.split(','):
        if '=' in kv:
            key, value = kv.split('=', 1)
            opts[key.strip()] = value.strip()
    return opts


def load_openings(path):
    if path is None:
        return [{"sfen": cshogi.STARTING_SFEN}]
    openings = []
    with open(path) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "sfen" not in record:
                raise ValueError(f"opening line {line_number} has no sfen")
            openings.append(record)
    if not openings:
        raise ValueError("opening dataset is empty")
    return openings


def finite_or_none(value):
    return value if math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description="USI engine match")
    parser.add_argument("--engine1", required=True, help="Engine 1 path")
    parser.add_argument("--engine1-options", default="",
                        help="Engine 1 options (key=value, comma-separated)")
    parser.add_argument("--engine2", required=True, help="Engine 2 path")
    parser.add_argument("--engine2-options", default="",
                        help="Engine 2 options (key=value, comma-separated)")
    parser.add_argument(
        "--games",
        type=int,
        help="Number of games (default: twice the opening count, otherwise 100)",
    )
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--nodes", type=int, help="Fixed nodes per move")
    search.add_argument("--byoyomi", type=int, help="Byoyomi in ms")
    parser.add_argument(
        "--openings",
        help="JSONL start positions; each is played twice with colors reversed",
    )
    parser.add_argument("--max-moves", type=int, default=512, help="Max moves per game")
    parser.add_argument("--output", help="Write aggregate and per-game results as JSON")
    args = parser.parse_args()

    engine1_opts = parse_options(args.engine1_options)
    engine2_opts = parse_options(args.engine2_options)
    go_params = {"nodes": args.nodes or 1_000_000}
    if args.byoyomi is not None:
        go_params = {"byoyomi": args.byoyomi}
    if next(iter(go_params.values())) <= 0:
        parser.error("search limit must be positive")
    openings = load_openings(args.openings)
    games = args.games
    if games is None:
        games = 2 * len(openings) if args.openings else 100
    if games <= 0 or games % 2:
        parser.error("--games must be a positive even number for color pairing")
    if args.openings and games > 2 * len(openings):
        parser.error("--games cannot exceed twice the number of opening positions")

    project_root = os.getcwd()
    engine1_path = resolve_path(args.engine1, project_root)
    engine2_path = resolve_path(args.engine2, project_root)
    engine1_opts = resolve_engine_option_paths(engine1_opts, project_root)
    engine2_opts = resolve_engine_option_paths(engine2_opts, project_root)

    print(f"Engine 1: {engine1_path}")
    print(f"Engine 1 options: {engine1_opts}")
    print(f"Engine 2: {engine2_path}")
    print(f"Engine 2 options: {engine2_opts}")
    print(f"Games: {games}, Search: {go_params}")
    print(f"Openings: {len(openings)}")
    print()

    # Initialize engines
    engine1 = Engine(engine1_path)
    engine2 = Engine(engine2_path)

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
    game_details = []

    for i in range(games):
        opening_index = (i // 2) % len(openings)
        start_sfen = openings[opening_index]["sfen"]
        if i % 2 == 0:
            # Engine1 = BLACK (先手), Engine2 = WHITE (後手)
            result, moves = play_game(
                engine1, engine2, go_params, start_sfen, args.max_moves
            )
            if result > 0:
                e1_wins += 1
                outcome = "E1 win (B)"
            elif result < 0:
                e1_losses += 1
                outcome = "E2 win (W)"
            else:
                draws += 1
                outcome = "Draw"
            engine1_color = "black"
        else:
            # Engine1 = WHITE (後手), Engine2 = BLACK (先手)
            result, moves = play_game(
                engine2, engine1, go_params, start_sfen, args.max_moves
            )
            if result > 0:
                e1_losses += 1
                outcome = "E2 win (B)"
            elif result < 0:
                e1_wins += 1
                outcome = "E1 win (W)"
            else:
                draws += 1
                outcome = "Draw"
            engine1_color = "white"

        engine1_result = "draw"
        if outcome.startswith("E1 win"):
            engine1_result = "win"
        elif outcome.startswith("E2 win"):
            engine1_result = "loss"
        game_details.append(
            {
                "game": i + 1,
                "opening_index": opening_index,
                "sfen": start_sfen,
                "engine1_color": engine1_color,
                "engine1_result": engine1_result,
                "moves": moves,
            }
        )

        elapsed = time.time() - start_time
        elo, elo_se = elo_diff(e1_wins, e1_losses, draws)
        print(f"[{i+1}/{games}] {outcome} ({moves} moves) | "
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

    if args.output:
        elo_radius = 1.959963984540054 * elo_se
        output = {
            "engine1": {"path": engine1_path, "options": engine1_opts},
            "engine2": {"path": engine2_path, "options": engine2_opts},
            "search": go_params,
            "openings_path": args.openings,
            "games": total,
            "wins": e1_wins,
            "losses": e1_losses,
            "draws": draws,
            "score_rate": (e1_wins + 0.5 * draws) / total if total else None,
            "elo_difference": finite_or_none(elo),
            "elo_standard_error": elo_se,
            "elo_95": {
                "lower": finite_or_none(elo - elo_radius),
                "upper": finite_or_none(elo + elo_radius),
            },
            "details": game_details,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print(f"Results written to {args.output}")

    engine1.quit()
    engine2.quit()


if __name__ == "__main__":
    main()
