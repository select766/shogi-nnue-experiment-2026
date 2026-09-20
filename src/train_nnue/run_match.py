"""
対局スクリプト: USIエンジン同士の対局を自動実行する。

cshogiのEngineクラスを使い、先後入れ替えて指定回数対局し、
勝敗・Elo計算を出力する。

Use scripts/eval_match.sh as the canonical entry point.
"""

from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
import json
import math
import os
import sys
import time

import cshogi
from cshogi.usi import Engine

try:
    from scripts.collect_search_leaf_groups import EngineProcess
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    # merge_match_results.py imports this module while its own scripts/
    # directory, rather than the repository root, is first on sys.path.
    from collect_search_leaf_groups import EngineProcess
from train_nnue.eval_accuracy import resolve_engine_option_paths, resolve_path
from train_nnue.root_search_statistics import shallow_multipv_features


class RootStatisticsEngine:
    """Candidate engine whose per-move gate receives a fixed shallow search."""

    def __init__(self, candidate, shallow, nodes=1024, multipv=4):
        self.candidate = candidate
        self.shallow = shallow
        self.nodes = nodes
        self.multipv = multipv
        self.sfen = None
        self.shallow_seconds = 0.0
        self.main_seconds = 0.0
        self.searches = 0

    def setoption(self, name, value):
        self.candidate.setoption(name, value)

    def isready(self):
        self.candidate.isready()
        self.shallow.send("isready")
        self.shallow.read_until(lambda line: line == "readyok")

    def usinewgame(self):
        self.candidate.usinewgame()
        self.shallow.send("usinewgame")

    def position(self, *, sfen, moves=None):
        prefix = "sfen "
        if not sfen.startswith(prefix):
            raise ValueError("RootStatisticsEngine requires an explicit SFEN")
        board = cshogi.Board(sfen[len(prefix):])
        for token in moves or []:
            move = board.move_from_usi(token)
            if not move or not board.is_legal(move):
                raise ValueError(f"illegal history move: {token}")
            board.push(move)
        self.sfen = board.sfen()
        self.history_sfen = sfen[len(prefix):] + (" moves " + " ".join(moves) if moves else "")
        self.candidate.position(sfen=sfen, moves=moves)

    def go(self, **go_params):
        if self.sfen is None:
            raise RuntimeError("position must be set before go")
        started = time.monotonic()
        features = shallow_multipv_features(
            self.shallow, self.history_sfen, nodes=self.nodes, multipv=self.multipv
        )
        self.shallow_seconds += time.monotonic() - started
        encoded = ",".join(f"{float(value):.8g}" for value in features)
        self.candidate.setoption("ExpertBlendingRootSearchStatistics", encoded)
        started = time.monotonic()
        result = self.candidate.go(**go_params)
        self.main_seconds += time.monotonic() - started
        self.searches += 1
        return result

    def quit(self):
        self.shallow.close()
        self.candidate.quit()

    def timing(self):
        return {
            "searches": self.searches,
            "shallow_seconds": self.shallow_seconds,
            "main_seconds": self.main_seconds,
            "added_time_fraction": (
                self.shallow_seconds / self.main_seconds
                if self.main_seconds > 0 else None
            ),
        }


def play_game(
    engine1, engine2, go_params, start_sfen=cshogi.STARTING_SFEN,
    max_moves=512, clear_hash_each_move=False, *, history=True, details=None,
    initial_sfen=None, history_usi=None, stop_event=None,
):
    """Play from an opening; return black's result and number of legal plies.

    Each game resets both engines. Optional source history is legally replayed
    and included in adjudication, and in USI positions when history=True.
    max_moves counts only newly played plies. ``details`` receives an audit trail.
    Protocol/illegal-move faults are recorded and raised, never scored as wins.
    """
    prefix = list(history_usi or [])
    if prefix and initial_sfen is None:
        raise ValueError("history_usi requires initial_sfen")
    origin = initial_sfen if initial_sfen is not None else start_sfen
    board = cshogi.Board(origin)
    engines = [engine1, engine2]
    trace = details if details is not None else {}
    trace.update(start_sfen=start_sfen, history=history, moves_usi=[], searches=[])
    started = time.monotonic()
    position_key = lambda: " ".join(board.sfen().split()[:3])
    occurrences = {position_key(): [0]}
    checks = []
    for token in prefix:
        turn = board.turn
        move = board.move_from_usi(token)
        if not move or not board.is_legal(move):
            raise ValueError(f"illegal source history: {token}")
        board.push(move)
        checks.append((turn, board.is_check()))
        occurrences.setdefault(position_key(), []).append(len(checks))
    if board.sfen() != start_sfen:
        raise ValueError("source history does not reconstruct opening SFEN")
    trace.update(initial_sfen=origin, history_usi=prefix)

    def finish(result, reason):
        trace.update(result=result, termination=reason, final_sfen=board.sfen(),
                     elapsed_seconds=time.monotonic() - started)
        return result, len(trace["moves_usi"])

    for engine in engines:
        engine.isready()
        engine.usinewgame()
    while True:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("match cancelled")
        turn = board.turn
        if board.is_game_over():
            return finish(-1 if turn == 0 else 1, "no_legal_moves")
        repetitions = occurrences[position_key()]
        if len(repetitions) >= 4:
            cycle = checks[repetitions[-4]:]
            for checker in (0, 1):
                own_checks = [check for side, check in cycle if side == checker]
                if own_checks and all(own_checks):
                    return finish(-1 if checker == 0 else 1, "perpetual_check_black" if checker == 0 else "perpetual_check_white")
            return finish(0, "repetition_draw")
        if len(trace["moves_usi"]) >= max_moves:
            return finish(0, "max_moves")
        engine = engines[turn]
        search_started = time.monotonic()
        position = f"sfen {origin if history else board.sfen()}"
        moves = prefix + list(trace["moves_usi"]) if history else []
        engine.position(sfen=position, moves=moves)
        if clear_hash_each_move:
            engine.setoption("Clear Hash", "")
        infos = []
        bestmove, _ = engine.go(**go_params, listener=infos.append)
        nodes = None
        for line in infos:
            fields = line.split()
            if fields[:1] == ["info"] and "nodes" in fields:
                nodes = int(fields[fields.index("nodes") + 1])
        trace["searches"].append(dict(
            turn=turn, position=position + (" moves " + " ".join(moves) if moves else ""),
            bestmove=bestmove, nodes=nodes, info=infos,
            elapsed_seconds=time.monotonic() - search_started))
        if bestmove == "resign":
            return finish(-1 if turn == 0 else 1, "resign")
        if bestmove == "win" and board.is_nyugyoku():
            return finish(1 if turn == 0 else -1, "entering_king")
        try:
            move = board.move_from_usi(bestmove) if bestmove not in (None, "win") else 0
        except (ValueError, TypeError):
            move = 0
        if not move or not board.is_legal(move):
            finish(None, "protocol_error" if bestmove is None else "illegal_move")
            raise ValueError(f"invalid bestmove {bestmove!r} at {board.sfen()}")
        board.push(move)
        trace["moves_usi"].append(bestmove)
        checks.append((turn, board.is_check()))
        occurrences.setdefault(position_key(), []).append(len(checks))


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
    parser.add_argument("--clear-hash-each-move", action="store_true")
    parser.add_argument("--no-history", action="store_true", help="Legacy protocol ablation")
    parser.add_argument("--engine1-root-stats-engine")
    parser.add_argument("--engine1-root-stats-options", default="")
    parser.add_argument("--root-stats-nodes", type=int, default=1024)
    parser.add_argument("--root-stats-multipv", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None, help="Independent color-pair workers (default: 4 for nodes, 1 for byoyomi)")
    parser.add_argument("--output", help="Write aggregate and per-game results as JSON")
    args = parser.parse_args()

    engine1_opts = parse_options(args.engine1_options)
    engine2_opts = parse_options(args.engine2_options)
    go_params = {"nodes": args.nodes or 1_000_000}
    if args.byoyomi is not None:
        go_params = {"byoyomi": args.byoyomi}
    if next(iter(go_params.values())) <= 0:
        parser.error("search limit must be positive")
    workers = args.workers if args.workers is not None else (1 if args.byoyomi is not None else 4)
    if not 1 <= workers <= 16:
        parser.error("workers must be 1..16")
    if workers > 1 and args.byoyomi is not None:
        parser.error("time-controlled matches require --workers 1")
    if workers > 1:
        for opts in (engine1_opts, engine2_opts):
            if str(opts.get("Threads", "1")) != "1":
                parser.error("parallel matches require Threads=1")
            opts["Threads"] = "1"
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
    shallow_path = None
    shallow_opts = {}
    if args.engine1_root_stats_engine:
        shallow_path = resolve_path(args.engine1_root_stats_engine, project_root)
        shallow_opts = resolve_engine_option_paths(
            parse_options(args.engine1_root_stats_options), project_root
        )
        if args.root_stats_nodes <= 0 or not 1 <= args.root_stats_multipv <= 4:
            parser.error("invalid root-statistics search parameters")

    print(f"Engine 1: {engine1_path}")
    print(f"Engine 1 options: {engine1_opts}")
    print(f"Engine 2: {engine2_path}")
    print(f"Engine 2 options: {engine2_opts}")
    print(f"Games: {games}, Search: {go_params}")
    print(f"Openings: {len(openings)}")
    print(f"Clear hash each move: {args.clear_hash_each_move}")
    if shallow_path:
        print(
            f"Engine 1 root statistics: {shallow_path}, {shallow_opts}, "
            f"nodes={args.root_stats_nodes}, multipv={args.root_stats_multipv}"
        )
    print()

    stopped = threading.Event()

    def worker(indices):
        owned = []
        try:
            # Initialize engines
            engine1 = Engine(engine1_path)
            owned.append(engine1)
            engine2 = Engine(engine2_path)
            owned.append(engine2)

            for k, v in engine1_opts.items():
                engine1.setoption(k, v)
            for k, v in engine2_opts.items():
                engine2.setoption(k, v)

            engine1.isready()
            engine2.isready()
            root_statistics_config = None
            if shallow_path:
                shallow = EngineProcess([shallow_path])
                owned.append(shallow)
                options = list(shallow_opts.items())
                options.append(("MultiPV", args.root_stats_multipv))
                shallow.initialize(options)
                engine1 = RootStatisticsEngine(
                    engine1, shallow, nodes=args.root_stats_nodes,
                    multipv=args.root_stats_multipv,
                )
                root_statistics_config = {
                    "engine_path": shallow_path,
                    "engine_options": shallow_opts,
                    "nodes": args.root_stats_nodes,
                    "multipv": args.root_stats_multipv,
                }

            # Play games (alternate colors)
            e1_wins = 0
            e1_losses = 0
            draws = 0
            start_time = time.time()
            game_details = []

            for i in indices:
                if stopped.is_set():
                    raise InterruptedError("another match worker stopped")
                opening_index = (i // 2) % len(openings)
                start_sfen = openings[opening_index]["sfen"]
                trace = {}
                if i % 2 == 0:
                    # Engine1 = BLACK (先手), Engine2 = WHITE (後手)
                    result, moves = play_game(
                        engine1, engine2, go_params, start_sfen, args.max_moves,
                        args.clear_hash_each_move, history=not args.no_history, details=trace,
                        stop_event=stopped,
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
                        engine2, engine1, go_params, start_sfen, args.max_moves,
                        args.clear_hash_each_move, history=not args.no_history, details=trace,
                        stop_event=stopped,
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
                        "trace": trace,
                        "opening_source": openings[opening_index],
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

            return game_details, root_statistics_config, (
                engine1.timing() if isinstance(engine1, RootStatisticsEngine) else None)
        except BaseException:
            stopped.set()
            raise
        finally:
            cleanup_error = None
            for engine in reversed(owned):
                try:
                    if isinstance(engine, EngineProcess):
                        engine.close()
                    else:
                        engine.quit()
                except Exception as error:
                    stopped.set()
                    cleanup_error = error
            if cleanup_error is not None:
                raise cleanup_error

    pool = ThreadPoolExecutor(max_workers=min(workers, games // 2))
    try:
        futures = [pool.submit(worker, [2*p+c for p in range(w, games//2, workers) for c in (0,1)])
                   for w in range(min(workers, games//2))]
        pieces = [future.result() for future in futures]
    finally:
        stopped.set()
        pool.shutdown(wait=True, cancel_futures=True)
    game_details = sorted([row for rows, _, _ in pieces for row in rows], key=lambda r: r['game'])
    root_statistics_config = pieces[0][1]
    timings = [timing for _, _, timing in pieces if timing is not None]
    root_timing = {k: sum(t[k] for t in timings) for k in timings[0]} if timings else None
    e1_wins = sum(r['engine1_result'] == 'win' for r in game_details)
    e1_losses = sum(r['engine1_result'] == 'loss' for r in game_details)
    draws = sum(r['engine1_result'] == 'draw' for r in game_details)
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
            "workers": min(workers, games // 2),
            "history": not args.no_history,
            "reset_each_game": True,
            "clear_hash_each_move": args.clear_hash_each_move,
            "engine1_root_statistics": root_statistics_config,
            "engine1_root_statistics_timing": (
                root_timing
            ),
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



if __name__ == "__main__":
    main()
