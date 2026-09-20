"""Fixed formal chunks with durable, audited complete-pair reuse on explicit retry."""
import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import time

import cshogi

from train_nnue.decision_replication_audit import audit_pair, load
from train_nnue.decision_replication_cost import POOL, verify_inputs
from train_nnue.match_protocol_pilot import AuditedEngine, dump, sha256
from train_nnue.match_protocol_qualification import runtime_manifest
from train_nnue.run_match import play_game

COST = Path('docs/research/results/decision-replication-cost-floodgate2025-20260920')


def check(run):
    verify_inputs(run)
    protocol = load(run / 'protocol.json')
    if protocol != load(COST / 'protocol.json'):
        raise ValueError('protocol differs from reviewed cost')
    chunk = load(run / 'chunk.json')
    if chunk not in load(COST / 'chunks.json')['chunks']:
        raise ValueError('chunk differs from fixed schedule')
    rows = [json.loads(line) for line in (POOL / 'openings10000.jsonl').read_text().splitlines()]
    selected = rows[chunk['first_pair']-1:chunk['last_pair']]
    for row in selected:
        board = cshogi.Board(row['initial_sfen'])
        for token in row['history_usi']:
            move = board.move_from_usi(token)
            if not move or not board.is_legal(move):
                raise ValueError('illegal formal source history')
            board.push(move)
        if board.sfen() != row['sfen']:
            raise ValueError('formal source history mismatch')
    return chunk, selected, protocol


def identity(run):
    return {name: sha256(run / name) for name in
            ('input-manifest.json', 'protocol.json', 'chunk.json', 'libraries.json')}


def reusable(run, pair_id, opening, protocol, libraries):
    marker = run / 'artifacts' / f'pair-{pair_id:05d}.json'
    if not marker.exists():
        return None
    entry = load(marker)
    if entry['identity'] != identity(run) or entry['pair_id'] != pair_id:
        raise ValueError('retry identity mismatch; review required')
    directory = Path(entry['directory'])
    if any(r['pair_id'] != pair_id for r in load(directory / 'records.json')):
        raise ValueError('stored pair ID mismatch')
    audit_pair(directory, opening, protocol, libraries)
    return entry


def save_log_slice(source, start, target):
    with Path(source).open('rb') as stream, gzip.open(target, 'wb') as dest:
        stream.seek(start)
        while data := stream.read(1024 * 1024):
            dest.write(data)


def execute(run, budget):
    out = run / 'artifacts' / (time.strftime('execution-%Y%m%dT%H%M%S') + f'-{os.getpid()}')
    out.mkdir(parents=True)
    dump(run / 'artifacts/latest.json', dict(directory=str(out)))
    deadline = time.monotonic() + budget
    engines, logs, accepted = [], {}, []
    status, failure = 'failed', 'interrupted'
    try:
        chunk, openings, protocol = check(run)
        libraries = load(run / 'libraries.json')
        dump(out / 'manifest.json', runtime_manifest(list(load(run / 'input-manifest.json'))))
        dump(out / 'identity.json', identity(run))
        for pair_id, opening in enumerate(openings, chunk['first_pair']):
            entry = reusable(run, pair_id, opening, protocol, libraries)
            if entry:
                accepted.append(entry)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError('formal chunk deadline')
            # Persistent two processes within an execution, as in cost16.
            if not engines:
                for model in ('candidate', 'control'):
                    log = f'/tmp/decision-formal-{out.name}-{model}.log'
                    engine = AuditedEngine(protocol['binary'], protocol['options'][model], deadline, log)
                    engines.append(engine)
                    logs[model] = log
                    maps = Path(f'/proc/{engine.process.pid}/maps').read_text()
                    (out / f'maps-{model}.txt').write_text(maps)
                    paths = sorted({line.split()[-1] for line in maps.splitlines()
                                    if line.split()[-1].startswith('/') and '.so' in line.split()[-1]})
                    actual = {p: sha256(p) for p in paths}
                    if actual != libraries[model]:
                        raise ValueError('loaded libraries differ from cost16')
                    dump(out / f'engine-{model}.json', dict(options=engine.options,
                         advertised=engine.advertised, log=log))
                    engine.log.flush()
                    initialization = Path(log).read_text().splitlines()
                    expected = ['usi'] + [f'setoption name {k} value {v}'
                                         for k, v in protocol['options'][model].items()] + ['isready']
                    if [s[2:] for s in initialization if s.startswith('> ')] != expected:
                        raise ValueError('initialization command mismatch')
                    if '< readyok' not in initialization or '< usiok' not in initialization:
                        raise ValueError('initialization handshake missing')
                    save_log_slice(log, 0, out / f'initialization-{model}.log.gz')
            pair = out / f'pair-{pair_id:05d}'
            pair.mkdir()
            dump(pair / 'openings.json', [opening])
            dump(pair / 'protocol.json', protocol)
            offsets = {}
            for model, engine in zip(('candidate', 'control'), engines):
                engine.log.flush()
                offsets[model] = Path(logs[model]).stat().st_size
                dump(pair / f'engine-{model}.json', load(out / f'engine-{model}.json'))
                dump(pair / f'libraries-{model}.json', libraries[model])
            records = []
            try:
                for color in (0, 1):
                    row = dict(opening_index=0, pair_id=pair_id, game_hash=opening['game_hash'],
                               candidate_color=color, complete=False, trace={})
                    records.append(row)
                    dump(out / 'progress.json', dict(pair_id=pair_id, candidate_color=color,
                         accepted_pairs=len(accepted), pair_directory=str(pair)))
                    start = time.monotonic()
                    try:
                        first, second = engines if color == 0 else engines[::-1]
                        result, _ = play_game(first, second, {'nodes': protocol['nodes']}, opening['sfen'],
                            max_moves=protocol['max_moves'], history=True, details=row['trace'],
                            initial_sfen=opening['initial_sfen'], history_usi=opening['history_usi'])
                        row.update(candidate_score=(1 + result*(1 if color == 0 else -1))/2, complete=True)
                        row['memory'] = [e.rss_kib() for e in engines]
                    except BaseException as error:
                        row['error'] = repr(error)
                        raise
                    finally:
                        row['wall_seconds'] = time.monotonic() - start
                        dump(pair / f'game-{color+1:03d}.json', row)
                        dump(pair / 'records.json', records)
            finally:
                for model, engine in zip(('candidate', 'control'), engines):
                    engine.log.flush()
                    save_log_slice(logs[model], offsets[model], pair / f'{model}.log.gz')
            audit = audit_pair(pair, opening, protocol, libraries)
            entry = dict(pair_id=pair_id, game_hash=opening['game_hash'], directory=str(pair),
                         identity=identity(run), audit=audit)
            # Only the atomic marker makes an audited complete pair reusable.
            dump(run / 'artifacts' / f'pair-{pair_id:05d}.json', entry)
            accepted.append(entry)
            dump(out / 'accepted.json', accepted)
        status, failure = 'complete', None
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        for engine in engines:
            try:
                engine.quit()
            except Exception as error:
                status, failure = 'failed', f'cleanup: {error!r}; previous={failure}'
        dump(out / 'summary.json', dict(status=status, failure=failure, accepted_pairs=len(accepted),
             pairs=accepted, strength_inference='not performed; requires all 10000 audited pairs'))
        (run / 'result-summary.md').write_text(
            f'# H-DECISION-REPLICATION formal chunk\n\nStatus: {status}\n'
            f'Failure: {str(failure)[:2000]}\nAccepted complete pairs: {len(accepted)}\n'
            f'Artifacts: {out}\nLogs: /tmp/decision-formal-{out.name}-*.log\n'
            'No interim strength decision. All 61 chunks are required for the single primary analysis.\n'
            'Failure/timeout requires review and explicit retry; incomplete pairs are not scored.\n')
    if status != 'complete':
        raise RuntimeError(failure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--budget-seconds', type=int, default=20400)
    args = parser.parse_args()
    if args.check_only or args.audit_only:
        chunk, rows, protocol = check(args.run_dir)
        if args.audit_only:
            libraries = load(args.run_dir / 'libraries.json')
            for pair_id, opening in enumerate(rows, chunk['first_pair']):
                if reusable(args.run_dir, pair_id, opening, protocol, libraries) is None:
                    raise ValueError(f'incomplete pair: {pair_id}')
            print('All fixed chunk pairs passed the complete-pair audit')
        print(f'Verified chunk {chunk["chunk"]}: {len(rows)} pairs; no engine started')
        return
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    execute(args.run_dir.resolve(), args.budget_seconds)


if __name__ == '__main__':
    main()
