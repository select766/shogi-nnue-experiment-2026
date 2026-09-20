"""Offline verification of completed protocol-v2 qualification; no engine launch."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cshogi

from train_nnue.match_protocol_pilot import sha256
from train_nnue.match_protocol_qualification import FIXTURES, FIXTURE_OPTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(json.loads((args.attempt / 'artifacts/latest.json').read_text())['directory'])
    read = lambda name: json.loads((root / name).read_text())
    rows, manifest, summary = read('records.json'), read('manifest.json'), read('summary.json')
    assert manifest['protocol_version'] == 2
    assert manifest['fixture_options'] == FIXTURE_OPTIONS
    assert manifest['fixtures'] == json.loads(json.dumps(FIXTURES))
    assert manifest['expected_cases'] == 48 and manifest['expected_searches'] == 480
    assert summary['status'] == 'pass' and summary['failure'] is None
    assert summary['complete'] == summary['expected'] == len(rows) == 48
    cells = Counter((r['model'], r['fixture'], r['history'], r['clear'], r['limit']) for r in rows)
    assert cells == Counter({(m, f[0], h, c, n): 1 for m in ('candidate', 'control')
                            for f in FIXTURES for h in (False, True)
                            for c in (False, True) for n in (8, 12)})
    fixtures = {f[0]: f[1:] for f in FIXTURES}
    nodes, terminations = [], Counter()
    for number, row in enumerate(rows, 1):
        assert row == read(f'fixture-{number:03d}.json')
        assert row['complete'] and row['failed_searches'] == []
        sfen, cycle, outcome = fixtures[row['fixture']]
        trace, limit = row['trace'], row['limit']
        assert trace['start_sfen'] == sfen and trace['history'] == row['history']
        assert trace['moves_usi'] == list(cycle) * (limit // 4)
        assert len(trace['searches']) == limit
        board = cshogi.Board(sfen)
        for i, search in enumerate(trace['searches']):
            position = 'sfen ' + (sfen if row['history'] else board.sfen())
            if row['history'] and i:
                position += ' moves ' + ' '.join(trace['moves_usi'][:i])
            assert search['position'] == position and search['turn'] == board.turn
            token = trace['moves_usi'][i]
            assert search['bestmove'] == token
            replies = [line.split()[1] for line in search['info'] if line.startswith('bestmove ')]
            assert replies == [token]
            assert sum(line.startswith('info string blending_weight=') for line in search['info']) == 1
            cache = [line for line in search['info'] if line.startswith('info string dynamic_weight_cache ')]
            assert len(cache) == 1 and f"clear={int(row['clear'])} " in cache[0]
            assert search['nodes'] > 0
            nodes.append(search['nodes'])
            move = board.move_from_usi(token)
            assert move and board.is_legal(move)
            board.push(move)
        assert trace['final_sfen'] == board.sfen()
        assert trace['result'] == (0 if limit == 8 else outcome)
        assert trace['termination'] == ('max_moves' if limit == 8 else row['fixture'])
        terminations[trace['termination']] += 1
    assert len(nodes) == 480
    resets = {}
    evidence = {name: sha256(root / name) for name in ('manifest.json', 'records.json', 'summary.json')}
    library_counts = {}
    for model in ('candidate', 'control'):
        config = read(f'engine-{model}.json')
        assert config['options']['GenerateAllLegalMoves'] == 'true'
        assert any('option name GenerateAllLegalMoves ' in line for line in config['advertised'])
        log = Path(config['log']).read_text().splitlines()
        commands = [line for line in log if line.startswith('> ')]
        assert '> setoption name GenerateAllLegalMoves value true' in commands
        searches = [s for r in rows if r['model'] == model for s in r['trace']['searches']]
        assert [line for line in commands if line.startswith('> position ')] == ['> position ' + s['position'] for s in searches]
        assert [line for line in commands if line.startswith('> go ')] == ['> go nodes 128 searchmoves ' + s['bestmove'] for s in searches]
        assert [line.split()[2] for line in log if line.startswith('< bestmove ')] == [s['bestmove'] for s in searches]
        starts = [i for i, line in enumerate(commands) if line == '> usinewgame']
        # The fixture shares one process between both sides; play_game resets each side.
        assert len(starts) == 48 and all(commands[i - 1] == '> isready' for i in starts)
        resets[model] = len(starts)
        libraries = read(f'libraries-{model}.json')
        assert libraries and all(sha256(p) == digest for p, digest in libraries.items())
        library_counts[model] = len(libraries)
        evidence[config['log']] = sha256(config['log'])
    assert all(sha256(p) == digest for p, digest in manifest['files'].items())
    inputs = json.loads((args.attempt / 'input-manifest.json').read_text())
    assert all(sha256(p) == digest for p, digest in inputs.items())
    execution = json.loads((args.attempt / 'execution.json').read_text())
    assert execution['exit_code'] == 0 and execution['reason'] == 'exited'
    result = dict(verified=True, cases=len(rows), searches=len(nodes), failed_searches=0,
                  terminations=dict(terminations), nodes_min=min(nodes), nodes_max=max(nodes),
                  per_model_resets=resets, runtime_cshogi_version=manifest['cshogi_version'],
                  runtime_files_verified=len(manifest['files']), prepare_inputs_verified=len(inputs),
                  loaded_libraries_verified=library_counts, evidence_sha256=evidence,
                  elapsed_seconds=execution['elapsed_seconds'], strength_inference='not measured')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
