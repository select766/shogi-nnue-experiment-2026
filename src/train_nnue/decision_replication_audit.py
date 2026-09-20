"""Read-only complete-pair audit, derived from the reviewed cost16 audit."""
import gzip
import json
from collections import Counter
import cshogi
from train_nnue.match_protocol_pilot import sha256


def load(path):
    return json.loads(path.read_text())


def audit_pair(OUT, opening, protocol, libraries):
    if not __debug__:
        raise RuntimeError('Audit requires Python assertions; do not use -O/PYTHONOPTIMIZE')
    openings = [opening]
    assert load(OUT / 'openings.json') == openings
    assert load(OUT / 'protocol.json') == protocol
    records = load(OUT/'records.json')
    assert [(r['opening_index'], r['candidate_color']) for r in records] == [(i,c) for i in range(1) for c in (0,1)]
    commands, responses = {}, {}
    for model in ('candidate','control'):
        meta = load(OUT/f'engine-{model}.json')
        assert meta['options'] == protocol['options'][model]
        commands[model] = []
        responses[model] = []
        assert load(OUT/f'libraries-{model}.json') == libraries[model]
        for p,h in libraries[model].items():
            assert sha256(p) == h
    nodes, terminations, plies = [], Counter(), 0
    for number,r in enumerate(records,1):
        assert r == load(OUT/f'game-{number:03}.json') and r['complete']
        o = openings[r['opening_index']]
        t = r['trace']
        assert r['game_hash'] == o['game_hash']
        assert t['initial_sfen'] == o['initial_sfen'] and t['history_usi'] == o['history_usi']
        assert t['start_sfen'] == o['sfen'] and t['history'] is True
        b = cshogi.Board(o['initial_sfen'])
        key = lambda: ' '.join(b.sfen().split()[:3])
        occurrences, checks = {key(): [0]}, []
        def push(token):
            move = b.move_from_usi(token)
            assert move and b.is_legal(move)
            turn = b.turn
            b.push(move)
            checks.append((turn,b.is_check()))
            occurrences.setdefault(key(),[]).append(len(checks))
        for token in o['history_usi']: push(token)
        assert b.sfen() == o['sfen']
        for model in commands: commands[model] += ['isready','usinewgame']
        played, terminal = [], None
        def adjudicate():
            if b.is_game_over(): return (-1 if b.turn == 0 else 1, 'no_legal_moves')
            hits = occurrences[key()]
            if len(hits) >= 4:
                for side in (0,1):
                    own = [check for turn,check in checks[hits[-4]:] if turn == side]
                    if own and all(own): return (-1 if side == 0 else 1, 'perpetual_check_black' if side == 0 else 'perpetual_check_white')
                return (0,'repetition_draw')
            if len(played) >= 512: return (0,'max_moves')
        for s in t['searches']:
            assert terminal is None and adjudicate() is None
            assert s['turn'] == b.turn
            position = 'sfen ' + o['initial_sfen']
            history = o['history_usi'] + played
            if history: position += ' moves ' + ' '.join(history)
            assert s['position'] == position
            model = 'candidate' if b.turn == r['candidate_color'] else 'control'
            commands[model] += ['position '+position,'go nodes 100000']
            responses[model].extend(s['info'])
            assert sum(x.startswith('info string blending_weight=') for x in s['info']) == 1
            cache = [x for x in s['info'] if x.startswith('info string dynamic_weight_cache ')]
            assert len(cache) == 1 and 'clear=1 ' in cache[0]
            ns = [int(x.split()[x.split().index('nodes')+1]) for x in s['info'] if x.startswith('info ') and 'nodes' in x.split()]
            assert ns and ns[-1] == s['nodes'] and all(n >= 0 for n in ns)
            assert s['nodes'] > 0 or (s['bestmove'] == 'win' and b.is_nyugyoku())
            nodes.append(s['nodes'])
            assert s['info'][-1].split()[:2] == ['bestmove',s['bestmove']]
            token = s['bestmove']
            if token == 'resign': terminal = (-1 if b.turn == 0 else 1,'resign')
            elif token == 'win':
                assert b.is_nyugyoku()
                terminal = (1 if b.turn == 0 else -1,'entering_king')
            else:
                push(token)
                played.append(token)
        terminal = terminal or adjudicate()
        assert terminal == (t['result'],t['termination'])
        assert played == t['moves_usi'] and b.sfen() == t['final_sfen']
        assert r['candidate_score'] == (1 + t['result']*(1 if r['candidate_color']==0 else -1))/2
        plies += len(played)
        terminations[t['termination']] += 1
    for model in commands:
        with gzip.open(OUT/f'{model}.log.gz', 'rt') as stream:
            log = stream.read().splitlines()
        assert log.count('< readyok') == 2
        assert [x[2:] for x in log if x.startswith('> ')] == commands[model]
        actual, inside = [], False
        for line in log:
            if line.startswith('> go '): inside = True
            elif inside and line.startswith('< '):
                actual.append(line[2:])
                if line.startswith('< bestmove '): inside = False
        assert actual == responses[model]
    return dict(games=2, searches=len(nodes), legal_new_plies=plies,
                actual_nodes_total=sum(nodes), terminations=dict(terminations))
