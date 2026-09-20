"""Compare sequential and concurrent fixed-node pairs on cost-only openings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from train_nnue.decision_replication_cost import verify_inputs
from train_nnue.decision_replication_formal import _worker
from train_nnue.decision_replication_audit import load
from train_nnue.match_protocol_pilot import dump


def signature(entry):
    rows = load(Path(entry['directory'])/'records.json')
    return [dict(color=r['candidate_color'], score=r['candidate_score'],
                 moves=r['trace']['moves_usi'], final=r['trace']['final_sfen'],
                 termination=r['trace']['termination'],
                 searches=[{k:s[k] for k in ('turn','position','bestmove','nodes')}
                           for s in r['trace']['searches']]) for r in rows]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verified-run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args=p.parse_args()
    rows=verify_inputs(args.verified_run)[:4]
    protocol=load(args.verified_run/'protocol.json')
    libraries=load(args.verified_run/'libraries.json')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results={}
    deadline=time.monotonic()+280
    for workers in (1,4):
        run=args.output_dir/f'workers-{workers}'
        run.mkdir()
        (run/"artifacts").mkdir()
        for name in ('input-manifest.json','protocol.json','chunk.json','libraries.json'):
            (run/name).write_bytes((args.verified_run/name).read_bytes())
        stop=threading.Event()
        tasks=list(enumerate(rows,1))
        started=time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(_worker,run,run/f'worker-{w}',tasks[w::workers],
                                 protocol,libraries,deadline,stop) for w in range(workers)]
            entries=sorted([e for f in futures for e in f.result()],key=lambda e:e['pair_id'])
        results[workers]=dict(seconds=time.monotonic()-started,
                              signatures=[signature(e) for e in entries])
    equal=results[1]['signatures']==results[4]['signatures']
    dump(args.output_dir/'comparison.json',dict(equal=equal,pairs=4,games_per_mode=8,
        sequential_seconds=results[1]['seconds'],parallel_seconds=results[4]['seconds'],
        comparison='all moves, final SFEN, termination, scores, positions, bestmoves and actual nodes',
        dataset='cost8 only; no strength inference'))
    if not equal:
        raise ValueError('sequential/concurrent results differ')
    print('PASS: all 4 cost pairs agree between 1 and 4 workers')


if __name__=='__main__':
    main()
