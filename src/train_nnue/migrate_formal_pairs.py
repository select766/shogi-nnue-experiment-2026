"""Explicit migration of the reviewed 63 accepted pairs; never starts engines."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess

from train_nnue.decision_replication_audit import audit_pair, load
from train_nnue.decision_replication_formal import identity, COST
from train_nnue.match_protocol_pilot import dump, sha256

LEGACY_COMMIT = 'f3e74358608b0917662a20977d05aae5e49796b0'
SOURCE = Path('.research-loop/experiments/decision-replication-formal-floodgate2025-chunk001/attempt-001')
# Only these reviewed implementation changes are allowed. Assets/protocol/data must match.
CHANGED = {
    'src/train_nnue/decision_replication_audit.py',
    'src/train_nnue/decision_replication_formal.py',
    'src/train_nnue/run_match.py',
    'src/train_nnue/match_protocol_pilot.py',
    'tests/test_decision_replication_formal.py',
}


def migrate(target):
    source = SOURCE.resolve()
    target = Path(target).resolve()
    if target == source or target.exists():
        raise ValueError('migration target must be a new directory')
    old = load(source / 'input-manifest.json')
    legacy_sources = {}
    for path, digest in old.items():
        if path in CHANGED:
            content = subprocess.check_output(['git', 'show', f'{LEGACY_COMMIT}:{path}'])
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f'legacy source mismatch: {path}')
            legacy_sources[path] = content
        elif sha256(path) != digest:
            raise ValueError(f'fixed input changed: {path}')
    protocol, chunk, libraries = [load(source / n) for n in ('protocol.json', 'chunk.json', 'libraries.json')]
    if protocol != load(COST/'protocol.json') or chunk != load(COST/'chunks.json')['chunks'][0]:
        raise ValueError('fixed protocol/chunk mismatch')
    entries = [load(p) for p in sorted((source/'artifacts').glob('pair-*.json'))]
    if [e['pair_id'] for e in entries] != list(range(1,64)):
        raise ValueError('expected exactly the reviewed 63 accepted pairs')
    rows = [__import__('json').loads(line) for line in Path(
        'dataset/floodgate2025/match-plan-v1/openings10000.jsonl').read_text().splitlines()]
    legacy = {}
    exec(compile(legacy_sources['src/train_nnue/decision_replication_audit.py'],
                 'legacy_audit.py', 'exec'), legacy)
    old_identity = identity(source)

    def verify(entry):
        i = entry['pair_id']
        if entry['identity'] != old_identity or entry['game_hash'] != rows[i-1]['game_hash']:
            raise ValueError('legacy marker identity mismatch')
        directory = Path(entry['directory'])
        if any(r['pair_id'] != i for r in load(directory/'records.json')):
            raise ValueError('legacy record pair ID mismatch')
        before = legacy['audit_pair'](directory, rows[i-1], protocol, libraries)
        after = audit_pair(directory, rows[i-1], protocol, libraries)
        if before != after or before != entry['audit']:
            raise ValueError('audit migration mismatch')
        return entry

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify, entries))
    target.mkdir(parents=True)
    (target/"artifacts").mkdir()
    for name in ('protocol.json', 'chunk.json', 'libraries.json', 'data-catalog.json'):
        (target/name).write_bytes((source/name).read_bytes())
    for path, content in legacy_sources.items():
        dest = target/'legacy-source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    dump(target/'legacy-input-manifest.json', old)
    # Replace implementation hashes explicitly; keep all immutable old input checks.
    manifest = {path: sha256(path) for path in old}
    for path in ('src/train_nnue/migrate_formal_pairs.py',):
        manifest[path] = sha256(path)
    for name in ('protocol.json', 'chunk.json', 'libraries.json'):
        manifest[str(target/name)] = sha256(target/name)
    dump(target/'input-manifest.json', manifest)
    new_identity = identity(target)
    for entry in verified:
        dump(target/'artifacts'/f"pair-{entry['pair_id']:05d}.json",
             {**entry, 'identity': new_identity, 'migration_source_identity': old_identity})
    dump(target/'migration.json', dict(source=str(source), source_commit=LEGACY_COMMIT,
         source_identity=old_identity, target_identity=new_identity,
         changed_sources={p: dict(old=old[p], new=sha256(p)) for p in sorted(CHANGED)},
         accepted_pairs=list(range(1,64)), unaccepted_pair64='preserved in source; replay both colors',
         validation='old and new auditors agree on all 63 pairs', workers=4,
         protocol_change='pair scheduling only; Threads=1, nodes and all game rules unchanged'))
    print(f'Migrated 63 audited pairs to {target}; no engines started')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    migrate(args.target)


if __name__ == '__main__':
    main()
