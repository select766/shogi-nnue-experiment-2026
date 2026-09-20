import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import cshogi

from train_nnue.decision_replication_formal import execute, reusable
from train_nnue.match_protocol_pilot import dump

MODULE = 'train_nnue.decision_replication_formal.'


class FormalTest(unittest.TestCase):
    def test_explicit_retry_reuses_complete_pairs_without_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            dump(run/'libraries.json', {})
            dump(run/'input-manifest.json', {})
            entry = dict(pair_id=1, directory='audited')
            with patch(MODULE+'check', return_value=({'first_pair': 1}, [{}], {})), \
                 patch(MODULE+'identity', return_value={}), \
                 patch(MODULE+'runtime_manifest', return_value={}), \
                 patch(MODULE+'reusable', return_value=entry) as reuse, \
                 patch(MODULE+'AuditedEngine') as engine:
                execute(run, 2)
                engine.assert_not_called()
                reuse.assert_called_once()
            out = Path(json.loads((run/'artifacts/latest.json').read_text())['directory'])
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['pairs'], [entry])
            self.assertEqual(summary['status'], 'complete')

    def test_input_failure_preserves_summary_without_starting_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch(MODULE+'check', side_effect=ValueError('hash mismatch')), \
                 patch(MODULE+'AuditedEngine') as engine:
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    execute(run, 2)
                engine.assert_not_called()
            self.assertIn('hash mismatch', (run/'result-summary.md').read_text())

    def test_marker_reaudited_and_identity_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'artifacts').mkdir()
            dump(run/'records.json', [dict(pair_id=1), dict(pair_id=1)])
            dump(run/'artifacts/pair-00001.json', dict(pair_id=1, identity={'version': 'a'}, directory=str(run)))
            with patch(MODULE+'identity', return_value={'version': 'b'}), patch(MODULE+'audit_pair') as audit:
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    reusable(run, 1, {}, {}, {})
                audit.assert_not_called()
            with patch(MODULE+'identity', return_value={'version': 'a'}), \
                 patch(MODULE+'audit_pair', side_effect=AssertionError('bad trace')):
                with self.assertRaisesRegex(AssertionError, 'bad trace'):
                    reusable(run, 1, {}, {}, {})
            self.assertIsNone(reusable(run, 2, {}, {}, {}))

    def test_driver_order_atomic_pair_acceptance_and_partial_failure(self):
        for fail_second in (False, True):
            with self.subTest(fail_second=fail_second), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                dump(run/'libraries.json', {'candidate': {}, 'control': {}})
                dump(run/'input-manifest.json', {})
                opening = dict(game_hash='fixed', sfen=cshogi.STARTING_SFEN,
                               initial_sfen=cshogi.STARTING_SFEN, history_usi=[])
                config = dict(binary='unused', options={'candidate': {}, 'control': {}}, nodes=100000, max_moves=512)
                engines = []
                def engine(*args):
                    e = Mock()
                    e.process.pid = __import__('os').getpid()
                    e.options, e.advertised = {}, []
                    e.rss_kib.return_value = []
                    # execute uses stat on this fixed /tmp log path.
                    Path(args[3]).write_text('> usi\n< usiok\n> isready\n< readyok\n')
                    engines.append(e)
                    return e
                colors = []
                def game(first, second, go, sfen, **kwargs):
                    colors.append(0 if first is engines[0] else 1)
                    self.assertEqual(go, {'nodes': 100000})
                    self.assertTrue(kwargs['history'])
                    kwargs['details']['searches'] = []
                    if fail_second and len(colors) == 2:
                        raise TimeoutError('partial pair')
                    return -1, 0
                original = Path.read_text
                def read(path, *args, **kwargs):
                    if str(path).startswith('/proc/'):
                        return ''
                    return original(path, *args, **kwargs)
                with patch(MODULE+'check', return_value=({'first_pair': 1}, [opening], config)), \
                     patch(MODULE+'runtime_manifest', return_value={}), \
                     patch(MODULE+'identity', return_value={'v': 'fixed'}), \
                     patch(MODULE+'AuditedEngine', side_effect=engine), \
                     patch(MODULE+'play_game', side_effect=game), \
                     patch(MODULE+'audit_pair', return_value={'games': 2}) as audit, \
                     patch.object(Path, 'read_text', read):
                    if fail_second:
                        with self.assertRaisesRegex(TimeoutError, 'partial pair'):
                            execute(run, 30)
                    else:
                        execute(run, 30)
                self.assertEqual(colors, [0, 1])
                self.assertEqual((run/'artifacts/pair-00001.json').exists(), not fail_second)
                self.assertEqual(audit.call_count, 0 if fail_second else 1)
                out = Path(json.loads((run/'artifacts/latest.json').read_text())['directory'])
                rows = json.loads((out/'pair-00001/records.json').read_text())
                self.assertTrue(rows[0]['complete'])
                self.assertEqual(rows[1]['complete'], not fail_second)
                self.assertLess((run/'result-summary.md').stat().st_size, 65536)
                for e in engines:
                    e.quit.assert_called_once()
