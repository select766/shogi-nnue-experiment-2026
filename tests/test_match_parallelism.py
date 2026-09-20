import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from train_nnue import run_match


class FakeEngine:
    instances=[]
    def __init__(self,path):
        self.owner=threading.get_ident()
        self.closed=False
        self.instances.append(self)
    def setoption(self,*args): pass
    def isready(self):
        assert self.owner==threading.get_ident()
    def usinewgame(self): pass
    def position(self,**kwargs): pass
    def go(self,listener,**kwargs):
        time.sleep(.01)
        listener('info nodes 1')
        listener('bestmove resign')
        return 'resign',None
    def quit(self): self.closed=True


class MatchParallelTest(unittest.TestCase):
    def run_match(self,workers,output,extra=()):
        with patch('sys.argv',['match','--engine1','a','--engine2','b','--games','8',
                              '--workers',str(workers),'--output',str(output),*extra]), \
             patch.object(run_match,'Engine',FakeEngine),contextlib.redirect_stdout(io.StringIO()):
            run_match.main()
        return json.loads(output.read_text())

    def test_parallel_pair_ids_and_results_equal_serial_with_separate_engines(self):
        with tempfile.TemporaryDirectory() as d:
            FakeEngine.instances=[]
            one=self.run_match(1,Path(d)/'one.json')
            four=self.run_match(4,Path(d)/'four.json')
        for key in ('wins','losses','draws','elo_difference'):
            self.assertEqual(one[key],four[key])
        fields=('game','opening_index','engine1_color','engine1_result','moves')
        self.assertEqual([[r[k] for k in fields] for r in one['details']],
                         [[r[k] for k in fields] for r in four['details']])
        self.assertEqual(len(FakeEngine.instances),10)
        self.assertTrue(all(e.closed for e in FakeEngine.instances))

    def test_time_control_parallelism_refused(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): self.run_match(4,Path(d)/'out', ['--byoyomi','10'])

    def test_failure_quits_every_created_engine_and_publishes_no_result(self):
        FakeEngine.instances=[]
        with tempfile.TemporaryDirectory() as d, patch.object(run_match,'play_game',side_effect=ValueError('bad move')):
            output=Path(d)/'out'
            with self.assertRaisesRegex((ValueError,InterruptedError),'bad move|stopped'):
                self.run_match(4,output)
            self.assertFalse(output.exists())
        self.assertTrue(all(e.closed for e in FakeEngine.instances))
