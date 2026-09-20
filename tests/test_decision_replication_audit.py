import gzip
from pathlib import Path
import tempfile
import unittest

import cshogi
from train_nnue.decision_replication_audit import audit_pair
from train_nnue.match_protocol_pilot import dump

WIN_SFEN = '5KG2/3+P+N+P2+S/+S1+N4G1/7sp/9/7g1/1+p4+p1P/+l1p2p1+l1/+b+rk+s4g w Rb2n2l10p 268'


class AuditTest(unittest.TestCase):
    def fixture(self, root, nodes=0, token='win', sfen=WIN_SFEN, mismatch=False):
        opening=dict(sfen=sfen,initial_sfen=sfen,history_usi=[],game_hash='synthetic')
        protocol=dict(options={'candidate':{},'control':{}})
        libraries={'candidate':{},'control':{}}
        dump(root/'openings.json',[opening]); dump(root/'protocol.json',protocol)
        logs={m:[] for m in libraries}
        records=[]
        for color in (0,1):
            side=cshogi.Board(sfen).turn
            model='candidate' if color==side else 'control'
            info=['info string blending_weight=[1]', 'info string dynamic_weight_cache clear=1 ']
            if nodes is not None: info.append(f'info score mate 1 nodes {nodes}')
            info.append('bestmove '+token)
            search=dict(turn=side,position='sfen '+sfen,bestmove=token,nodes=nodes,info=info)
            trace=dict(initial_sfen=sfen,history_usi=[],start_sfen=sfen,history=True,
                       searches=[search],result=-1,termination='entering_king',moves_usi=[],final_sfen=sfen)
            row=dict(opening_index=0,candidate_color=color,complete=True,game_hash='synthetic',
                     trace=trace,candidate_score=color)
            records.append(row); dump(root/f'game-{color+1:03d}.json',row)
            for m in libraries: logs[m].extend(['> isready','< readyok','> usinewgame'])
            logs[model].extend(['> position sfen '+sfen,'> go nodes 100000']+['< '+v for v in info])
        dump(root/'records.json',records)
        for m in libraries:
            dump(root/f'engine-{m}.json',dict(options={}))
            dump(root/f'libraries-{m}.json',{})
            if mismatch: logs[m][-1]='< bestmove resign'
            with gzip.open(root/f'{m}.log.gz','wt') as f: f.write('\n'.join(logs[m])+'\n')
        return opening,protocol,libraries

    def test_zero_node_legal_declaration_passes_full_communication_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            result=audit_pair(root,*self.fixture(root))
            self.assertEqual(result['actual_nodes_total'],0)
            self.assertEqual(result['terminations'],{'entering_king':2})

    def test_missing_negative_ordinary_zero_illegal_win_and_corruption_rejected(self):
        cases=[dict(nodes=None),dict(nodes=-1),dict(token='resign'),
               dict(sfen=cshogi.STARTING_SFEN),dict(sfen=cshogi.STARTING_SFEN,token='7g7f'),dict(mismatch=True)]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as d:
                root=Path(d)
                with self.assertRaises(AssertionError): audit_pair(root,*self.fixture(root,**case))
