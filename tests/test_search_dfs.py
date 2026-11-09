import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hrm.analyse.dynamics.search_baselines.run_dfs import solve,legal_candidates,choose_mrv_cell,sampled_distance,main


class DFSTests(unittest.TestCase):
    def test_state_events_and_budget(self):
        puzzle=np.array(list(map(int,'530070000600195000098000060800060003400803001700020006060000280000419005000080079')))
        boards,actions,status=solve(puzzle)
        self.assertEqual(status,'solved');self.assertEqual(actions[0],'initial')
        for previous,now,action in zip(boards,boards[1:],actions[1:]):
            changed=np.flatnonzero(previous!=now)
            if action=='dead_end':self.assertEqual(len(changed),0);self.assertEqual(choose_mrv_cell(now)[1],())
            else:
                self.assertEqual(len(changed),1);cell=int(changed[0]);self.assertEqual(puzzle[cell],0)
                if action=='assign':self.assertEqual(previous[cell],0);self.assertIn(now[cell],legal_candidates(previous,cell));self.assertEqual(cell,choose_mrv_cell(previous)[0])
                else:self.assertEqual(action,'undo');self.assertEqual(now[cell],0)
        ids,distance=sampled_distance(boards)
        expected=(boards[ids][:,None,:]!=boards[ids][None,:,:]).sum(-1)/81
        np.testing.assert_array_equal(distance,expected);self.assertEqual(ids[0],0);self.assertEqual(ids[-1],len(boards)-1)
        impossible=puzzle.copy();impossible[0]=1
        failed,events,outcome=solve(impossible);self.assertEqual(outcome,'unsatisfiable');self.assertIn('undo',events);self.assertIn('dead_end',events);np.testing.assert_array_equal(failed[-1],impossible)
        short,_,status=solve(puzzle,3);self.assertEqual(status,'state_budget_exhausted');self.assertEqual(len(short),3)
        solved,ops,status=solve(boards[-1]);self.assertEqual(status,'solved');self.assertEqual(len(solved),1)

    def test_cli_uses_clues_not_target(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();out=root/'records'
            answer=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
            # Valid source CSV shape, deliberately different target: search must not use it.
            (data/'test.csv').write_text('question,answer\n.'+answer[1:]+','+'1'*81+'\n')
            args=['--dataset',str(data),'--board-id','0','--output-dir',str(out)]
            main(args+['--prepare-only']);main(args)
            report=json.loads((out/'manifest.json').read_text());self.assertEqual(report['solver_status'],'solved');self.assertFalse(report['matches_target'])
            with self.assertRaises(FileExistsError):main(args)
