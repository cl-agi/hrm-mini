import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
try:
    import numba
except ImportError:
    numba=None


@unittest.skipIf(numba is None,'Install .[search] to check E31 annealing')
class AnnealingTests(unittest.TestCase):
    def test_solver_invariants_and_recorded_window(self):
        from hrm.analyse.dynamics.search_baselines.annealing import solve,energy,valid
        from hrm.analyse.dynamics.search_baselines.run_sa import check_puzzle,final_window
        puzzle=np.zeros(81,dtype=np.int64);puzzle[0]=1
        check_puzzle(puzzle)
        a=solve(puzzle,1794199000,500,17);b=solve(puzzle,1794199000,500,17)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        final,trace,boards,*_=a
        self.assertTrue(np.all(boards[:,0]==1))
        for board,row in zip(boards,trace):
            independent=sum(9-len(set(u)) for u in [*board.reshape(9,9),*board.reshape(9,9).T])
            self.assertEqual(energy(board),independent);self.assertEqual(row[1],independent)
            for r in (0,3,6):
                for c in (0,3,6):self.assertEqual(set(board.reshape(9,9)[r:r+3,c:c+3].flatten()),set(range(1,10)))
        self.assertEqual(valid(final,puzzle),energy(final)==0)
        window=final_window(trace,boards);self.assertEqual(window['distance'].shape,(99,99))
        np.testing.assert_array_equal(window['distance'],window['distance'].T)
        self.assertEqual(window['window_end'],500);self.assertEqual(np.trace(window['distance']),0)
        bad=puzzle.copy();bad[1]=1
        with self.assertRaises(ValueError):check_puzzle(bad)
        trace=np.array([[0,5,1,0,0],[10,5,1,1,1],[20,0,1,1,0]],dtype=float)
        w=final_window(trace,np.stack([puzzle,puzzle,puzzle]),3)
        self.assertEqual(w['window_start'],10);self.assertTrue(np.all(w['record_indices']>=1))

    def test_fresh_records_freeze_seeds_and_fail_on_reuse(self):
        from hrm.analyse.dynamics.search_baselines.run_sa import main
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();out=root/'records'
            answer=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
            (data/'test.csv').write_text('question,answer\n'+'.'*81+','+answer+'\n')
            args=['--dataset',str(data),'--board-id','0','--budget','20','--stride','5','--seeds','1794199000','1794199001','1794199002','--output-dir',str(out)]
            main(args+['--prepare-only']);main(args)
            report=json.loads((out/'manifest.json').read_text());self.assertEqual(report['status'],'complete');self.assertEqual(len(report['runs']),3)
            self.assertTrue(all(r['steps']==20 and not r['window_ends_at_solution'] for r in report['runs']))
            with self.assertRaises(FileExistsError):main(args)
