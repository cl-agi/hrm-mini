import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
try:
    import numba
except ImportError:
    numba=None
from hrm.models.hrm import HRM
from hrm.analyse.dynamics.search_baselines.extract_source import main
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


@unittest.skipIf(numba is None,'Install .[search] for combined E31 check')
class SearchComparisonTests(unittest.TestCase):
    def test_chain_and_tamper_rejection(self):
        from hrm.analyse.dynamics.search_baselines.run_sa import main as sa
        from hrm.analyse.dynamics.search_baselines.run_dfs import main as dfs
        from hrm.analyse.dynamics.search_baselines.run_hrm import main as hrm
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir();records=root/'records'
            answer=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
            (data/'test.csv').write_text('question,answer\n.'+answer[1:]+','+answer+'\n')
            shared=['--dataset',str(data),'--board-id','0']
            sa(shared+['--seeds','1794199000','1794199001','1794199002','--budget','10','--output-dir',str(records/'sa')])
            dfs(shared+['--output-dir',str(records/'dfs')])
            arch=dict(name='hrm.models.hrm@HRM',seq_len=82,vocab_size=10,hidden_size=8,intermediate_size=16,head_dim=4,num_layers=1,norm_eps=1e-6,rope_theta=10000.,forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True)
            torch.manual_seed(1794199000);model=HRM(arch);torch.save(model.state_dict(),root/'model.pt');(root/'model_config.json').write_text(json.dumps(dict(arch=arch)))
            hrm(shared+['--checkpoint',str(root/'model.pt'),'--cycles','1','--device','cpu','--output-dir',str(records/'hrm')])
            args=['--records',str(records)];main(args)
            report=json.loads((records/'comparison/comparison.json').read_text());self.assertEqual(report['status'],'complete');self.assertEqual(report['results']['dfs']['solver_status'],'solved')
            self.assertTrue(all(r['solved'] for r in report['results']['sa']))
            with self.assertRaises(FileExistsError):main(args)
            path=records/'hrm/trajectory.npz'
            with np.load(path,allow_pickle=False) as f:values={k:f[k] for k in f.files}
            values['distance'][0,1]=.9;np.savez_compressed(path,**values)
            mpath=records/'hrm/manifest.json';manifest=json.loads(mpath.read_text());manifest['artifacts']['trajectory.npz']=sha(path);mpath.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Distance reconstruction'):main(args+['--output-dir',str(root/'bad-distance')])
            planpath=records/'dfs/run_config.json';plan=json.loads(planpath.read_text());plan['board_id']=1;planpath.write_text(json.dumps(plan))
            mpath=records/'dfs/manifest.json';manifest=json.loads(mpath.read_text());manifest['plan_sha256']=sha(planpath);mpath.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'different puzzles'):main(args+['--output-dir',str(root/'bad-cohort')])
