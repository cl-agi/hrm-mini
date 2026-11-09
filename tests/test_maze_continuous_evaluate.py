import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from hrm.analyse.maze.continuous_inference import evaluate as module
from hrm.analyse.maze.continuous_inference.data import make_model
from hrm.analyse.maze.continuous_inference.metrics import geometry
from hrm.analyse.maze.continuous_inference.summarize import combine


class MazeContinuousEvaluateTests(unittest.TestCase):
    def test_current_ever_and_recovery(self):
        trace = np.zeros((2, 4, 2), dtype=bool)
        trace[0, :, 0] = [True, False, True, False]
        trace[1, :, 0] = [False, False, False, True]
        summary = module.summarize(trace)
        self.assertEqual(summary['correct_to_wrong_transitions'], 2)
        self.assertEqual(summary['examples_with_correct_to_wrong'], 1)
        self.assertEqual(summary['curve'][-1]['current_valid'], 1)
        self.assertEqual(summary['curve'][-1]['ever_valid'], 2)

    def test_native_models_and_shard_accounting(self):
        small = dict(seq_len=901, vocab_size=6, hidden_size=8, intermediate_size=16, head_dim=4,
                     num_layers=1, H_cycles=1, L_cycles=1, cycles=1, bptt=True, norm_eps=1e-6,
                     rope_theta=10000., forward_dtype='float32', dual_module=False, hh_link=True,
                     sym_io=False)
        x = np.ones((3, 901), dtype=np.int64); x[:, 0] = 0
        x[:, 1:4] = [3, 2, 4]
        y = x.copy(); y[:, 2] = 5
        ds, dg = geometry(x)
        with tempfile.TemporaryDirectory() as d, patch(
                'hrm.analyse.maze.continuous_inference.data.model_config', return_value=small):
            root = Path(d)
            np.savez_compressed(root/'maze_test.npz', x=x, y=y, ds=ds, dg=dg, ids=[10, 11, 12])
            for kind in ('hrm', 'trm', 'rt'):
                cp = root/f'{kind}.pt'
                torch.save(make_model('maze', kind, 'cpu').state_dict(), cp)
                outputs = []
                for shard in range(2):
                    out = root/f'{kind}-{shard}'; outputs.append(out)
                    result = module.evaluate(kind, cp, root, out, cycles=3,
                                             batch_size=1, device='cpu', shard=shard, shards=2)
                    self.assertTrue(result['final_cpu_audit'])
                    self.assertTrue(all(p['logits_bitwise_equal'] for p in result['parity']))
                self.assertEqual(combine(outputs)['count'], 3)
                with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
                    combine(outputs[:1])
                with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
                    combine(outputs+outputs)
                with self.assertRaises(FileExistsError):
                    module.evaluate(kind, cp, root, outputs[0], cycles=3, device='cpu')
            receipt = outputs[0]/'summary.json'
            data = json.loads(receipt.read_text()); data['curve'][-1]['current_valid'] += 1
            receipt.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'counts'):
                combine(outputs)


if __name__ == '__main__':
    unittest.main()
