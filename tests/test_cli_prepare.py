"""Exercise the documented preparation command from an empty local run."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from hrm.paths import repository_root


class PreparationCliTests(unittest.TestCase):
    def test_local_inputs_and_frozen_seed_survive_repeated_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();data=root/'data';data.mkdir()
            for split in ('train','test_hard'):
                with (data/f'{split}.csv').open('w') as f:
                    writer=csv.writer(f);writer.writerow(['question','answer'])
                    writer.writerow(['0'*81, ''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))])
            output=root/'run'
            env=dict(os.environ,HRM_REPOSITORY=str(repository_root()),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
            env.pop('RANK',None)
            command=[sys.executable,'-m','hrm','train','--experiment','sudoku',
                     '--recipe','sudoku-hrm','--dataset-path',str(data),
                     '--eval-dataset-path',str(data),'--fresh-seed','--prepare-only','--output',str(output)]
            for _ in range(2):
                subprocess.run(command,cwd=repository_root(),env=env,check=True,capture_output=True,text=True)
                config=json.loads((output/'resolved_config.json').read_text())
                seed=json.loads((output/'training_seed.json').read_text())['seed']
                self.assertEqual(config['seeds'],[seed])
                self.assertEqual(Path(config['data']['dataset_name']),data)
                self.assertEqual(Path(config['eval_dataset_name']),data)
                if _==0:first=config
                else:self.assertEqual(config,first)
