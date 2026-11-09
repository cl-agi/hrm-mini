import argparse
import csv
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hrm import hub
from hrm.evaluation.published import add_arguments


class HubDefaultsTest(unittest.TestCase):
    def test_checkpoint_and_dataset_overrides_are_independent(self):
        p=argparse.ArgumentParser();add_arguments(p)
        a=p.parse_args(['--task','maze','--ckpt-dir','weights','--output','scores.json'])
        self.assertEqual(a.checkpoint,Path('weights'));self.assertIsNone(a.data)
        a=p.parse_args(['--dataset-path','test.csv','--output','scores.json'])
        self.assertEqual(a.data,Path('test.csv'));self.assertIsNone(a.checkpoint)

    def test_explicit_local_inputs_never_download(self):
        with tempfile.TemporaryDirectory() as d, patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('network called')):
            d=Path(d).resolve();(d/'model.pt').write_bytes(b'local');(d/'test_hard.csv').write_text('question,answer\n')
            (d/'validation.npz').write_bytes(b'local')
            for density in ['4.2','4.3']:(d/f'data_{density}.npz').write_bytes(b'local')
            self.assertEqual(hub.checkpoint('sudoku',d),d/'model.pt')
            self.assertEqual(hub.puzzle_data('sudoku',d),d/'test_hard.csv')
            self.assertEqual(hub.cremp_data(d),d/'validation.npz')
            self.assertEqual(hub.sat_public_data(d),d)
            for fn in [lambda:hub.checkpoint('sudoku',d/'missing'),lambda:hub.puzzle_data('sudoku',d/'missing'),lambda:hub.cremp_data(d/'missing'),lambda:hub.sat_public_data(d/'missing')]:
                with self.assertRaises((FileNotFoundError,NotADirectoryError)):fn()

    def test_public_sat_official_ids_and_density_order_survive_conversion(self):
        rows=[{'id':f'public:n512:a{d}:{i}','clauses':[[1,-2,3]],'solution':None} for d in ['4.2','4.3'] for i in [7,3]]
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'HRM_ASSET_CACHE':d}), patch.object(hub,'dataset',return_value=rows):
            root=hub.sat_public_data(n_vars=512)
            for density in ['4.2','4.3']:
                z=np.load(root/f'data_{density}.npz')
                self.assertEqual(z['official_ids'].tolist(),[7,3]);self.assertFalse(z['answers'].any())
                self.assertEqual(z['answers'].shape,(2,512));self.assertEqual(z['clauses'].tolist(),[[[1,-2,3]],[[1,-2,3]]])

    def test_public_sat_rejects_label_cues(self):
        rows=[{'id':'public:n512:a4.2:7','clauses':[[1,2,3]],'solution':[1]*512}]
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'HRM_ASSET_CACHE':d}), patch.object(hub,'dataset',return_value=rows):
            with self.assertRaisesRegex(ValueError,'answer cues'):hub.sat_public_data(n_vars=512)

    def test_maze_size_filter_preserves_spaces_and_row_order(self):
        rows=[{'maze':['S ',' G'],'solution':['So','oG']},{'maze':['S  ','   ','  G'],'solution':['Soo','  o','  G']},{'maze':['SG','##'],'solution':['SG','##']}]
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'HRM_ASSET_CACHE':d}), patch.object(hub,'dataset',return_value=rows):
            path=hub.puzzle_data('maze',size=2)
            with path.open() as f:actual=list(csv.DictReader(f))
            self.assertEqual([r['question'] for r in actual],['S  G','SG##'])

    def test_hf_training_conversion_binds_geometry_to_new_cache(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d)/'train.npz';cache.write_bytes(b'converted training inputs')
            original={'source_split':'train','source_cache_sha256':'original','mean_bond_distances':{'N':1.455}}
            with patch.object(hub,'data_file',return_value=Path('geometry.json')),patch.object(hub,'cremp_geometry',return_value=dict(original)):
                hub._bind_cremp_training_geometry(cache)
            metadata=json.loads(cache.with_suffix('.geometry.json').read_text())
            self.assertEqual(metadata['source_cache_sha256'],hashlib.sha256(cache.read_bytes()).hexdigest())
            self.assertEqual(metadata['published_source_cache_sha256'],'original')
            self.assertEqual(metadata['mean_bond_distances'],original['mean_bond_distances'])

    def test_geometry_requires_training_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'geometry.json'
            path.write_text(json.dumps({'mean_bond_distances':{'N':1.4}}))
            with self.assertRaisesRegex(ValueError,'Geometry must identify'):hub.cremp_geometry(path)
            metadata={'source_split':'train','source_cache_sha256':'a'*64,'mean_bond_distances':{'N':1.4}}
            path.write_text(json.dumps(metadata))
            self.assertEqual(hub.cremp_geometry(path),metadata)
            metadata['source_split']='validation';path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):hub.cremp_geometry(path)

    def test_wrong_downloaded_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.pt';p.write_bytes(b'bad')
            with patch('huggingface_hub.hf_hub_download',return_value=str(p)):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):hub.checkpoint('maze')

if __name__=='__main__':unittest.main()
