import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('ringer') and importlib.util.find_spec('rdkit'), 'CREMP optional dependencies required')
class CREMPScoreTests(unittest.TestCase):
    def test_failed_candidates_stay_in_denominator_and_filter_empty_is_explicit(self):
        from rdkit import Chem
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from hrm.evaluation.cremp_score import score
        from ringer.utils import evaluation
        mol=Chem.MolFromSmiles('CC');conf=Chem.Conformer(2)
        conf.SetAtomPosition(0,(0.,0.,0.));conf.SetAtomPosition(1,(1.5,0.,0.));mol.AddConformer(conf)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            pq.write_table(pa.Table.from_pylist([{'id':'molecule','reference_conformers':{'molblock':Chem.MolToMolBlock(mol),'coordinates':[[[0.,0.,0.],[1.5,0.,0.]]]}}]),root/'refs.parquet')
            rows=[{'candidate':0,'id':'molecule','success':True,'filter_accept':False},{'candidate':1,'id':'molecule','success':False}]
            (root/'candidates.json').write_text(json.dumps(rows))
            (root/'molecules.pickle').write_bytes(pickle.dumps({0:mol}))
            with patch.multiple(evaluation,compute_rmsd_matrix=lambda *a,**k:np.zeros((1,1)),compute_ring_rmsd_matrix=lambda *a,**k:np.zeros((1,1)),compute_ring_tfd_matrix=lambda *a,**k:np.zeros((1,1))):
                report=score(root,root/'score.json',root/'refs.parquet')
                self.assertTrue(report['complete_split'])
                self.assertEqual(report['requested_candidates'],2)
                self.assertEqual(report['accepted_candidates'],1)
                metric=report['means']['ring_tfd']
                self.assertEqual(metric['cov_precision'],1.)
                self.assertEqual(metric['cov_precision_per_requested'],.5)
                report=score(root,root/'filtered.json',root/'refs.parquet',filter_mode='combined')
                self.assertEqual(report['means']['ring_tfd']['cov_recall'],0.)
                self.assertIsNone(report['means']['ring_tfd']['mat_recall'])
            (root/'candidates.json').write_text(json.dumps(rows+[rows[0]]))
            with self.assertRaisesRegex(ValueError,'Duplicate candidate'):score(root,root/'bad.json',root/'refs.parquet')


if __name__=='__main__':unittest.main()
