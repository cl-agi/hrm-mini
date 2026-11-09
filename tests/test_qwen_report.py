import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from hrm.analyse.baselines.qwen.report import cluster_statistics,report


class QwenReportTest(unittest.TestCase):
    def test_matches_original_paired_multinomial_bootstrap(self):
        base=np.array([0,.25,.5,1]);student=np.array([.5,.25,1,1])
        seed=12345;reps=1000
        weights=np.random.default_rng(seed).multinomial(4,np.full(4,.25),size=reps)/4
        actual=cluster_statistics(base,student,reps,seed)
        for name,values in zip(actual,(base,student,student-base)):
            interval=np.quantile(weights@values,[.025,.975])*100
            np.testing.assert_allclose([actual[name]['ci95_low_percent'],actual[name]['ci95_high_percent']],interval,rtol=0,atol=1e-12)
        self.assertEqual(actual['paired_sft_minus_instruct']['mean_percent'],25.)

    def test_mismatched_cohort_and_changed_settings_rejected(self):
        first=({0:('q','a')},{0:1},4,{'source':'a'})
        second=({0:('different','a')},{0:2},4,{'source':'b'})
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'summary.json'
            with patch('hrm.analyse.baselines.qwen.report.read_condition',side_effect=[first,second]):
                with self.assertRaisesRegex(ValueError,'same indexed puzzles'):report('a','b',out)
            second=({0:('q','a')},{0:2},4,{'source':'b'})
            with patch('hrm.analyse.baselines.qwen.report.read_condition',side_effect=[first,second]):report('a','b',out,replicates=100,seed=12345)
            before=out.read_bytes()
            with patch('hrm.analyse.baselines.qwen.report.read_condition',side_effect=[first,second]):report('a','b',out,replicates=100)
            self.assertEqual(before,out.read_bytes())
            with patch('hrm.analyse.baselines.qwen.report.read_condition',side_effect=[first,second]):
                with self.assertRaisesRegex(ValueError,'settings changed'):report('a','b',out,replicates=101)
