import unittest
import numpy as np
from hrm.analyse.sat.graph_coloring.evaluate_querysat import score_selected


class QuerySATScoringTest(unittest.TestCase):
    def test_only_selected_head_counts(self):
        for flat,n in [(False,1024),(True,600)]:
            bits=np.zeros((2,8,n),dtype=bool)
            bits[:,1,0]=True
            bits[:,1,4]=True
            clauses=np.array([[1,1,1]])
            edges=np.array([[0,1]])
            rejected=score_selected(bits,np.array([0,0]),clauses,edges,3,flat)
            self.assertFalse(any(rejected.values()))
            accepted=score_selected(bits,np.array([0,1]),clauses,edges,3,flat)
            self.assertTrue(all(accepted.values()))

    def test_invalid_head_is_rejected(self):
        with self.assertRaises(ValueError):
            score_selected(np.zeros((1,8,600),bool),np.array([8]),np.array([[1,2,3]]),np.array([[0,1]]),3,True)
