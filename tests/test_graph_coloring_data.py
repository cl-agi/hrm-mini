import unittest
from unittest.mock import patch
import numpy as np
from hrm.analyse.sat.graph_coloring.data import graph_to_cnf,load_data


def fixture(k):
    return dict(id=f'coloring:k{k}:0',num_vertices=256,num_colors=k,
                edges=[[v,(v+step)%256] for v in range(256) for step in (1,2)],
                encoding_sources=list(range(256)) if k==3 else [])


class GraphColoringDataTests(unittest.TestCase):
    def test_padding_and_copy_constraints(self):
        for k in (3,4):
            row=fixture(k);c=graph_to_cnf(row)
            self.assertEqual(c.shape,(2304,k))
            np.testing.assert_array_equal(c[0],np.arange(1,k+1))
            np.testing.assert_array_equal(c[256],[-1]+[-(k+1)]*(k-1))
            if k==3:np.testing.assert_array_equal(c[-2:],np.array([[-256,1024,1024],[256,-1024,-1024]]))
    def test_bad_rows_and_local_path_no_fallback(self):
        row=fixture(3);row['encoding_sources'][0]=768
        with self.assertRaises(ValueError):graph_to_cnf(row)
        row=fixture(4);row['edges'][0]=row['edges'][1]
        with self.assertRaisesRegex(ValueError,'Duplicate'):graph_to_cnf(row)
        with patch('hrm.hub.dataset',side_effect=AssertionError('must not fetch')):
            with self.assertRaises(FileNotFoundError):load_data(3,'/nonexistent/graphs.parquet')

    def test_flat200_exact_one_color_constraints(self):
        from hrm.analyse.sat.graph_coloring.flat_metrics import check_predictions,check_boolean_graph_predictions
        row=dict(id='flat200-fixture',num_vertices=200,num_colors=3,
                 edges=[[u,v] for u in range(100) for v in range(100,200)][:479],encoding_sources=None)
        c=graph_to_cnf(row)
        self.assertEqual(c.shape,(2237,3))
        np.testing.assert_array_equal(c[:4],[[-1,-2,-2],[-1,-3,-3],[-2,-3,-3],[1,2,3]])
        assignments=np.ones((1,1,600),dtype=np.int32)
        for v in range(200):assignments[0,0,3*v+int(v>=100)]=2
        full,_=check_predictions(assignments,c[None])
        self.assertTrue(full.all())
        flags=check_boolean_graph_predictions(assignments==2,c[None],np.asarray(row["edges"])[None])
        self.assertTrue(flags.all())
        assignments[0,0,1]=2 # second true color violates at-most-one
        full,_=check_predictions(assignments,c[None])
        self.assertFalse(full.any())
        flags=check_boolean_graph_predictions(assignments==2,c[None],np.asarray(row["edges"])[None])
        self.assertFalse(flags[0,0,0]);self.assertTrue(flags[0,0,1])
