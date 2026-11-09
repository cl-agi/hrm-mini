"""CPU integration checks for the standalone TensorFlow environment."""
import random
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from pysat.solvers import Glucose4

from hrm_querysat.data.CNFGen import SAT_3, Clique, KColor
from hrm_querysat.data.k_sat import KSAT
from hrm_querysat.model.querysat import QuerySAT
from hrm_querysat.optimization.AdaBelief import AdaBeliefOptimizer


class InstallationTest(unittest.TestCase):
    def test_generators_produce_satisfiable_integer_cnf(self):
        # Fixed test fixture seed, not a scientific repetition.
        random.seed(12345)
        np.random.seed(12345)
        with tempfile.TemporaryDirectory() as directory:
            generators = [SAT_3(directory, min_vars=5, max_vars=5),
                          Clique(directory, min_vertices=5, max_vertices=5),
                          KColor(directory, min_vertices=5, max_vertices=5),
                          KSAT(directory, min_vars=5, max_vars=5)]
            for generator in generators:
                with self.subTest(generator=type(generator).__name__):
                    n, clauses, *_ = next(generator.train_generator())
                    self.assertTrue(clauses)
                    self.assertTrue(all(0 < abs(int(x)) <= n for c in clauses for x in c))
                    with Glucose4(bootstrap_with=clauses) as solver:
                        self.assertTrue(solver.solve())

    def test_predict_update_and_checkpoint(self):
        tf.keras.utils.set_random_seed(12345)
        def sparse(indices, shape):
            return tf.sparse.reorder(tf.SparseTensor(indices, np.ones(len(indices), np.float32), shape))
        # (x1 or x2) and (not x1 or x3) and (not x2 or not x3).
        inputs = dict(adj_matrix=sparse([[0,0],[1,0],[3,1],[2,1],[4,2],[5,2]], [6,3]),
                      clauses_graph=sparse([[0,0],[0,1],[0,2]], [1,3]),
                      variables_graph=sparse([[0,0],[0,1],[0,2]], [1,3]),
                      solutions=tf.ragged.constant([[1,0,1]], dtype=tf.int32, row_splits_dtype=tf.int32))
        def model():
            return QuerySAT(AdaBeliefOptimizer(.0002, beta_1=.6, clip_gradients=True),
                            train_rounds=2, test_rounds=2)
        net = model()
        output = net.predict_step(**inputs)
        self.assertEqual(tuple(output['prediction'].shape), (3,))
        self.assertTrue(np.isfinite(output['prediction'].numpy()).all())
        before = [v.numpy().copy() for v in net.trainable_variables]
        update = net.train_step(**inputs)
        self.assertTrue(np.isfinite(update['loss'].numpy()).all())
        self.assertTrue(any(not np.array_equal(a, b.numpy()) for a,b in zip(before,net.trainable_variables)))
        with tempfile.TemporaryDirectory() as directory:
            prefix = tf.train.Checkpoint(model=net).save(directory + '/ckpt')
            restored = model()
            restored.train_step(**inputs)  # Materialize optimizer slots before restoring.
            tf.train.Checkpoint(model=restored).restore(prefix).assert_consumed()
            for a,b in zip(net.trainable_variables,restored.trainable_variables):
                np.testing.assert_array_equal(a.numpy(),b.numpy())
            for a,b in zip(net.optimizer.variables(),restored.optimizer.variables()):
                np.testing.assert_array_equal(a.numpy(),b.numpy())


if __name__ == '__main__':
    unittest.main()
