"""Archived graph-coloring metrics, shared by recurrent evaluation and audit."""
import numpy as np


def check_predictions(predictions, clauses, edges, k):
    p = np.asarray(predictions)
    assert p.ndim == 3 and p.shape[2] == 1024
    assert k in (3, 4)
    original_n, original_m = 256*k, 256+512*k
    flags = np.zeros((len(p), p.shape[1], 3), dtype=bool)
    min_violations = np.zeros(p.shape[1], dtype=np.int32)
    for j in range(p.shape[1]):
        q, c, e = p[:, j], clauses[j], edges[j]
        sat = (q[:, np.abs(c)-1] == np.where(c > 0, 2, 1)[None]).any(-1)
        flags[:, j, 0] = sat.all(1) & np.isin(q, [1, 2]).all(1)
        flags[:, j, 1] = sat[:, :original_m].all(1) & np.isin(q[:, :original_n], [1, 2]).all(1)
        colors = (q[:, :original_n].reshape(len(q), 256, k) == 2).argmax(2)
        flags[:, j, 2] = (colors[:, e[:, 0]] != colors[:, e[:, 1]]).all(1)
        min_violations[j] = (~sat).sum(1).min()
    assert np.all(~flags[:, :, 0] | flags[:, :, 1])
    assert np.all(~flags[:, :, 1] | flags[:, :, 2])
    return flags, min_violations
