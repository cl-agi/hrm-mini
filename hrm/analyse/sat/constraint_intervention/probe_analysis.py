"""CPU-only ridge probes and puzzle-level uncertainty for the SAT critic study.

Ridge minimizes ||X_c w - y_c||² + lambda_primal ||w||², with
lambda_primal = alpha * trace(X_c.T @ X_c) / n_train. This matches the old
SAT probe's K = X_c @ X_c.T / D and lambda_dual = alpha * trace(K) / n:
lambda_primal = D * lambda_dual. Centering, covariance and coefficients use
training rows only. Validation selects alpha; there is no validation refit.

Bootstrap functions resample whole puzzles and keep the fitted probe fixed.
They measure evaluation-sample uncertainty, not uncertainty from probe fitting.
All bootstrap counts and seeds are explicit arguments. Arrays returned by the
fit are intended for NPZ storage; reported metrics are JSON-safe Python values.
"""

import numpy as np
from scipy import linalg, stats


DEFAULT_ALPHAS = np.logspace(-6, 4, 21)


def _vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty, finite vector")
    return value


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or 0 in value.shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty, finite matrix")
    return value


def regression_metrics(y, prediction, *, include_spearman=False):
    """Pooled row metrics; undefined correlations/R² are returned as None."""
    y = _vector(y, "y")
    prediction = _vector(prediction, "prediction")
    if y.shape != prediction.shape:
        raise ValueError("y and prediction must have the same shape")
    yc = np.zeros_like(y) if np.all(y == y[0]) else y - y.mean()
    pc = np.zeros_like(prediction) if np.all(prediction == prediction[0]) else prediction - prediction.mean()
    yy, pp = float(yc @ yc), float(pc @ pc)
    error = y - prediction
    sse = float(error @ error)
    result = {
        "n": int(y.size),
        "pearson": float(np.clip((yc @ pc) / np.sqrt(yy * pp), -1, 1))
        if yy > 0 and pp > 0 else None,
        "r2": float(1 - sse / yy) if yy > 0 else None,
        "rmse": float(np.sqrt(sse / y.size)),
    }
    if include_spearman:
        result["spearman"] = float(stats.spearmanr(y, prediction).statistic) \
            if yy > 0 and pp > 0 else None
    return result


def fit_ridge_probe(train_x, train_y, valid_x, valid_y, *, alphas=None):
    """Select ridge alpha on validation MSE, retaining train-only coefficients.

    ``covariance`` is the unbiased centered training covariance (denominator
    n_train - 1), suitable for a covariance-preconditioned intervention.
    The intercept is in the original feature coordinates: X @ weight + intercept.
    Ties in validation MSE select the first supplied alpha (default: smallest).
    There is deliberately no test-data argument.
    """
    train_x, valid_x = _matrix(train_x, "train_x"), _matrix(valid_x, "valid_x")
    train_y, valid_y = _vector(train_y, "train_y"), _vector(valid_y, "valid_y")
    n, d = train_x.shape
    if n < 2 or train_y.size != n or valid_y.size != len(valid_x):
        raise ValueError("training needs at least two rows and matching target lengths")
    if valid_x.shape[1] != d:
        raise ValueError("training and validation feature dimensions must match")
    alphas = _vector(DEFAULT_ALPHAS if alphas is None else alphas, "alphas")
    if (alphas <= 0).any():
        raise ValueError("alphas must be positive")
    mean = train_x.mean(axis=0)
    y_mean = float(train_y[0] if np.all(train_y == train_y[0]) else train_y.mean())
    xc, yc = train_x - mean, train_y - y_mean
    gram = xc.T @ xc
    covariance = gram / (n - 1)
    scale = float(np.trace(gram) / n)
    if scale == 0:
        weights = np.zeros((d, len(alphas)), dtype=np.float64)
    else:
        eigenvalues, eigenvectors = linalg.eigh(gram, check_finite=False)
        rhs = eigenvectors.T @ (xc.T @ yc)
        weights = eigenvectors @ (
            rhs[:, None] / (np.maximum(eigenvalues, 0)[:, None] + scale * alphas)
        )
    predictions = (valid_x - mean) @ weights + y_mean
    losses = np.mean((predictions - valid_y[:, None]) ** 2, axis=0)
    selected = int(np.argmin(losses))
    weight = weights[:, selected]
    return {
        "weight": weight,
        "intercept": float(y_mean - mean @ weight),
        "mean": mean,
        "covariance": covariance,
        "alpha": float(alphas[selected]),
        "lambda_primal": float(scale * alphas[selected]),
        "n_train": int(n),
        "n_features": int(d),
        "validation_metrics": regression_metrics(valid_y, predictions[:, selected]),
        "validation_mse_by_alpha": [
            {"alpha": float(alpha), "mse": float(loss)}
            for alpha, loss in zip(alphas, losses)
        ],
    }


def predict_probe(probe, x):
    x = _matrix(x, "x")
    weight = _vector(probe["weight"], "weight")
    if x.shape[1] != weight.size:
        raise ValueError("prediction feature dimension does not match the probe")
    return x @ weight + float(probe["intercept"])


def _groups(puzzle_ids, n_rows):
    ids = np.asarray(puzzle_ids)
    if ids.ndim != 1 or len(ids) != n_rows:
        raise ValueError("puzzle_ids must provide one ID per row")
    unique, inverse = np.unique(ids, return_inverse=True)
    return unique, inverse


def _bootstrap_setup(n_bootstrap, seed, confidence):
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be an explicit nonnegative integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    return np.random.default_rng(seed)


def _ci(values, confidence):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return None
    tail = (1 - confidence) / 2
    return np.quantile(finite, [tail, 1 - tail]).tolist()


def _metrics_from_moments(moments):
    n, sy, sp, syy, spp, syp, sse = moments.T
    yy = np.maximum(syy - sy * sy / n, 0)
    pp = np.maximum(spp - sp * sp / n, 0)
    # A resample can be constant despite a nonconstant original sample. Treat
    # cancellation-scale residual variance as zero instead of reporting noise.
    eps = 16 * np.finfo(np.float64).eps
    yy[yy <= eps * np.maximum(np.abs(syy), np.abs(sy * sy / n))] = 0
    pp[pp <= eps * np.maximum(np.abs(spp), np.abs(sp * sp / n))] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = (syp - sy * sp / n) / np.sqrt(yy * pp)
        r2 = 1 - sse / yy
    pearson[(yy == 0) | (pp == 0)] = np.nan
    r2[yy == 0] = np.nan
    return {"pearson": np.clip(pearson, -1, 1), "r2": r2,
            "rmse": np.sqrt(np.maximum(sse, 0) / n)}


def cluster_bootstrap_metrics(y, prediction, puzzle_ids, *, n_bootstrap, seed,
                              confidence=0.95):
    """Percentile CIs from whole-puzzle resampling of fixed predictions.

    Puzzles are sampled uniformly with replacement; every row in a sampled
    puzzle is retained. Metrics pool those rows (so unequal puzzle lengths keep
    their row weights). Seven grouped sufficient statistics avoid recreating
    trajectories for each bootstrap draw. Spearman is not bootstrapped because
    it cannot be reconstructed from these grouped moments.
    """
    y, prediction = _vector(y, "y"), _vector(prediction, "prediction")
    estimate = regression_metrics(y, prediction)
    unique, inverse = _groups(puzzle_ids, len(y))
    rng = _bootstrap_setup(n_bootstrap, seed, confidence)
    n_groups = len(unique)
    yc, pc = y - y.mean(), prediction - prediction.mean()
    row_moments = np.column_stack((np.ones(len(y)), yc, pc, yc * yc,
                                   pc * pc, yc * pc, (y - prediction) ** 2))
    grouped = np.zeros((n_groups, 7), dtype=np.float64)
    np.add.at(grouped, inverse, row_moments)
    sampled = {key: np.empty(n_bootstrap) for key in ("pearson", "r2", "rmse")}
    probabilities = np.full(n_groups, 1 / n_groups)
    for start in range(0, n_bootstrap, 512):
        stop = min(start + 512, n_bootstrap)
        counts = rng.multinomial(n_groups, probabilities, size=stop - start)
        metrics = _metrics_from_moments(counts @ grouped)
        for key in sampled:
            sampled[key][start:stop] = metrics[key]
    return {
        "estimate": estimate,
        "ci": {key: _ci(value, confidence) for key, value in sampled.items()},
        "valid_bootstrap_replicates": {
            key: int(np.isfinite(value).sum()) for key, value in sampled.items()
        },
        "n_puzzles": int(n_groups), "n_bootstrap": int(n_bootstrap),
        "seed": int(seed), "confidence": float(confidence),
        "resampling_unit": "puzzle", "probe_refitted": False,
    }


def paired_puzzle_effect(control, treatment, puzzle_ids, *, n_bootstrap, seed,
                         confidence=0.95):
    """Average treatment-minus-control effect and inference across puzzles.

    Multiple paired rows within a puzzle are averaged first. Each puzzle then
    receives equal weight. A negative effect means improvement for quantities
    such as violated-clause count. The two-sided t-test is performed on puzzle
    means, not on the individual correlated rows.
    """
    control, treatment = _vector(control, "control"), _vector(treatment, "treatment")
    if control.shape != treatment.shape:
        raise ValueError("control and treatment must have the same shape")
    unique, inverse = _groups(puzzle_ids, len(control))
    rng = _bootstrap_setup(n_bootstrap, seed, confidence)
    n_groups = len(unique)
    counts = np.bincount(inverse)
    effects = np.bincount(inverse, weights=treatment - control) / counts
    bootstrap = np.empty(n_bootstrap)
    probabilities = np.full(n_groups, 1 / n_groups)
    for start in range(0, n_bootstrap, 512):
        stop = min(start + 512, n_bootstrap)
        multiplicities = rng.multinomial(n_groups, probabilities, size=stop - start)
        bootstrap[start:stop] = multiplicities @ effects / n_groups
    if n_groups < 2:
        statistic, pvalue = None, None
    elif np.all(effects == effects[0]):
        statistic, pvalue = None, 1.0 if effects[0] == 0 else 0.0
    else:
        test = stats.ttest_1samp(effects, popmean=0)
        statistic, pvalue = float(test.statistic), float(test.pvalue)
    return {
        "mean_effect": float(effects.mean()), "ci": _ci(bootstrap, confidence),
        "puzzle_ids": unique.tolist(), "puzzle_effects": effects.tolist(),
        "n_puzzles": int(n_groups), "ttest_statistic": statistic,
        "ttest_pvalue": pvalue, "ttest_alternative": "two-sided",
        "effect_definition": "treatment - control; average within puzzle, then across puzzles",
        "n_bootstrap": int(n_bootstrap), "seed": int(seed),
        "confidence": float(confidence),
    }


def holm_adjust(p_values):
    """Holm adjusted p-values, retaining order and undefined values as None.

    Call separately with all six prespecified doses for each module. Undefined
    tests remain in the family size (conservatively equivalent to p=1 for the
    other tests), rather than silently shrinking the correction family.
    """
    p_values = list(p_values)
    finite = []
    for i, value in enumerate(p_values):
        if value is None:
            continue
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("p-values must be in [0, 1] or None")
        finite.append((float(value), i))
    adjusted = [None] * len(p_values)
    running = 0.0
    for rank, (pvalue, index) in enumerate(sorted(finite)):
        running = max(running, min(1.0, (len(p_values) - rank) * pvalue))
        adjusted[index] = running
    return adjusted
