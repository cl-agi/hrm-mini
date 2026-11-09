"""Single global q -> latent-coefficient regressors, with no phase/branch inputs.

fit_global(..., method='kernel' or 'fourier') returns a model and JSON report.
model.predict(q), model.save(path), and load_global(path) support grid scoring.
All normalizers and RBF centers use training observations only. The objective
is weighted squared error in original RMS coefficient units, never HRM scores.
The map is smooth but is not identified off the observed trajectory.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


class _GlobalModel(nn.Module):
    @torch.inference_mode()
    def predict(self, q, batch_size=2048):
        # Preserve the small differences in the original double-precision plane
        # for the analytic RBF; the neural network itself computes in float32.
        q = np.asarray(q, dtype=np.float64 if self.kind == 'kernel' else np.float32)
        if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
            raise ValueError('q must be a finite [N,2] array')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not len(q):
            return np.empty((0, self.config['rank']), dtype=np.float32)
        self.eval()
        return np.concatenate([
            self(torch.as_tensor(q[i:i+batch_size], device=self.q_mean.device)).float().cpu().numpy()
            for i in range(0, len(q), batch_size)
        ])

    def save(self, path):
        torch.save(dict(format_version=1, kind=self.kind, config=self.config,
                        state_dict={k: v.detach().cpu() for k, v in self.state_dict().items()},
                        report=self.report), Path(path))


class GlobalFourier(_GlobalModel):
    kind = 'fourier'

    def __init__(self, rank, n_features=128, width=256, depth=4):
        super().__init__()
        self.config = dict(rank=int(rank), n_features=int(n_features), width=int(width), depth=int(depth))
        self.register_buffer('q_mean', torch.zeros(2))
        self.register_buffer('q_scale', torch.ones(2))
        self.register_buffer('target_mean', torch.zeros(rank))
        self.register_buffer('target_scale', torch.ones(rank))
        self.register_buffer('frequencies', torch.zeros(n_features, 2))
        layers = []
        size = 2 + 2*n_features
        for _ in range(depth):
            layers.extend([nn.Linear(size, width), nn.SiLU()])
            size = width
        layers.append(nn.Linear(size, rank))
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.report = None

    def forward(self, q):
        x = (q - self.q_mean) / self.q_scale
        angle = 2*math.pi * (x @ self.frequencies.T)
        f = torch.cat([x, angle.sin(), angle.cos()], dim=-1)
        return self.target_mean + self.target_scale*self.network(f)


class GlobalRBF(_GlobalModel):
    kind = 'kernel'

    def __init__(self, rank, n_centers, scales=(.5, 1., 2.), kernel='matern52'):
        super().__init__()
        self.config = dict(rank=int(rank), n_centers=int(n_centers),
                           scales=[float(x) for x in scales], kernel=str(kernel))
        self.register_buffer('q_mean', torch.zeros(2, dtype=torch.float64))
        self.register_buffer('q_scale', torch.ones(2, dtype=torch.float64))
        self.register_buffer('target_mean', torch.zeros(rank, dtype=torch.float64))
        self.register_buffer('centers', torch.zeros(n_centers, 2, dtype=torch.float64))
        self.register_buffer('radii', torch.ones(n_centers, dtype=torch.float64))
        size = 3 + len(scales)*n_centers
        self.register_buffer('feature_scale', torch.ones(size, dtype=torch.float64))
        self.register_buffer('beta', torch.zeros(size, rank, dtype=torch.float64))
        self.report = None

    def raw_features(self, q):
        x = (q.to(torch.float64) - self.q_mean) / self.q_scale
        # Explicit coordinate differences avoid cancellation at very close points.
        dx = x[:, None, 0] - self.centers[None, :, 0]
        dy = x[:, None, 1] - self.centers[None, :, 1]
        distance2 = dx.square() + dy.square()
        fs = [torch.ones_like(x[:, :1]), x]
        for scale in self.config['scales']:
            r2 = distance2 / (self.radii[None, :]*scale).square()
            if self.config['kernel'] == 'gaussian':
                phi = (-.5*r2).exp()
            else:
                r = (5*r2 + 1e-30).sqrt()
                phi = (1+r+r.square()/3)*(-r).exp()
            fs.append(phi)
        return torch.cat(fs, dim=1)

    def forward(self, q):
        return self.target_mean + (self.raw_features(q)/self.feature_scale) @ self.beta


def load_global(path, device='cuda'):
    ckpt = torch.load(Path(path), map_location='cpu', weights_only=True)
    if ckpt['format_version'] != 1:
        raise ValueError('Unsupported global-lift checkpoint')
    cls = GlobalRBF if ckpt['kind'] == 'kernel' else GlobalFourier
    model = cls(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    model.report = ckpt.get('report')
    return model.to(device).eval()


def _indices(value, n, name, empty=False):
    value = np.asarray(value)
    if value.dtype == np.bool_:
        if value.shape != (n,):
            raise ValueError(f'{name} mask must have length N')
        value = np.flatnonzero(value)
    if value.ndim != 1 or (value.size and not np.issubdtype(value.dtype, np.integer)):
        raise ValueError(f'{name} must be 1D integer indices or a boolean mask')
    value = value.astype(np.int64)
    if not len(value) and empty:
        return value
    if not len(value) or value.min() < 0 or value.max() >= n or len(np.unique(value)) != len(value):
        raise ValueError(f'{name} must contain distinct valid indices')
    return value


def _prepare(q, coeff, train_idx, val_idx, weights):
    q = np.asarray(q, dtype=np.float64)
    coeff = np.asarray(coeff, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 2 or coeff.ndim != 2 or len(q) != len(coeff) or coeff.shape[1] < 1:
        raise ValueError('Expected q[N,2] and coeff[N,R]')
    if not np.isfinite(q).all() or not np.isfinite(coeff).all():
        raise ValueError('Inputs must be finite')
    tr = _indices(train_idx, len(q), 'train_idx')
    va = _indices(val_idx, len(q), 'val_idx', empty=True)
    if np.intersect1d(tr, va).size:
        raise ValueError('Training and validation overlap')
    w = np.ones(len(q), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(q),) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('weights must be finite nonnegative [N]')
    if w[tr].sum() <= 0 or (len(va) and w[va].sum() <= 0):
        raise ValueError('Each nonempty split needs positive total weight')
    wt = w[tr] / w[tr].sum()
    mean = np.average(q[tr], axis=0, weights=wt)
    std = np.maximum(np.sqrt(np.average((q[tr]-mean)**2, axis=0, weights=wt)), 1e-8)
    ym = np.average(coeff[tr], axis=0, weights=wt)
    ys = np.maximum(np.sqrt(np.average((coeff[tr]-ym)**2, axis=0, weights=wt)), 1e-7)
    variance = max(float(np.average(((coeff[tr]-ym)**2).sum(1), weights=wt)), 1e-12)
    return q, coeff, tr, va, w, mean, std, ym, ys, variance


@torch.inference_mode()
def _metrics(model, q, y, w, batch_size=2048):
    if not len(q):
        return None
    errors = []
    for i in range(0, len(q), batch_size):
        errors.append((model(q[i:i+batch_size])-y[i:i+batch_size]).double().square().sum(1))
    squared = torch.cat(errors)
    if not torch.isfinite(squared).all():
        raise FloatingPointError('Nonfinite prediction')
    distance = squared.sqrt()
    return dict(weighted_coefficient_rms=float((squared*w/w.sum()).sum().sqrt()),
                coefficient_rms=float(squared.mean().sqrt()),
                mean_sample_l2=float(distance.mean()), median_sample_l2=float(distance.median()),
                p95_sample_l2=float(torch.quantile(distance, .95)), max_sample_l2=float(distance.max()))


def _fps_centers(x, weight, n_centers, seed, fraction):
    """FPS coverage plus weighted sampled centers for densely revisited small rings."""
    unique, first, inverse = np.unique(x, axis=0, return_index=True, return_inverse=True)
    uw = np.bincount(inverse, weights=weight, minlength=len(unique))
    n_centers = min(int(n_centers), len(unique))
    n_fps = min(n_centers, max(1, int(round(n_centers*fraction))))
    selected = np.zeros(len(unique), dtype=bool)
    dist2 = np.full(len(unique), np.inf)
    center = np.sum(unique * (uw/uw.sum())[:, None], axis=0)
    current = int(np.argmax(((unique-center)**2).sum(1)))
    chosen = []
    for _ in range(n_fps):
        chosen.append(current)
        selected[current] = True
        dist2 = np.minimum(dist2, ((unique-unique[current])**2).sum(1))
        dist2[selected] = -1
        current = int(np.argmax(dist2))
    remaining = n_centers-len(chosen)
    if remaining:
        rng = np.random.default_rng(seed)
        pool = np.flatnonzero(~selected)
        # A uniform component permits zero-weight points without sampling errors.
        p = .5/len(pool) + .5*uw[pool]/max(uw[pool].sum(), 1e-300)
        p /= p.sum()
        chosen.extend(rng.choice(pool, remaining, replace=False, p=p).tolist())
    return unique[chosen], first[np.asarray(chosen)], n_fps


def _fit_rbf(data, device, seed, selected, *, n_centers=1024, scales=(.5, 1., 2.),
             kernel='matern52', ridge=None, bandwidth_multiplier=1., center_neighbors=8,
             fps_fraction=.75, feature_batch_size=2048):
    q, y, tr, va, w, qm, qs, ym, ys, variance = data
    if selected:
        p = selected['selected_hyperparameters']
        n_centers, scales, kernel, ridge = p['n_centers'], p['scales'], p['kernel'], p['ridge']
        bandwidth_multiplier = p['bandwidth_multiplier']
        center_neighbors, fps_fraction = p['center_neighbors'], p['fps_fraction']
    if n_centers < 1 or center_neighbors < 1 or not 0 <= fps_fraction <= 1:
        raise ValueError('Invalid center settings')
    if kernel not in ('gaussian', 'matern52') or not len(scales) or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('Kernel must be gaussian/matern52 with positive scales')
    if not math.isfinite(bandwidth_multiplier) or bandwidth_multiplier <= 0 or feature_batch_size < 1:
        raise ValueError('Invalid bandwidth or feature batch size')
    ridges = ([1e-8, 1e-6, 1e-4] if len(va) else [1e-6]) if ridge is None else [float(ridge)]
    if any(not math.isfinite(r) or r <= 0 for r in ridges):
        raise ValueError('ridge must be finite and positive')
    centers, source, n_fps = _fps_centers((q[tr]-qm)/qs, w[tr], n_centers, seed, fps_fraction)
    model = GlobalRBF(y.shape[1], len(centers), scales, kernel).to(device)
    with torch.no_grad():
        for name, value in [('q_mean', qm), ('q_scale', qs), ('target_mean', ym), ('centers', centers)]:
            getattr(model, name).copy_(torch.as_tensor(value, dtype=torch.float64, device=device))
        # Distances between centers, not global standard deviation, set local widths.
        dx = model.centers[:, None, 0]-model.centers[None, :, 0]
        dy = model.centers[:, None, 1]-model.centers[None, :, 1]
        d2 = dx.square()+dy.square()
        d2.fill_diagonal_(math.inf)
        if len(centers) > 1:
            k = min(center_neighbors, len(centers)-1)
            radii = d2.topk(k, dim=1, largest=False).values[:, -1].sqrt()
        else:
            radii = torch.ones(1, dtype=torch.float64, device=device)
        model.radii.copy_(radii.clamp_min(1e-7)*bandwidth_multiplier)
    qt = torch.as_tensor(q[tr], dtype=torch.float64, device=device)
    yt = torch.as_tensor(y[tr], dtype=torch.float64, device=device)
    wt = torch.as_tensor(w[tr]/w[tr].sum(), dtype=torch.float64, device=device)
    qv = torch.as_tensor(q[va], dtype=torch.float64, device=device)
    yv = torch.as_tensor(y[va], dtype=torch.float64, device=device)
    wv = torch.as_tensor(w[va], dtype=torch.float64, device=device)
    size = model.beta.shape[0]
    gram = torch.zeros(size, size, dtype=torch.float64, device=device)
    rhs = torch.zeros(size, y.shape[1], dtype=torch.float64, device=device)
    with torch.inference_mode():
        for i in range(0, len(qt), feature_batch_size):
            f = model.raw_features(qt[i:i+feature_batch_size])
            fw = f * wt[i:i+len(f), None]
            gram.add_(f.T @ fw)
            rhs.add_(fw.T @ (yt[i:i+len(f)]-model.target_mean))
        model.feature_scale.copy_(gram.diag().sqrt().clamp_min(1e-8))
        gram.div_(model.feature_scale[:, None]*model.feature_scale[None, :])
        rhs.div_(model.feature_scale[:, None])
        trials, best, best_score = [], None, math.inf
        for regularization in ridges:
            system = gram.clone()
            penalty = torch.full((size,), regularization, dtype=torch.float64, device=device)
            penalty[0] = 0  # Unpenalized constant; affine slopes and RBFs are regularized.
            system.diagonal().add_(penalty)
            chol, info = torch.linalg.cholesky_ex(system)
            jitter = 0.
            if int(info.item()) != 0:
                jitter = max(regularization*1e-3, 1e-12)
                system.diagonal().add_(jitter)
                chol = torch.linalg.cholesky(system)
            beta = torch.cholesky_solve(rhs, chol)
            model.beta.copy_(beta)
            train = _metrics(model, qt, yt, wt)
            validation = _metrics(model, qv, yv, wv)
            score = (validation or train)['weighted_coefficient_rms']
            trials.append(dict(ridge=regularization, cholesky_jitter=jitter, train=train, validation=validation))
            if score < best_score:
                best_score, best = score, (beta.clone(), len(trials)-1)
        model.beta.copy_(best[0])
    trial = trials[best[1]]
    chosen = dict(n_centers=int(len(centers)), scales=list(scales), kernel=kernel, ridge=trial['ridge'],
                  bandwidth_multiplier=bandwidth_multiplier, center_neighbors=center_neighbors,
                  fps_fraction=fps_fraction)
    report = dict(selected_hyperparameters=chosen, train=trial['train'], validation=trial['validation'],
                  trials=trials, n_features=size, solve_dtype='float64',
                  center_strategy='Farthest-point coverage plus weighted unique-point sampling; one global shared basis',
                  n_farthest_centers=n_fps, center_source_train_indices=tr[source].tolist(),
                  radius_normalized_min_median_max=[float(model.radii.min()), float(model.radii.median()), float(model.radii.max())],
                  penalty='Ridge on train-RMS-normalized features; intercept unpenalized',
                  selection='Minimum weighted validation coefficient RMS' if len(va) else 'Fixed selected ridge; full-data refit')
    return model.eval(), report


def _fit_fourier(data, device, seed, selected, *, max_steps=2000, width=256, depth=4,
                 n_features=128, frequency_bands=None, batch_size=4096, learning_rate=2e-3,
                 patience=500, eval_every=100, refit_min_steps=300):
    q, y, tr, va, w, qm, qs, ym, ys, variance = data
    if selected:
        p = selected['selected_hyperparameters']
        width, depth, n_features = p['width'], p['depth'], p['n_features']
        frequency_bands = p['frequency_bands']
        batch_size, learning_rate = p['batch_size'], p['learning_rate']
        if not len(va):
            max_steps = max(refit_min_steps, int(selected['best_step']))
    if min(max_steps, width, depth, n_features, batch_size, patience, eval_every) < 1 or learning_rate <= 0:
        raise ValueError('Training settings must be positive')
    if frequency_bands is None:
        candidates = [(.5, 2., 8., 32.), (.5, 4., 32., 256.)] if len(va) else [(.5, 2., 8., 32.)]
    else:
        candidates = [tuple(float(b) for b in frequency_bands)]
    if any(not bands or any(not math.isfinite(b) or b <= 0 for b in bands) for bands in candidates):
        raise ValueError('frequency_bands must be positive finite frequencies')
    qt = torch.as_tensor(q[tr], dtype=torch.float32, device=device)
    yt = torch.as_tensor(y[tr], dtype=torch.float32, device=device)
    wt = torch.as_tensor(w[tr]/w[tr].sum(), dtype=torch.float32, device=device)
    qv = torch.as_tensor(q[va], dtype=torch.float32, device=device)
    yv = torch.as_tensor(y[va], dtype=torch.float32, device=device)
    wv = torch.as_tensor(w[va], dtype=torch.float32, device=device)
    trials, chosen, chosen_error = [], None, math.inf
    for bands in candidates:
        start = time.monotonic()
        torch.manual_seed(seed)
        model = GlobalFourier(y.shape[1], n_features, width, depth).to(device)
        g = torch.Generator(device='cpu').manual_seed(seed)
        raw = torch.randn(n_features, 2, generator=g)
        scale = torch.tensor([bands[i % len(bands)] for i in range(n_features)])
        with torch.no_grad():
            for name, value in [('q_mean', qm), ('q_scale', qs), ('target_mean', ym), ('target_scale', ys)]:
                getattr(model, name).copy_(torch.as_tensor(value, dtype=torch.float32, device=device))
            model.frequencies.copy_((raw*scale[:, None]).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        train = _metrics(model, qt, yt, wt)
        validation = _metrics(model, qv, yv, wv)
        best_error = validation['weighted_coefficient_rms'] if validation else math.inf
        best_step, best_state = 0, copy.deepcopy(model.state_dict()) if len(va) else None
        history = [dict(step=0, train=train, validation=validation)]
        order, cursor = None, len(qt)
        for step in range(1, max_steps+1):
            model.train()
            optimizer.param_groups[0]['lr'] = learning_rate*(.1+.45*(1+math.cos(math.pi*(step-1)/max_steps)))
            optimizer.zero_grad(set_to_none=True)
            if batch_size >= len(qt):
                squared = (model(qt)-yt).square().sum(1)
                loss = (squared*wt).sum()/variance
            else:
                if cursor >= len(qt):
                    order = torch.randperm(len(qt), device=device)
                    cursor = 0
                ix = order[cursor:cursor+batch_size]
                cursor += len(ix)
                squared = (model(qt[ix])-yt[ix]).square().sum(1)
                loss = (squared*wt[ix]).sum()*len(qt)/len(ix)/variance
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite Fourier loss at step {step}')
            loss.backward()
            optimizer.step()
            if step % eval_every and step != max_steps:
                continue
            model.eval()
            train = _metrics(model, qt, yt, wt)
            validation = _metrics(model, qv, yv, wv)
            history.append(dict(step=step, train=train, validation=validation))
            if validation and validation['weighted_coefficient_rms'] < best_error:
                best_error, best_step = validation['weighted_coefficient_rms'], step
                best_state = copy.deepcopy(model.state_dict())
            if validation and step-best_step >= patience:
                break
        if len(va):
            model.load_state_dict(best_state)
        else:
            best_step = step
        train = _metrics(model, qt, yt, wt)
        validation = _metrics(model, qv, yv, wv)
        score = (validation or train)['weighted_coefficient_rms']
        p = dict(width=width, depth=depth, n_features=n_features, frequency_bands=list(bands),
                 batch_size=batch_size, learning_rate=learning_rate)
        trials.append(dict(hyperparameters=p, best_step=best_step, steps_run=step, train=train,
                           validation=validation, history=history, seconds=time.monotonic()-start))
        if score < chosen_error:
            chosen, chosen_error, chosen_index = model, score, len(trials)-1
    trial = trials[chosen_index]
    report = dict(selected_hyperparameters=trial['hyperparameters'], best_step=trial['best_step'],
                  steps_run=trial['steps_run'], max_steps_per_candidate=max_steps, trials=trials,
                  train=trial['train'], validation=trial['validation'], full_batch=batch_size >= len(qt),
                  sample_coverage='Shuffled complete passes, weighted uniform minibatch objective',
                  selection='Minimum weighted validation coefficient RMS' if len(va) else 'Fixed selected frequency bands and training steps; full-data refit')
    return chosen.eval(), report


def fit_global(q, coeff, train_idx, val_idx, weights=None, method='kernel', device='cuda',
               *, seed=6566, max_steps=2000, selected=None, **kwargs):
    """Fit ONE f(x,y), returning (model, report); no phase or branch arguments.

    q[N,2], coeff[N,R] are arrays. Optional weights[N] balance observations.
    val_idx=[] disables selection. selected=an earlier report reuses its chosen
    hyperparameters; Fourier full-data refits run max(300, best_step) updates.
    RBF kwargs: n_centers=1024, scales=(.5,1,2), kernel='matern52', ridge=None,
      center_neighbors=8, bandwidth_multiplier=1, fps_fraction=.75.
    Fourier kwargs: width=256, depth=4, n_features=128, frequency_bands=None,
      batch_size=4096, learning_rate=.002, patience=500, eval_every=100.
    None selects three RBF ridge values or two Fourier frequency-band sets on
    validation. max_steps is per Fourier candidate, ignored by kernel models.
    Normalizers, FPS centers, and kernel widths see train_idx only. The external
    coefficient basis may have its own transductive fitting convention.
    """
    started = time.monotonic()
    method = str(method).lower()
    if method in ('rbf', 'matern', 'matern52'):
        method = 'kernel'
    if method in ('mlp', 'fourier_mlp'):
        method = 'fourier'
    if method not in ('kernel', 'fourier'):
        raise ValueError('method must be kernel/rbf or fourier/mlp')
    if selected and selected.get('method') != method:
        raise ValueError('selected report belongs to a different method')
    data = _prepare(q, coeff, train_idx, val_idx, weights)
    q, y, tr, va, w, qm, qs, ym, ys, variance = data
    device = torch.device(device)
    if method == 'kernel':
        model, report = _fit_rbf(data, device, seed, selected, **kwargs)
    else:
        model, report = _fit_fourier(data, device, seed, selected, max_steps=max_steps, **kwargs)
    report.update(method=method, seed=int(seed), device=str(device), torch_version=str(torch.__version__),
                  n_observations=len(q), n_train=len(tr), n_validation=len(va), rank=y.shape[1],
                  q_mean_train=qm.tolist(), q_scale_train=qs.tolist(),
                  objective='Weighted mean squared coefficient L2 in original RMS units; no quality-score loss',
                  loss_scale_train=variance, no_phase_or_branch_inputs=True,
                  selected_refit=bool(selected), validation_is_independent_final_test=False,
                  off_path='Smooth extension prior from the regression architecture; not observed ground truth',
                  seconds=time.monotonic()-started)
    model.report = report
    return model, report


def _smoke_test():
    import tempfile
    torch.set_num_threads(2)
    rng = np.random.default_rng(6566)
    q = rng.uniform(-1, 1, (160, 2))
    y = np.column_stack([.1*q[:, 0], .05*np.sin(2*q[:, 1]), .02*q[:, 0]*q[:, 1]]).astype(np.float32)
    tr, va = np.arange(120), np.arange(120, 160)
    w = np.linspace(.5, 1.5, len(q))
    for method, options in [('kernel', dict(n_centers=32, scales=(.5, 1., 2.))),
                            ('fourier', dict(width=32, depth=2, n_features=16, frequency_bands=(.5, 1.),
                                             batch_size=32, max_steps=250, eval_every=50))]:
        model, report = fit_global(q, y, tr, va, w, method, 'cpu', **options)
        np.testing.assert_allclose(model.q_mean.cpu(), np.average(q[tr], axis=0, weights=w[tr]), rtol=1e-6, atol=1e-7)
        assert report['validation']['weighted_coefficient_rms'] < .025, report
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            model.save(path)
            restored = load_global(path, 'cpu')
            np.testing.assert_allclose(model.predict(q), restored.predict(q), rtol=0, atol=0)
        assert model.predict(np.empty((0, 2))).shape == (0, 3)
        refit, rr = fit_global(q, y, np.arange(len(q)), [], w, method, 'cpu', selected=report,
                              **({'refit_min_steps': 1} if method == 'fourier' else {}))
        assert rr['validation'] is None and rr['selected_refit']
        assert refit.predict(q[:3]).shape == (3, 3)
        print(method, 'passed', report['validation'])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        ap.print_help()
