"""Matched critic interventions; FP64 geometry followed by one native cast.

The covariance is the training pooled-feature covariance, not its inverse.
The caller supplies every acceptance tolerance; this module never calibrates,
repairs, retries, or removes an arm. A false ``passed`` value is a failed gate.
Only torch and the standard library are imported; there are no model imports.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Tolerances:
    """All bounds are required, nonnegative, and applied to every token/case.

    ``relative_chord_absolute`` also bounds pairwise arm dose mismatch.
    Control probe bounds apply to both each token's projection and the pooled
    readout. Normalization divides probe drift by ||w|| * alpha * state norm
    (mean token norm for the pooled readout), never by the baseline score.
    """

    token_norm_relative: float
    relative_chord_absolute: float
    control_probe_absolute: float
    control_probe_normalized: float

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _state(s):
    if s.ndim != 3 or min(s.shape) == 0 or not s.is_floating_point():
        raise ValueError("state must be a nonempty floating [B,S,D] tensor")
    sd = s.double()
    if not bool(torch.isfinite(sd).all()) or bool((sd.norm(dim=-1) == 0).any()):
        raise ValueError("Nonfinite or zero token state; retain cohort and diagnose")
    return sd


def _weight(s, w):
    if w.shape != (s.shape[-1],) or not w.is_floating_point():
        raise ValueError("w must be floating [D]")
    wd = w.to(device=s.device, dtype=torch.float64)
    if not bool(torch.isfinite(wd).all()) or bool(wd.norm() == 0):
        raise ValueError("Nonfinite or zero critic weight")
    return wd


def _alpha(alpha):
    if not math.isfinite(alpha) or not 0 < alpha < 2:
        raise ValueError("alpha must be a finite relative chord strictly between 0 and 2")
    return float(alpha)


def _sign(sign):
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")


def _unit(v, name):
    norm = v.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm == 0).any()):
        raise ValueError(f"Degenerate {name}; retain cohort and diagnose")
    return v / norm


def tangent(s, v):
    """Project v onto each token state's tangent plane, in FP64."""
    sd = _state(s)
    vd = v.to(device=sd.device, dtype=torch.float64)
    return vd - sd * (sd * vd).sum(-1, keepdim=True) / sd.square().sum(-1, keepdim=True)


def directions(s, w, covariance, randomseed):
    """Constraint geometry: normalize P_s((P_s(w/S)) C); four critic-null rays.

    Each random ray uses one seeded channel vector shared across tokens/cases,
    then projects away both the state and the tangent critic gradient. Seeds
    are randomseed+k: the supplied seed is the first control's actual seed.
    """
    sd = _state(s)
    wd = _weight(sd, w)
    if sd.shape[-1] < 3:
        raise ValueError("At least three channels are needed for critic-null controls")
    if covariance.shape != (sd.shape[-1], sd.shape[-1]):
        raise ValueError("covariance must have shape [D,D]")
    cov = covariance.to(device=sd.device, dtype=torch.float64)
    if not bool(torch.isfinite(cov).all()):
        raise ValueError("covariance must be finite")
    pg = tangent(sd, wd / sd.shape[1])
    _unit(pg, "critic tangent")
    result = {"critic": _unit(tangent(sd, pg @ cov), "covariance critic direction")}
    for k in range(4):
        gen = torch.Generator(device="cpu").manual_seed(int(randomseed) + k)
        ray = torch.randn(sd.shape[-1], generator=gen, dtype=torch.float64).to(sd.device)
        r = tangent(sd, ray)
        r = r - pg * (r * pg).sum(-1, keepdim=True) / pg.square().sum(-1, keepdim=True)
        result[f"random{k}"] = _unit(r, f"random{k} direction")
    return result


def rotate(s, v, alpha, sign):
    """FP64 norm-preserving critic rotation with per-token chord alpha."""
    sd = _state(s)
    theta = 2 * math.asin(_alpha(alpha) / 2)
    _sign(sign)
    return sd * math.cos(theta) + sign * sd.norm(dim=-1, keepdim=True) * v.double() * math.sin(theta)


def score_fixed_control(s, r, w, theta, sign):
    """FP64 control preserving each token's critic projection and norm.

    This is the reference ``score_fixed_control`` construction. Theta can be
    a scalar or [B] tensor; no clipping of an unreachable chord is allowed.
    """
    sd = _state(s)
    wd = _weight(sd, w)
    _sign(sign)
    t = torch.as_tensor(theta, dtype=torch.float64, device=sd.device)
    if t.ndim == 0:
        t = t.expand(sd.shape[0])
    if t.shape != (sd.shape[0],) or not bool(torch.isfinite(t).all()):
        raise ValueError("theta must be a finite scalar or [B] tensor")
    if bool(((t.abs() <= 0) | (t.abs() >= math.pi)).any()):
        raise ValueError("theta magnitude must be strictly between zero and pi")
    u = wd / wd.norm()
    parallel = (sd @ u)[..., None] * u
    perpendicular = sd - parallel
    radius = perpendicular.norm(dim=-1, keepdim=True)
    chord_half = (t.abs() / 2).sin()[:, None, None] * sd.norm(dim=-1, keepdim=True)
    if bool((radius == 0).any()) or bool((chord_half >= radius).any()):
        raise ValueError("Cannot match chord in critic-null subspace; retain cohort and diagnose")
    angle = 2 * (chord_half / radius).asin()
    return parallel + perpendicular * angle.cos() + sign * radius * r.double() * angle.sin()


def check_arm(original, altered, w, alpha, *, kind, sign, tolerances):
    """Return actual FP64 measurements of a reference or already-cast arm.

    ``kind`` is critic or control. Critic arms must move the pooled readout
    strictly in their requested sign. Controls must preserve every token's
    projection. Raw measurements and failed case/token masks are retained.
    """
    if not isinstance(tolerances, Tolerances):
        raise TypeError("Explicit Tolerances are required")
    sd = _state(original)
    wd = _weight(sd, w)
    alpha = _alpha(alpha)
    _sign(sign)
    if kind not in ("critic", "control") or altered.shape != sd.shape:
        raise ValueError("kind must be critic/control and altered must match original shape")
    ad = altered.to(device=sd.device, dtype=torch.float64)
    delta = ad - sd
    before = sd.norm(dim=-1)
    after = ad.norm(dim=-1)
    token_chord = delta.norm(dim=-1) / before
    norm_error = (after - before) / before
    token_probe_delta = delta @ wd
    probe_before, probe_after = sd.mean(1) @ wd, ad.mean(1) @ wd
    probe_delta = delta.mean(1) @ wd
    token_probe_normalized = token_probe_delta / (wd.norm() * alpha * before)
    probe_normalized = probe_delta / (wd.norm() * alpha * before.mean(1))
    pooled_chord = delta.flatten(1).norm(dim=-1) / sd.flatten(1).norm(dim=-1)
    failures = {
        "nonfinite": ~torch.isfinite(ad).all(-1),
        "token_norm": norm_error.abs() > tolerances.token_norm_relative,
        "token_chord": (token_chord - alpha).abs() > tolerances.relative_chord_absolute,
        "pooled_chord": (pooled_chord - alpha).abs() > tolerances.relative_chord_absolute,
    }
    if kind == "control":
        failures.update(
            control_token_probe_absolute=token_probe_delta.abs() > tolerances.control_probe_absolute,
            control_pooled_probe_absolute=probe_delta.abs() > tolerances.control_probe_absolute,
            control_token_probe_normalized=token_probe_normalized.abs() > tolerances.control_probe_normalized,
            control_pooled_probe_normalized=probe_normalized.abs() > tolerances.control_probe_normalized,
        )
    else:
        failures["critic_sign"] = ~(sign * probe_delta > 0)
    return dict(
        passed=not any(bool(mask.any()) for mask in failures.values()), failures=failures,
        token_norm_before=before, token_norm_after=after, token_norm_relative_error=norm_error,
        token_relative_displacement=token_chord, token_relative_displacement_error=token_chord-alpha,
        pooled_relative_displacement=pooled_chord, probe_before=probe_before, probe_after=probe_after,
        probe_delta=probe_delta, probe_delta_normalized=probe_normalized,
        token_probe_delta=token_probe_delta, token_probe_delta_normalized=token_probe_normalized,
    )


def _matching(arms, check_name, tolerances):
    chords = torch.stack([arm[check_name]["token_relative_displacement"] for arm in arms.values()])
    mismatch = chords.max(0).values - chords.min(0).values
    failed = ~torch.isfinite(mismatch) | (mismatch > tolerances.relative_chord_absolute)
    return dict(passed=not bool(failed.any()), token_chord_pairwise_span=mismatch, failed_tokens=failed)


@torch.no_grad()
def generate_arms(state, w, cov, alpha, randomseed, *, tolerances):
    """Generate critic +/- and four random +/- controls, with strict audits.

    Returns ``arms[name]`` with ``state`` (input dtype), ``reference`` (FP64),
    both checks, and native-minus-reference probe/chord/norm errors. ``passed`` is
    false if any constraint or pairwise chord match fails. All ten arms remain
    present on failure. Unreachable geometry raises ValueError; there is no
    tolerance default, search, correction after casting, or cohort filtering.
    Supply the actual native intervention state; BF16 inputs stay BF16 and
    FP32 inputs stay FP32, without changing the geometry or acceptance bounds.
    """
    if not isinstance(tolerances, Tolerances):
        raise TypeError("Explicit Tolerances are required")
    alpha = _alpha(alpha)
    sd = _state(state)
    wd = _weight(sd, w)
    vectors = directions(sd, wd, cov, randomseed)
    theta = 2 * math.asin(alpha / 2)
    arms = {}
    for direction, vector in vectors.items():
        kind = "critic" if direction == "critic" else "control"
        for sign, suffix in [(1, "plus"), (-1, "minus")]:
            reference = (rotate(sd, vector, alpha, sign) if kind == "critic" else
                         score_fixed_control(sd, vector, wd, theta, sign))
            native = reference.to(state.dtype)
            reference_check = check_arm(sd, reference, wd, alpha, kind=kind, sign=sign, tolerances=tolerances)
            native_check = check_arm(sd, native, wd, alpha, kind=kind, sign=sign, tolerances=tolerances)
            arms[f"{direction}_{suffix}"] = dict(
                kind=kind, direction=direction, sign=sign, state=native, reference=reference,
                reference_check=reference_check, native_check=native_check,
                cast_probe_delta_error=native_check["probe_delta"]-reference_check["probe_delta"],
                cast_token_probe_delta_error=native_check["token_probe_delta"]-reference_check["token_probe_delta"],
                cast_token_chord_error=native_check["token_relative_displacement"]-reference_check["token_relative_displacement"],
                cast_token_norm_error=native_check["token_norm_after"]-reference_check["token_norm_after"],
            )
    matching = {name: _matching(arms, name, tolerances) for name in ("reference_check", "native_check")}
    passed = (all(arm[name]["passed"] for arm in arms.values() for name in matching)
              and all(check["passed"] for check in matching.values()))
    return dict(passed=passed, alpha=alpha, randomseed=int(randomseed), arms=arms,
                matching=matching, tolerances=vars(tolerances).copy(), native_dtype=str(state.dtype).removeprefix("torch."))
