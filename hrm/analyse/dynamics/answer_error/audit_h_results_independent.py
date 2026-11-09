"""Independent CPU audit of saved H-intervention results; never runs a model."""
from hrm.paths import analysis_directory
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
from scipy import stats
import torch

ROOT = analysis_directory(__file__)
TARGETS = ("original_excess", "ce", "mse")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def holm_six(values):
    assert len(values) == 6
    order = np.argsort(values)
    result = np.empty(6, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, float(values[index]) * (6 - position))
        result[index] = min(1.0, running)
    return result


def audit_target(target, frozen):
    directory = ROOT / "fixed_state_H" / target
    done = read(directory / "completion.json")
    protocol = read(directory / "protocol.json")
    assert done["status"] == "complete" and done["criterion"] == protocol["criterion"] == target
    assert done["n"] == 81 and done["arms"] == 55 and done["trajectories"] == 4455
    assert done["modules"] == protocol["modules"] == ["H"]
    assert done["probe"] == protocol["probe"] == "linear_H"
    assert done["primary_event_index"] == protocol["primary_event_index"] == 0
    assert done["secondary_event_index"] == protocol["secondary_event_index"] == 6
    assert protocol["events"] == ["L"] * 6 + ["H"] + ["L"] * 6 + ["H"]
    assert done["parent_protocol_sha256"] == sha(ROOT / "h_intervention_protocol.json")
    assert done["checks_sha256"] == sha(ROOT / "h_correctness_checks.json")
    for filename, digest in done["metadata_sha256"].items():
        assert sha(directory / filename) == digest, filename
    assert set(done["files"]) == {p.name for p in directory.glob("batch_*.pt")}
    parts = []
    for filename, digest in sorted(done["files"].items()):
        assert sha(directory / filename) == digest, filename
        parts.append(torch.load(directory / filename, map_location="cpu", weights_only=True))
    ids = torch.cat([part["ids"] for part in parts]).tolist()
    assert ids == protocol["included_ids"] == frozen["included_ids"]
    assert len(ids) == len(set(ids)) == 81
    norms = torch.cat([part["step_norm"] for part in parts], 1).double().numpy()
    trace = torch.cat([part["critic"] for part in parts], 1)
    features = torch.cat([part["initial"]["feature"] for part in parts], 1)
    scores = torch.cat([part["initial"]["critic"] for part in parts], 1).double().numpy()
    assert norms.shape == (55, 81, 14) and np.isfinite(norms).all()
    assert trace.shape == (55, 81, 15)
    assert torch.equal(trace[:, :, :7], trace[:, :, :1].expand(-1, -1, 7))
    assert torch.equal(trace[:, :, 7:14], trace[:, :, 7:8].expand(-1, -1, 7))
    assert torch.equal(features[:, :, :512], features[:1, :, :512].expand(55, -1, -1))
    fitted = torch.load(Path(protocol["probe_source"]) / "probes.pt", map_location="cpu", weights_only=True)["linear_H"]
    rebuilt = features[:, :, 512:].double().numpy() @ fitted["weight"].double().numpy() + float(fitted["bias"])
    np.testing.assert_allclose(rebuilt, scores, atol=1e-8, rtol=0)
    arms = protocol["arms"]
    assert len(arms) == 55 and arms[0]["kind"] == "native"
    endpoints = {}
    for endpoint, index in (("primary_next_L", 0), ("secondary_first_H", 6)):
        rows = []
        for q in frozen["doses"]:
            actual = [i for i, arm in enumerate(arms) if arm["kind"] == "critic" and arm["signed_state_displacement"] == q]
            controls = [i for i, arm in enumerate(arms) if arm["kind"] == "control" and arm["signed_state_displacement"] == q]
            assert len(actual) == 1 and len(controls) == 8
            assert {arms[i]["module"] for i in actual + controls} == {"H"}
            assert {(arms[i]["seed"], arms[i]["sign"]) for i in controls} == {(k, sign) for k in range(4) for sign in (1, -1)}
            # Average eight controls within each puzzle, then compare 81 paired values.
            difference = norms[actual[0], :, index] - norms[controls, :, index].mean(axis=0)
            assert difference.shape == (81,)
            mean = float(difference.mean())
            standard_error = float(difference.std(ddof=1) / np.sqrt(81))
            t_value = mean / standard_error
            p_value = float(2 * stats.t.sf(abs(t_value), df=80))
            scipy_result = stats.ttest_1samp(difference, 0)
            np.testing.assert_allclose([t_value, p_value], [scipy_result.statistic, scipy_result.pvalue], atol=1e-12, rtol=1e-12)
            rows.append(dict(dose=q, n=81, controls_per_puzzle=8, df=80,
                             mean_paired_control_difference=mean, t=t_value, p_t=p_value))
        adjusted = holm_six([row["p_t"] for row in rows])
        for row, value in zip(rows, adjusted):
            row["p_t_holm"] = float(value)
        endpoints[endpoint] = dict(event_index=index, event=protocol["events"][index], rows=rows)
    return dict(status="passed", target=target, subjects=81, arms=55,
                files_verified=len(parts), completion_sha256=sha(directory / "completion.json"),
                protocol_sha256=sha(directory / "protocol.json"),
                L_pooled_features_unchanged_at_injection=True,
                H_score_constant_in_L_blocks=True,
                H_score_reconstruction_max_error=float(np.max(np.abs(rebuilt-scores))),
                endpoints=endpoints)


def main():
    torch.set_num_threads(4)
    output = ROOT / "independent_H_endpoint_audit.json"
    assert not output.exists(), "Preserve previous independent audit."
    frozen = read(ROOT / "h_intervention_protocol.json")
    result = dict(status="passed", created_utc=datetime.now(timezone.utc).isoformat(),
                  model_forward=False, model_or_probe_fitting=False, script_sha256=sha(Path(__file__)),
                  protocol_sha256=sha(ROOT / "h_intervention_protocol.json"),
                  interpretation="Independent recomputation from real saved batches. Six-dose Holm families kept separate for each target and endpoint.",
                  targets={target: audit_target(target, frozen) for target in TARGETS})
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "audit": str(output)}))


if __name__ == "__main__":
    main()
