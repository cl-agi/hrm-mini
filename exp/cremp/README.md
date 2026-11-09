# Molecular conformations (CREMP)

[Experiment guide](../../entry.md)

## Published HRM checkpoint

Install the CREMP and Ringer packages in the environment containing HRM:

```bash
python -m pip install -e ./third_party/ringer -e ./third_party/cremp
```

Generate conformations, reconstruct coordinates and score:

```bash
hrm evaluate --task cremp --k 64 --output "$HRM_WORKDIR/cremp"
python -m hrm_cremp.project_angles --samples "$HRM_WORKDIR/cremp/generated_angles.npz" --output "$HRM_WORKDIR/cremp-projected"
python -m hrm.evaluation.cremp_score --projected "$HRM_WORKDIR/cremp-projected" --filter combined --output "$HRM_WORKDIR/cremp-scores.json"
```

The last command uses all reference conformers and the combined physical-validity and reconstruction-drift filter for the reference comparison. Use `--filter none` for unfiltered scores, or `drift` and `physical` for individual filters. All requested candidates remain counted, including failures. For a CPU engineering check use `--device cpu --limit 1 --k 1 --batch-molecules 1` during generation, and `--reference-cap 2` during scoring. This command reports scores for the released checkpoint.

## Train an HRM denoiser

```bash
python -m hrm_cremp.train_angles --family hrm --output "$HRM_WORKDIR/cremp-training" \
  --steps 20000 --batch 64 --width 384 --intermediate 512 --heads 12 \
  --device cuda --cyclic-augment --max-seconds 86400 --no-final-generation
```

This uses HF training and validation data by default and freezes a new timestamp seed. Inspect `manifest.json` for the new run’s seed and resolved settings. Check that `summary.json` reports 20,000 completed updates; a wall-time-limited run is incomplete. The final checkpoint is `last.pt`; `best.pt` is selected by validation denoising loss. The fixed-final-step comparisons use `last.pt` after completing the specified budget. Use `--checkpoint "$HRM_WORKDIR/cremp-training/last.pt"` with `hrm_cremp.generate_angles` for subsequent sampling. The training geometry is retained with the checkpoint and carried into projection automatically.

## Classical comparison

ETKDGv3 + MMFF94 is an optional RDKit baseline in `hrm.analyse.cremp.classical`. It generates conformers with RDKit. Match molecule split, sample count, filtering and scoring to the HRM evaluation. Run it through the CREMP comparison workflow.
