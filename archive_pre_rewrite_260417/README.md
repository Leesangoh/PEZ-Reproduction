# PEZ Reproduction

Reproduction code for the paper `Interpreting Physics in Video World Models` (`arXiv:2602.07050`).

This repository contains a three-step pipeline:

1. `step1_generate.py`
   Builds the synthetic ball-motion datasets used for probing.
   The current implementation supports:
   - Kubric + Blender rendering
   - PyBullet trajectories at 240 Hz
   - 10 simulation substeps per rendered frame
   - fresh start-position sampling for each `(direction, speed)` and `(direction, acceleration)` pair

2. `step2_extract.py` / `step2_extract_raw.py` / `step2_extract_preblock.py`
   Extracts layer-wise V-JEPA 2 features from the generated videos. Three
   variants are provided:
   - `step2_extract.py`: default path, uses V-JEPA 2's built-in `out_layers` API
     (applies the final LayerNorm to each captured layer).
   - `step2_extract_raw.py`: captures the residual stream *after* each block
     without the final LayerNorm.
   - `step2_extract_preblock.py`: captures the residual stream *before* each
     block. This matches the paper's layer-numbering convention (layer 0 is the
     patch-embedding output, before any transformer block), which is what the
     PEZ figures in the paper are indexed against. Use this variant for faithful
     reproduction of Figure 2b/2c.

3. `step3_probe.py`
   Runs grouped 5-fold linear probes and generates PEZ-style figures.
   Supported probe sets:
   - `polar`: speed, direction, acceleration magnitude
   - `cartesian`: `(vx, vy)` and `(ax, ay)`

## Repository Layout

- `constants.py`: shared constants and local paths
- `step1_generate.py`: dataset generation
- `step2_extract.py`: feature extraction (final-LN variant)
- `step2_extract_raw.py`: post-block raw residual stream
- `step2_extract_preblock.py`: pre-block raw residual stream (paper convention)
- `step3_probe.py`: probing and figure generation

Large generated outputs are intentionally excluded from git. By default they live under:

- `artifacts/data/kubric_data`
- `artifacts/features`
- `artifacts/results`

External dependencies expected by the scripts:

- `/home/solee/kubric`
- `/home/solee/vjepa2`

## Typical Workflow

Step 1 inside Blender:

```bash
PYTHONPATH=/home/solee/kubric blender --background \
  --python /home/solee/pez/step1_generate.py -- --backend kubric
```

Step 2 feature extraction:

```bash
CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh /home/solee/pez/step2_extract.py --models large
CUDA_VISIBLE_DEVICES=1 /isaac-sim/python.sh /home/solee/pez/step2_extract.py --models giant

# For paper-faithful PEZ reproduction (pre-block convention):
CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh /home/solee/pez/step2_extract_preblock.py --models large
```

Step 3 probing:

```bash
/isaac-sim/python.sh /home/solee/pez/step3_probe.py --models vjepa2_L vjepa2_G --probe-set polar --solver trainable --grouping condition
/isaac-sim/python.sh /home/solee/pez/step3_probe.py --models vjepa2_L vjepa2_G --probe-set cartesian --solver trainable --grouping condition
```

## Current Status

This reproduction has already fixed several major mismatches against the paper:

- analytic trajectories replaced with PyBullet simulation
- fixed global start positions replaced with fresh per-condition sampling
- grouped 5-fold probe evaluation added
- trainable and ridge linear-probe solvers both supported

The main open question is whether the corrected dataset and probe protocol recover the paper's PEZ-style late emergence of polar direction information, especially for `V-JEPA 2-L` and `V-JEPA 2-G`.

### Partial reproduction — ViT-L result (2026-04-17)

This reproduction recovers the paper's PEZ pattern **only at layer 0** (patch
embedding). The paper's quantitative claim of a *sharp mid-depth transition*
at ~1/3 of the network's depth is **not** reproduced on our dataset, regardless
of probe strength.

#### Phase 1 — layer-indexing fix (pre-block convention)

Using `step2_extract_preblock.py` with the batched trainable probe
(`--solver trainable --grouping position --probe-set polar`):

| Layer | Speed R² | Direction R² | Accel R² |
|-------|----------|--------------|----------|
| 0 (patch_embed) | 0.254 | **0.087** | -0.037 |
| 1 (post-block-0) | 0.960 | 0.699 | 0.952 |
| 3 | 0.964 | 0.933 | 0.952 |
| 8 | 0.986 | 0.985 | 0.979 |
| 23 | 0.992 | 0.996 | 0.989 |

At layer 0 (patch-embed only, no transformer blocks applied):
- direction R² ≈ 0 — matches the paper at that layer
- speed R² is small but already above direction — matches the paper's
  speed-before-direction ordering
- acceleration R² ≈ 0 — matches the paper

But already at **layer 1** (just one transformer block beyond patch_embed) the
direction R² jumps to ~0.70 — i.e. the transition is almost entirely inside the
*first* block rather than across the first ~1/3 of the network.

#### Phase 2 — probe strength check (weak probes)

To rule out the possibility that our strong probe (AdamW over a 5×4 HP grid,
400 epochs with early stopping) was artificially inflating R² at early layers,
we reran the pipeline with two deliberately weaker probes:

- `--solver ridge_weak`: closed-form ridge, α ∈ {1, 10, 100}
- `--solver adamw100`: single-HP AdamW (lr=1e-3, wd=0.1), 100 epochs, patience 10

Same seed, same position grouping, same folds. 2×2 grid (feature convention ×
weak probe) saved as `artifacts/results/pez_reproduction_vitl_phase2_2x2.png`.

Phase 1 decision criteria evaluated on the pre-block × weak-probe runs:

| Criterion | Target | pre+trainable | pre+ridge_weak | pre+adamw100 |
|-----------|--------|----------------|----------------|--------------|
| L0 direction | ≈ 0.05–0.10 | 0.087 ✓ | 0.040 ✓ | 0.038 ✓ |
| L1–L6 direction mean | < 0.4 | 0.897 ✗ | 0.863 ✗ | 0.838 ✗ |
| L0 speed | 0.0–0.3 | 0.254 ✓ | 0.174 ✓ | 0.206 ✓ |
| Direction first ≥ 0.8 | L7–L10 | L2 ✗ | L2 ✗ | L3 ✗ |
| **Score** |  | **2/4** | **2/4** | **2/4** |

Weak probes drop absolute R² by ~1–3 % but leave the curve *shape* essentially
unchanged. Block 0 is still doing the bulk of the direction emergence.
Probe strength is therefore **not** the dominant cause of the mismatch.

#### Phase 3 — leakage check (direction-grouped CV)

Position grouping in Phases 1–2 still allows the same direction angle to appear
in both train and val folds, so the probe can memorize per-angle features.
Phase 3 switches to GroupKFold with `grouping=direction`.

Pure LOO (`--cv-splits 8`) turned out to be ill-posed for the direction probe:
with 8 angles / 8 folds, each val fold contains a single angle, so the
`(sin θ, cos θ)` target is constant within that fold, `ss_tot = 0`, and R² is
undefined (the batched trainable produced -3.7e12; adamw100 returned 0.000 by
its edge-case handler). The 8-fold CSVs were moved to `artifacts/results/archive/`.

The valid setup is `--grouping direction --cv-splits 4` — each val fold now holds
out 2 of the 8 angles, so the targets still vary inside the val fold but no
held-out angle ever appears in training.

Direction-R² summary (pre-block features):

| probe / grouping | L0 | L1–L6 mean | L3 | L8 | first ≥ 0.8 |
|------------------|----|------------|----|----|-------------|
| trainable / position 5-fold | 0.087 | 0.897 | 0.933 | 0.985 | L2 |
| trainable / direction 4-fold | **−0.220** | 0.806 | 0.804 | 0.974 | L3 |
| adamw100 / position 5-fold | 0.038 | 0.838 | 0.855 | 0.966 | L3 |
| adamw100 / direction 4-fold | **−0.300** | 0.688 | 0.609 | 0.955 | **L5** |

Switching to direction grouping:
- drops direction R² at every layer (so some of the earlier signal was
  angle-memorization),
- also drops **speed** R² at layer 0 (trainable: 0.254 → 0.070, adamw100:
  0.206 → 0.036), consistent with speed and direction being entangled in the
  raw patch embedding,
- shifts the direction "first ≥ 0.8" layer right — from L2 (trainable, position)
  to L5 (adamw100, direction grouping). That is closer to the paper's ~L8
  target but still does not reach the paper's ~1/3-depth transition.

#### Current verdict (Phases 1–3)

Combining the indexing fix (Phase 1), probe-strength check (Phase 2), and
direction-grouped CV (Phase 3):

- L0 qualitative pattern reproduces (direction ≈ 0, speed low, accel ≈ 0).
- Mid-depth sharp transition **does not** land at ~1/3 depth. The best run
  (adamw100 + direction grouping) puts the direction `first ≥ 0.8` layer at L5
  on a 24-layer ViT-L. Paper target is L7–L10.

Remaining hypothesis to test:

**Scene complexity (Phase 4)**: our OpenCV/Kubric renders are visually much
simpler than the paper's Kubric scenes (no texture, no HDRI, single blue ball
on gray floor), so a single transformer block may plausibly suffice to
separate motion from appearance. If so, the "failure" is real model behavior
on our dataset rather than a reproduction bug.

## Notes

- The local paper PDF is included in this repository as `pez_paper.pdf`.
- Generated datasets, features, and figures are not committed by default because they are large and change frequently during reruns.
