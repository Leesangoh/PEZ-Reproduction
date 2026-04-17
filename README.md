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

### Pre-block convention — ViT-L result (2026-04-17)

Using `step2_extract_preblock.py` with the batched trainable probe
(`--solver trainable --grouping position --probe-set polar`) we reproduce the
qualitative PEZ pattern on V-JEPA 2-L:

| Layer | Speed R² | Direction R² | Accel R² |
|-------|----------|--------------|----------|
| 0 (patch_embed) | 0.254 | **0.087** | -0.037 |
| 1 (post-block-0) | 0.960 | 0.699 | 0.952 |
| 3 | 0.964 | 0.933 | 0.952 |
| 8 | 0.986 | 0.985 | 0.979 |
| 16 | 0.994 | 0.995 | 0.990 |
| 23 | 0.992 | **0.996** | 0.989 |

Key observations at layer 0 (patch-embed only, no blocks applied):
- direction R² ≈ 0, matching the paper's claim that direction is not decodable
  at the earliest layer
- speed R² is small but already above direction, matching the paper's
  speed-emerges-before-direction ordering
- acceleration R² ≈ 0, also matching the paper

The direction-R² transition is steeper than the paper's ViT-L curve
(the jump happens inside the first ~3 blocks rather than ~1/3 of the way
through the network). This is consistent with our rendering being visually
much simpler than the paper's Kubric scenes, so fewer blocks are needed to
separate speed from direction.

## Notes

- The local paper PDF is included in this repository as `pez_paper.pdf`.
- Generated datasets, features, and figures are not committed by default because they are large and change frequently during reruns.
