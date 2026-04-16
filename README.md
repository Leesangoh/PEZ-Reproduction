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

2. `step2_extract.py`
   Extracts layer-wise V-JEPA 2 features from the generated videos.
   The current extraction path probes the raw residual stream before the final model normalization.

3. `step3_probe.py`
   Runs grouped 5-fold linear probes and generates PEZ-style figures.
   Supported probe sets:
   - `polar`: speed, direction, acceleration magnitude
   - `cartesian`: `(vx, vy)` and `(ax, ay)`

## Repository Layout

- `constants.py`: shared constants and local paths
- `step1_generate.py`: dataset generation
- `step2_extract.py`: feature extraction
- `step2_extract_raw.py`: earlier extraction variant kept for reference
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

## Notes

- The local paper PDF is kept separately and is not tracked in git.
- Generated datasets, features, and figures are not committed by default because they are large and change frequently during reruns.
