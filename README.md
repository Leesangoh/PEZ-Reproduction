# PEZ Figure 2 Rewrite

This directory was rewritten around reproducing and stress-testing
Figure 2 from [pez_paper.pdf](/home/solee/pez/pez_paper.pdf), with the main
focus on:

- Figure 2(c): polar
- Figure 2(b): cartesian

The old code and results were moved to:

- `/home/solee/pez/archive_pre_rewrite_260417/`

## Current Pipeline

- [step1_generate.py](/home/solee/pez/step1_generate.py)
  Paper-faithful synthetic ball data generator.
- [step2_extract.py](/home/solee/pez/step2_extract.py)
  V-JEPA v2-L feature extractor with:
  - `resid_pre` vs `resid_post`
  - `resize` vs `eval_preproc`
- [step3_probe.py](/home/solee/pez/step3_probe.py)
  Linear probe runner for Figure 2(c) and Figure 2(b), plus summary generation.

## Paper-Faithful Base

Base config used for the rewrite:

- model: `V-JEPA v2-L`
- input: `256 x 256`, `16 frames`, `24 fps`
- feature: `resid_pre`
- pooling: mean over space-time tokens
- probe: trainable linear probe `f(h)=Wh+b`
- hyperparameter sweep:
  - `lr ∈ {1e-4, 3e-4, 1e-3, 3e-3, 5e-3}`
  - `wd ∈ {0.01, 0.1, 0.4, 0.8}`
- validation: `5-fold grouped CV`
- target: polar variables
  - `speed`
  - `direction`
  - `acceleration magnitude`

Base outputs:

- [results_base_pre_resize_position_sincos_gpu.csv](/home/solee/pez/artifacts/results/results_base_pre_resize_position_sincos_gpu.csv)
- [figure_base_pre_resize_position_sincos_gpu.png](/home/solee/pez/artifacts/results/figure_base_pre_resize_position_sincos_gpu.png)

Base direction metrics:

- `L0 R² = 0.0386`
- `L1-L6 mean R² = 0.8846`
- `first layer with R² >= 0.8 = 2`
- `peak direction R² = 0.9953 @ layer 16`

## Completed Runs

Completed runs from the base config:

- grouping:
  - `condition`
  - `video`
  - `direction`
- direction target:
  - `angle scalar [-pi, pi]`
- preprocessing:
  - `eval_preproc (resize short side then center crop 256)`
- residual capture:
  - `resid_post`

Outputs:

- [ablation_summary.csv](/home/solee/pez/artifacts/results/ablation_summary.csv)
- [figure2c_ablation_overlay.png](/home/solee/pez/artifacts/results/figure2c_ablation_overlay.png)
- [best_config.json](/home/solee/pez/artifacts/results/best_config.json)

Summary over all 7 runs:

| run | L0 dir R² | L1-L6 mean | first >= 0.8 | peak dir R² | peak layer | rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ablate_direction_target_angle_gpu` | 0.0597 | 0.7269 | 6 | 0.9748 | 15 | 1 |
| `base_pre_resize_position_sincos_gpu` | 0.0386 | 0.8846 | 2 | 0.9953 | 16 | 2 |
| `ablate_preproc_eval_gpu` | 0.1549 | 0.9093 | 2 | 0.9953 | 16 | 3 |
| `ablate_group_condition_gpu` | 0.0252 | 0.8688 | 2 | 0.9942 | 23 | 4 |
| `ablate_group_video_gpu` | 0.0577 | 0.8826 | 2 | 0.9955 | 23 | 5 |
| `ablate_residual_capture_post_gpu` | 0.6915 | 0.9356 | 1 | 0.9958 | 21 | 6 |
| `ablate_group_direction_gpu` | -0.1612 | 0.4606 | inf | 0.5833 | 11 | 7 |

## Current Readout

What the completed ablations say already:

- `position -> condition` and `position -> video` barely change the curve.
- `eval_preproc` does not recover a PEZ-like mid-depth jump.
- `direction target = angle scalar` reduces early-layer direction decodability the most
  among the completed ablations and is currently the closest match under the
  simple paper-distance heuristic.
- `resid_post` sharply increases early-layer direction decodability
  (`L0 R² = 0.6915`, `first >= 0.8 = layer 1`) and is clearly less paper-like
  than `resid_pre`.
- `direction grouping` strongly suppresses early direction decoding, but it also
  suppresses overall peak performance, so it does not look like the paper figure.

## Figure 2(c) Verdict

- [final_verdict.md](/home/solee/pez/artifacts/results/final_verdict.md)

Current best Figure 2(c) result:

- qualified reproduction achieved under:
  - `resid_pre`
  - true spatial holdout
  - `angle` direction target

## Figure 2(b) Cartesian

Main outputs:

- [fig2b_ablation_summary.csv](/home/solee/pez/artifacts/results/fig2b_ablation_summary.csv)
- [fig2b_overlay.png](/home/solee/pez/artifacts/results/fig2b_overlay.png)
- [final_verdict_fig2b.md](/home/solee/pez/artifacts/results/final_verdict_fig2b.md)

Current best Cartesian runs:

- [results_fig2b_velocity_xy_spatial_sector.csv](/home/solee/pez/artifacts/results/results_fig2b_velocity_xy_spatial_sector.csv)
- [results_fig2b_acceleration_xy_spatial_sector.csv](/home/solee/pez/artifacts/results/results_fig2b_acceleration_xy_spatial_sector.csv)

Cartesian summary:

- `velocity_xy` is early-decodable, but not strongly from `L0`
- `acceleration_xy` rises sharply, but too early (`first >= 0.8` at `L5`)
- both variables peak in middle layers and weaken slightly toward the output
- overall: partial qualitative match, not a full Figure 2(b) reproduction

## Possible/Impossible Physics

Main outputs:

- [possible_impossible_reproduction.md](/home/solee/pez/artifacts/results/possible_impossible_reproduction.md)
- [results_intphys_possible_impossible.csv](/home/solee/pez/artifacts/results/results_intphys_possible_impossible.csv)
- [figure_intphys_possible_impossible.png](/home/solee/pez/artifacts/results/figure_intphys_possible_impossible.png)

Current status:

- paper target is **Figure 1** (`IntPhys` possible vs impossible)
- deep root-cause analysis:
  - [intphys_deep_rootcause.md](/home/solee/pez/artifacts/results/intphys_deep_rootcause.md)
- key finding:
  - **clip-level accuracy** does not reproduce Figure 1
  - **scene-relative grouped accuracy** does show a PEZ-like `L7/L8` jump
- best evidence:
  - 16-frame full dev:
    - clip accuracy `L8 = 73.9%`
    - relative accuracy `L8 = 97.8%`
  - 64-frame full dev:
    - clip accuracy `L8 = 71.7%`
    - relative accuracy `L8 = 100%`

## Results Directory

All current rewrite outputs live under:

- [artifacts/results](/home/solee/pez/artifacts/results)

The archived pre-rewrite outputs remain under:

- [archive_pre_rewrite_260417/artifacts/results](/home/solee/pez/archive_pre_rewrite_260417/artifacts/results)
