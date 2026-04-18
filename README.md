# PEZ Figure 2 Rewrite

## Best Configuration for Reproduction

Final consolidated outputs:

- [figure2c_final.png](/home/solee/pez/artifacts/results/figure2c_final.png)
- [figure2b_final.png](/home/solee/pez/artifacts/results/figure2b_final.png)
- [figure_reproduction_summary.png](/home/solee/pez/artifacts/results/figure_reproduction_summary.png)
- [final_reproduction_report.md](/home/solee/pez/artifacts/results/final_reproduction_report.md)

### Figure 2(c) Polar (재현 qualified)

재현 상태: `qualified`

Config:

| field | value |
| --- | --- |
| capture | `resid_post` |
| pooling | `temporal_last` |
| grouping | `direction_spatial_sector` |
| target | `angle` |
| norm | `zscore` |
| solver | `trainable (20 HP sweep)` |
| selected run | `fig2c_iter11_residpost_tlast_dirsector_angle` |

Key metrics:

| probe | L0 | L8 | peak | late decline |
| --- | ---: | ---: | ---: | ---: |
| `speed` | `0.895` | `0.983` | `0.988 @ L19` | `0.988 -> 0.985` |
| `direction` | `0.326` | `0.816` | `0.876 @ L16` | `0.876 -> 0.835` |
| `acceleration` | `0.866` | `0.974` | `0.986 @ L20` | `0.986 -> 0.981` |

실행 명령어:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /isaac-sim/python.sh /home/solee/pez/step2_extract.py \
  --capture resid_post \
  --transform resize \
  --pooling temporal_last \
  --output-root /home/solee/pez/artifacts/features/resid_post_resize_temporal_last \
  --device cuda:0
```

```bash
env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /isaac-sim/python.sh /home/solee/pez/step3_probe.py run \
  --run-name fig2c_iter11_residpost_tlast_dirsector_angle \
  --feature-root /home/solee/pez/artifacts/features/resid_post_resize_temporal_last \
  --probe-set fig2c \
  --solver trainable \
  --norm-mode zscore \
  --grouping direction_spatial_sector \
  --direction-target angle \
  --residual-capture resid_post \
  --preprocessing resize \
  --device cuda:0
```

출력 파일:

- [results_fig2c_iter11_residpost_tlast_dirsector_angle.csv](/home/solee/pez/artifacts/results/results_fig2c_iter11_residpost_tlast_dirsector_angle.csv)
- [figure2c_final.png](/home/solee/pez/artifacts/results/figure2c_final.png)

### Figure 2(b) Cartesian (partial)

재현 상태: `partial`

Config:

| probe | capture | pooling | grouping | target | norm | solver | selected run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `velocity_xy` | `resid_post` | `temporal_last_patch` | `magnitude_spatial_sector` | `vxy` | `center` | `trainable (20 HP sweep)` | `fig2b_iter23_velocity_residpost_tlastpatch_magsector_center` |
| `acceleration_xy` | `resid_post` | `temporal_last` | `magnitude` | `vxy` | `center` | `trainable (20 HP sweep)` | `fig2b_iter16_accel_residpost_tlast_magnitude_center` |

Key metrics:

| probe | L0 | L8 | peak | late decline |
| --- | ---: | ---: | ---: | ---: |
| `velocity_xy` | `0.527` | `0.908` | `0.926 @ L12` | `0.926 -> 0.908` |
| `acceleration_xy` | `0.454` | `0.915` | `0.944 @ L21` | `0.944 -> 0.939` |

실행 명령어:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /isaac-sim/python.sh /home/solee/pez/step2_extract.py \
  --capture resid_post \
  --transform resize \
  --pooling temporal_last_patch \
  --output-root /home/solee/pez/artifacts/features/resid_post_resize_temporal_last_patch \
  --device cuda:0
```

```bash
env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /isaac-sim/python.sh /home/solee/pez/step3_probe.py run \
  --run-name fig2b_iter23_velocity_residpost_tlastpatch_magsector_center \
  --feature-root /home/solee/pez/artifacts/features/resid_post_resize_temporal_last_patch \
  --probe-set fig2b_velocity_xy \
  --solver trainable \
  --norm-mode center \
  --grouping magnitude_spatial_sector \
  --direction-target vxy \
  --residual-capture resid_post \
  --preprocessing resize \
  --device cuda:0
```

```bash
env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 /isaac-sim/python.sh /home/solee/pez/step2_extract.py \
  --capture resid_post \
  --transform resize \
  --pooling temporal_last \
  --output-root /home/solee/pez/artifacts/features/resid_post_resize_temporal_last \
  --device cuda:0
```

```bash
env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 /isaac-sim/python.sh /home/solee/pez/step3_probe.py run \
  --run-name fig2b_iter16_accel_residpost_tlast_magnitude_center \
  --feature-root /home/solee/pez/artifacts/features/resid_post_resize_temporal_last \
  --probe-set fig2b_acceleration_xy \
  --solver trainable \
  --norm-mode center \
  --grouping magnitude \
  --direction-target vxy \
  --residual-capture resid_post \
  --preprocessing resize \
  --device cuda:0
```

출력 파일:

- [results_fig2b_iter23_velocity_residpost_tlastpatch_magsector_center.csv](/home/solee/pez/artifacts/results/results_fig2b_iter23_velocity_residpost_tlastpatch_magsector_center.csv)
- [results_fig2b_iter16_accel_residpost_tlast_magnitude_center.csv](/home/solee/pez/artifacts/results/results_fig2b_iter16_accel_residpost_tlast_magnitude_center.csv)
- [figure2b_final.png](/home/solee/pez/artifacts/results/figure2b_final.png)

### Figure 1 IntPhys possible/impossible (qualified)

재현 상태: `qualified`

Config:

| field | value |
| --- | --- |
| dataset | `IntPhys full dev` |
| capture | `resid_pre` |
| transform | `resize` |
| frames | `16-frame sample` |
| label | `possible vs impossible` |
| metric | `scene-relative accuracy` |
| solver | `trainable linear probe` |
| selected run | `intphys_possible_impossible_full_select_relative` |

Key metrics:

| metric | L0 | L8 | peak | late decline |
| --- | ---: | ---: | ---: | ---: |
| `relative_accuracy` | `0.733` | `1.000` | `1.000` | `1.000 -> 1.000` |
| `clip_accuracy` | `0.519` | `0.736` | `0.769 @ L18` | `0.769 -> 0.728` |

실행 명령어:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /isaac-sim/python.sh /home/solee/pez/step_intphys_probe.py \
  --device cuda:0 \
  --capture resid_pre \
  --transform resize \
  --batch-size 8 \
  --feature-root /home/solee/pez/artifacts/features/intphys_resid_pre_resize_fulldev \
  --reuse-features \
  --run-name intphys_possible_impossible_full_select_relative \
  --n-frames-sample 16 \
  --selection-metric relative_accuracy
```

출력 파일:

- [results_intphys_possible_impossible_full_select_relative.csv](/home/solee/pez/artifacts/results/results_intphys_possible_impossible_full_select_relative.csv)
- [figure_intphys_possible_impossible_full_select_relative.png](/home/solee/pez/artifacts/results/figure_intphys_possible_impossible_full_select_relative.png)
- [summary_intphys_possible_impossible_full_select_relative.json](/home/solee/pez/artifacts/results/summary_intphys_possible_impossible_full_select_relative.json)

## Reproduction Notes

Paper에 명시된 spec:

- `V-JEPA v2-L`
- `16 frames`, `24 fps`, `256 x 256`
- synthetic ball video on Kubric/PyBullet/Blender
- layer-wise residual probing
- linear probe with `20 HP` sweep
- `5-fold grouped CV`

Paper에 안 적히거나 불충분했던 hidden detail:

- exact residual capture point: `resid_pre` vs `resid_post`
- exact grouping key for grouped CV
- direction target parameterization: `angle` vs `sin/cos`
- temporal pooling detail: `mean` vs `temporal_last`
- whether patch-level readout is needed for some Cartesian variables
- IntPhys metric: clip accuracy vs scene-relative accuracy

24+ iteration 뒤에 확인한 critical config choice:

- Figure 2(c)는 `resid_post + temporal_last + direction_spatial_sector + angle`에서만 paper-like onset/peak/decline이 동시에 나왔습니다.
- Figure 2(b)는 probe-specific recipe가 필요했습니다.
  - `velocity_xy`: `temporal_last_patch + magnitude_spatial_sector + center`
  - `acceleration_xy`: `temporal_last + magnitude + center`
- Figure 1 IntPhys는 clip accuracy로는 안 맞고, `scene-relative accuracy`로 봐야 paper-like `L7/L8` jump가 나왔습니다.

Trade-offs:

- `temporal_diff`는 `L0`를 지나치게 낮춰서 absolute scale을 망쳤습니다.
- `norm=none`은 polar direction의 `L0`는 낮추지만 `L8` onset을 깨뜨렸습니다.
- patch-level probing은 `velocity_xy`에는 도움이 됐지만 `acceleration_xy`는 오히려 악화됐습니다.
- 그래서 현재 final reproduction은 single universal recipe가 아니라, panel/probe-specific best recipe입니다.

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
