# Final Reproduction Report

This report consolidates the best runs after the full Figure 2(b)/(c) sweep.
No further iteration is assumed here. The goal is to declare the best current
configuration per probe and the final reproduction verdict.

## Final Verdict

- Figure 2(c): `yes`
- Figure 2(b): `no`

## Figure 2(c) Final Verdict

Selected final config:

- source run: [results_fig2c_iter11_residpost_tlast_dirsector_angle.csv](/home/solee/pez/artifacts/results/results_fig2c_iter11_residpost_tlast_dirsector_angle.csv)
- capture: `resid_post`
- pooling: `temporal_last`
- grouping: `direction_spatial_sector`
- direction target: `angle`
- normalization: `zscore`
- solver: `trainable 20-HP sweep`

Paper criteria and outcome:

1. `direction` should become reliably available at the PEZ marker.
   Result: `first R^2 >= 0.8 = layer 8`
2. `direction` should peak in the middle layers.
   Result: peak `0.876 @ layer 16`
3. `direction` should weaken toward output layers.
   Result: `0.876 -> 0.835` (`drop = 0.041`)
4. scalar magnitudes should already be strongly decodable from early layers.
   Result:
   - `speed L0 = 0.895`
   - `acceleration L0 = 0.866`

Final judgment:

- The key Figure 2(c) PEZ pattern is reproduced.
- `iter14` ([results_fig2c_iter14_residpost_tlast_dirsector_angle_center.csv](/home/solee/pez/artifacts/results/results_fig2c_iter14_residpost_tlast_dirsector_angle_center.csv)) is a valid alternate because it slightly increases the late decline (`drop = 0.043`), but `iter11` remains the cleaner overall choice because its `direction` onset and absolute values are slightly stronger.

## Figure 2(b) Final Verdict

There is no single global config that matches both Cartesian probes best.
The best current readout is probe-specific.

### Selected Final Per-Probe Configs

| probe | selected run | capture | pooling | grouping | target | norm | solver | why selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `speed` | `fig2c_iter11_residpost_tlast_dirsector_angle` | `resid_post` | `temporal_last` | `direction_spatial_sector` | `angle` | `zscore` | `trainable` | strongest early scalar magnitude with preserved PEZ direction |
| `direction` | `fig2c_iter11_residpost_tlast_dirsector_angle` | `resid_post` | `temporal_last` | `direction_spatial_sector` | `angle` | `zscore` | `trainable` | `L8` onset + middle/late decline |
| `acceleration` | `fig2c_iter11_residpost_tlast_dirsector_angle` | `resid_post` | `temporal_last` | `direction_spatial_sector` | `angle` | `zscore` | `trainable` | strong early scalar magnitude |
| `velocity_xy` | `fig2b_iter23_velocity_residpost_tlastpatch_magsector_center` | `resid_post` | `temporal_last_patch` | `magnitude_spatial_sector` | `vxy` | `center` | `trainable` | best current compromise for delayed PEZ-like onset (`L8`) |
| `acceleration_xy` | `fig2b_iter16_accel_residpost_tlast_magnitude_center` | `resid_post` | `temporal_last` | `magnitude` | `vxy` | `center` | `trainable` | best current compromise for `L8` transition with acceptable absolute scale |

Supporting files:

- velocity selected:
  [results_fig2b_iter23_velocity_residpost_tlastpatch_magsector_center.csv](/home/solee/pez/artifacts/results/results_fig2b_iter23_velocity_residpost_tlastpatch_magsector_center.csv)
- acceleration selected:
  [results_fig2b_iter16_accel_residpost_tlast_magnitude_center.csv](/home/solee/pez/artifacts/results/results_fig2b_iter16_accel_residpost_tlast_magnitude_center.csv)

Key metrics:

- `velocity_xy` selected final:
  - `L0 = 0.527`
  - `L8 = 0.908`
  - `first R^2 >= 0.8 = layer 8`
  - peak `0.926 @ layer 12`
  - late `0.908`
- `acceleration_xy` selected final:
  - `L0 = 0.454`
  - `L8 = 0.915`
  - `first R^2 >= 0.8 = layer 8`
  - peak `0.944 @ layer 21`
  - late `0.939`

Why Figure 2(b) is still `no`:

1. The best velocity and best acceleration curves require different configs.
2. `velocity_xy` can be pushed to an `L8` transition with patch-level probing, but the absolute curve is weaker than the paper-like pooled bests.
3. `acceleration_xy` can keep the `L8` transition and good magnitude, but its peak remains later than ideal.
4. A single unified Cartesian recipe still does not simultaneously give:
   - low/moderate shallow baseline
   - `L8` transition
   - middle-layer peak
   - clear late weakening

## Remaining Gaps

### Figure 2(c)

Main remaining gap:

- late decline is present but mild (`drop ~0.04`) rather than strongly visible.

Likely causes:

1. synthetic scene is still too easy once the correct temporal slice is exposed
2. the paper may use a slightly different readout / smoothing / evaluation detail
3. our best config already relies on a hidden pooling choice (`temporal_last`) that is not stated in the paper

### Figure 2(b)

Main remaining gaps:

1. no single config is best for both Cartesian probes
2. velocity and acceleration prefer different recipes
3. patch-level probing improves velocity onset but hurts acceleration magnitude
4. `temporal_diff` makes shallow baselines unrealistically small and over-corrects the curves

Likely causes:

1. Cartesian variables are especially sensitive to pooling semantics
2. the public description is likely underspecified for the exact Figure 2(b) readout
3. scene simplicity still lets low-level motion cues leak too strongly unless the split/pooling is heavily constrained

## Final Declared Outcome

- Figure 2(c): reproduced
- Figure 2(b): not reproduced

## Final Artifacts

- [figure2c_final.png](/home/solee/pez/artifacts/results/figure2c_final.png)
- [figure2b_final.png](/home/solee/pez/artifacts/results/figure2b_final.png)
- [figure_reproduction_summary.png](/home/solee/pez/artifacts/results/figure_reproduction_summary.png)
