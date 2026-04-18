## Figure 2(b)/(c) Perfection Log

### Iteration 0

- Hypothesis: current best settings are still limited by split semantics rather than pure data generation.
- Baseline reference runs:
  - Fig 2(c): `results_step3_spatial_sector_angle.csv`
    - speed `L0=0.149`, `L8=0.987`, peak `L19`
    - direction `L0=-0.058`, first `R^2>=0.8` at `L8`, peak `L16`
    - acceleration `L0=-0.099`, `L8=0.967`, peak `L11`
  - Fig 2(c) alternate: `results_step7_spatial_sector_angle_resid_post.csv`
    - speed `L0=0.943`
    - direction `L0=0.613`, first `R^2>=0.8` at `L7`, peak `L11`
    - acceleration `L0=0.917`
  - Fig 2(b): `results_fig2b_velocity_xy_spatial_sector.csv`
    - velocity_xy `L0=0.072`, first `R^2>=0.8` at `L3`, peak `L12`
  - Fig 2(b): `results_fig2b_acceleration_xy_spatial_sector.csv`
    - acceleration_xy `L0=-0.070`, first `R^2>=0.8` at `L5`, peak `L12`
- Diff from paper:
  - Fig 2(c): `resid_pre` gets direction onset right but speed/acc early availability too low.
  - Fig 2(c): `resid_post` gets scalar early availability right but direction is too available too early.
  - Fig 2(b): both Cartesian curves rise too early relative to the paper's Layer-8 transition story.
- Next experiments:
  1. leave-directions-out under `resid_post`
  2. leave-magnitudes-out
  3. composite spatial+direction / spatial+magnitude splits

### Iteration 1

- Hypothesis: `resid_post` fixes the too-low scalar onset in Fig 2(c), and direction-aware holdouts can suppress the overly-early direction decode enough to recover the paper-like onset.
- What changed:
  - added `direction`, `direction_spatial_sector` groupings to the `resid_post` branch
  - ran:
    - `fig2c_iter1_residpost_direction_angle`
    - `fig2c_iter1_residpost_dirsector_angle`
    - `fig2b_iter1_velocity_residpost_direction`
    - `fig2b_iter1_accel_residpost_direction`
- Result:
  - Fig 2(c), `resid_post + direction`:
    - speed `L0=0.935`, acceleration `L0=0.904`
    - direction is over-suppressed: `L0=-0.401`, `L8=-0.024`, never reaches `0.8`
  - Fig 2(c), `resid_post + direction_spatial_sector`:
    - speed `L0=0.943`
    - direction `L0=0.354`, `first>=0.8 @ L8`, peak `L13=0.937`, late `L23=0.854`
    - acceleration `L0=0.923`
  - Fig 2(b), `resid_post + direction`:
    - velocity_xy `L0=-0.336`, `first>=0.8 @ L8`, peak `L10=0.912`
    - acceleration_xy `L0=-0.374`, `first>=0.8 @ L5`, peak `L9=0.914`
- Diff from paper:
  - Fig 2(c): `direction_spatial_sector` is the best run so far because it simultaneously keeps speed/acc high from L0 and puts direction onset at L8.
  - Fig 2(c): remaining gap is direction being too decodable at L0 (`0.354`) versus the paper's lower pre-PEZ regime.
  - Fig 2(b): direction holdout solves velocity onset but not acceleration onset.

### Iteration 2

- Hypothesis: leave-magnitudes-out directly targets the remaining Cartesian mismatch; it should raise velocity early decodability while potentially delaying acceleration.
- What changed:
  - added `magnitude`, `magnitude_spatial_sector` groupings
  - ran:
    - `fig2b_iter2_velocity_residpost_magnitude`
    - `fig2b_iter2_accel_residpost_magnitude`
    - `fig2b_iter2_accel_residpost_magsector`
- Result:
  - velocity_xy, `resid_post + magnitude`:
    - `L0=0.678`, `L8=0.978`, peak `L11=0.987`
    - early availability now matches the paper much better
  - acceleration_xy, `resid_post + magnitude`:
    - `L0=0.503`, `L8=0.909`, peak `L21=0.941`
    - but onset is still too early: `first>=0.8 @ L5`
  - acceleration_xy, `resid_post + magnitude_spatial_sector`:
    - `L0=0.614`, `L8=0.976`, peak `L21=0.980`
    - even more early-decodable than desired
- Diff from paper:
  - Fig 2(b): velocity_xy is now closer on absolute scale.
  - Fig 2(b): acceleration_xy still rises by L5, so PEZ-at-L8 is not yet matched.
- Next experiment:
  - test `condition` holdout under `resid_post` as the combined unseen direction×magnitude split.

### Iteration 3

- Hypothesis: `condition` holdout (joint unseen direction×magnitude) might outperform pure direction or pure magnitude holdout.
- What changed:
  - ran:
    - `fig2c_iter3_residpost_condition_angle`
    - `fig2b_iter3_velocity_residpost_condition`
    - `fig2b_iter3_accel_residpost_condition`
- Result:
  - Fig 2(c), `resid_post + condition`:
    - speed `L0=0.943`
    - direction `L0=0.558`, `first>=0.8 @ L8`, peak `L13=0.955`
    - acceleration collapses to `0` at every layer because condition grouping makes the acceleration validation folds degenerate
  - Fig 2(b), `resid_post + condition`:
    - velocity_xy is nearly identical to pure `magnitude`
    - acceleration_xy is nearly identical to pure `magnitude`
- Diff from paper:
  - Fig 2(c): unusable because acceleration becomes undefined.
  - Fig 2(b): no gain over pure `magnitude`.
- Next experiment:
  - test `direction_spatial_sector` for Figure 2(b), then try `resid_pre + magnitude` to see whether a less-easy capture fixes the too-early acceleration rise.

### Iteration 4

- Hypothesis: `direction_spatial_sector` may be a softer version of pure direction holdout for Figure 2(b), possibly delaying acceleration without crushing velocity.
- What changed:
  - ran:
    - `fig2b_iter4_velocity_residpost_dirsector`
    - `fig2b_iter4_accel_residpost_dirsector`
- Result:
  - velocity_xy:
    - `L0=0.645`, `L8=0.977`, peak `L11=0.987`, `first>=0.8 @ L4`
  - acceleration_xy:
    - `L0=0.582`, `L8=0.976`, peak `L11=0.978`, `first>=0.8 @ L5`
- Diff from paper:
  - both curves become even more early-decodable than desired.
  - this split is worse than pure `magnitude` for the remaining acceleration gap.
- Next experiment:
  - move off `resid_post` for Figure 2(b) and test `resid_pre + magnitude`.

### Iteration 5

- Hypothesis: `resid_pre + magnitude` may preserve the harder, later-emerging acceleration pattern while still giving Cartesian velocity enough early signal.
- What changed:
  - ran:
    - `fig2b_iter5_velocity_residpre_magnitude`
    - `fig2b_iter5_accel_residpre_magnitude`
- Result:
  - velocity_xy:
    - `L0=0.081`, `L8=0.955`, peak `L12=0.987`, `first>=0.8 @ L5`
  - acceleration_xy:
    - `L0=-0.343`, `L8=0.757`, peak `L22=0.941`, `first>=0.8 @ L6`
- Diff from paper:
  - acceleration onset is delayed slightly, but absolute performance is too low at the PEZ.
  - velocity loses the desirable early-layer availability.
- Next experiment:
  - test whether the Cartesian aggregation itself is mismatched (joint 2D probe vs. averaging separate x/y scalar probes).

### Iteration 6

- Hypothesis: Figure 2(b) may average separate scalar probes for `x` and `y` rather than using a single 2D multivariate R^2.
- What changed:
  - added scalar targets:
    - `velocity_x`, `velocity_y`
    - `acceleration_x`, `acceleration_y`
  - ran:
    - `fig2b_iter6_velocity_axes_residpost_magnitude`
    - `fig2b_iter6_accel_axes_residpost_magnitude`
- Result:
  - averaged velocity axes:
    - `L0=0.679`, `L8=0.979`, peak `L11=0.988`, `first>=0.8 @ L2`
  - averaged acceleration axes:
    - `L0=0.503`, `L8=0.913`, peak `L17=0.946`, `first>=0.8 @ L5`
- Diff from paper:
  - essentially unchanged from the joint 2D metric.
  - aggregation mismatch is not the main blocker.
- Next experiment:
  - test probe-strength variation directly on the best Cartesian split (`resid_post + magnitude`).

### Iteration 7

- Hypothesis: weaker probes might delay Cartesian emergence the same way they changed the polar curves.
- What changed:
  - ran:
    - `fig2b_iter7_velocity_residpost_magnitude_adamw100`
    - `fig2b_iter7_accel_residpost_magnitude_adamw100`
- Result:
  - velocity_xy:
    - `L0=0.509`, `L8=0.942`, peak `L8=0.942`, `first>=0.8 @ L8`
  - acceleration_xy:
    - underfit severely, never reaches `0.8`
- Diff from paper:
  - velocity becomes the closest Cartesian run so far.
  - acceleration collapses, so a single weak-probe recipe cannot reproduce the panel.
- Next experiment:
  - test `ridge` as an intermediate-strength probe.

### Iteration 8

- Hypothesis: `ridge` may sit between full trainable and weak AdamW100, preserving acceleration while delaying onset slightly.
- What changed:
  - ran:
    - `fig2b_iter8_velocity_residpost_magnitude_ridge`
    - `fig2b_iter8_accel_residpost_magnitude_ridge`
- Result:
  - velocity_xy:
    - `L0=0.722`, `L8=0.977`, peak `L12=0.984`, `first>=0.8 @ L4`
  - acceleration_xy:
    - `L0=0.516`, `L8=0.905`, peak `L16=0.939`, `first>=0.8 @ L5`
- Diff from paper:
  - ridge behaves almost like the original trainable run.
  - it does not solve the stubborn acceleration onset problem.
- Next experiment:
  - test a pooling / frame-horizon hypothesis (`temporal_last`) without regenerating the whole dataset.

### Iteration 9

- Hypothesis: the paper may effectively pool later temporal evidence rather than uniformly averaging all space-time tokens; using the last temporal slice could delay acceleration emergence without regenerating data.
- What changed:
  - added pooling modes to `step2_extract.py`
  - extracted `resid_post + temporal_last` features to:
    - `artifacts/features/resid_post_resize_temporal_last`
  - ran:
    - `fig2b_iter9_velocity_residpost_tlast_magnitude`
    - `fig2b_iter9_accel_residpost_tlast_magnitude`
- Result:
  - velocity_xy:
    - `L0=0.600`, `L8=0.951`, peak `L23=0.978`, `first>=0.8 @ L6`
  - acceleration_xy:
    - `L0=0.414`, `L8=0.890`, peak `L20=0.954`, `first>=0.8 @ L8`
- Diff from paper:
  - this is the first run where the stubborn Cartesian acceleration onset lands at `L8`.
  - velocity remains early-available, which is qualitatively consistent with the paper, though its peak is still later than ideal.
- Next experiment:
  - check whether weak probing on top of temporal-last improves the peak timing without losing the acceleration transition.

### Iteration 10

- Hypothesis: `adamw100` on top of `temporal_last` might keep the improved acceleration onset while pulling the velocity peak earlier.
- What changed:
  - ran:
    - `fig2b_iter10_velocity_residpost_tlast_magnitude_adamw100`
    - `fig2b_iter10_accel_residpost_tlast_magnitude_adamw100`
- Result:
  - velocity_xy:
    - `L0=0.484`, `L8=0.887`, peak `L23=0.957`, `first>=0.8 @ L8`
  - acceleration_xy:
    - `L0=0.307`, `L8=0.352`, never reaches `0.8`
- Diff from paper:
  - weak probing again destroys the acceleration curve.
  - `temporal_last + trainable` remains the best Figure 2(b) candidate.
- Next experiment:
  - verify that `temporal_last` does not break Figure 2(c) under the current best polar split.

### Iteration 11

- Hypothesis: the same `temporal_last` pooling that fixed Figure 2(b) may also preserve the Figure 2(c) PEZ pattern, giving a more unified explanation of the paper's hidden setup.
- What changed:
  - ran:
    - `fig2c_iter11_residpost_tlast_dirsector_angle`
- Result:
  - speed:
    - `L0=0.895`, `L8=0.983`, peak `L19=0.988`
  - direction:
    - `L0=0.326`, `L8=0.816`, peak `L16=0.876`, `first>=0.8 @ L8`, late `L23=0.835`
  - acceleration:
    - `L0=0.866`, `L8=0.974`, peak `L20=0.986`
- Diff from paper:
  - polar shape remains intact:
    - scalar quantities are strongly available from early layers
    - direction still crosses the PEZ threshold at `L8`
    - late decline is present
  - this is now the strongest unified hypothesis across both panels.

### Current Best So Far

- Figure 2(c):
  - `fig2c_iter11_residpost_tlast_dirsector_angle`
- Figure 2(b):
  - `fig2b_iter9_velocity_residpost_tlast_magnitude`
  - `fig2b_iter9_accel_residpost_tlast_magnitude`

Interpretation:

- A hidden pooling/detail mismatch is now the leading explanation.
- `resid_post + temporal_last` improves Figure 2(b) substantially while keeping Figure 2(c) consistent with the PEZ story.
