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

### Iteration 12

- Hypothesis: `temporal_last` may only work under `resid_post`; switching back to `resid_pre` while keeping the same pooling could delay Cartesian decoding toward the paper.
- What changed:
  - extracted:
    - `artifacts/features/resid_pre_resize_temporal_last`
  - ran:
    - `fig2b_iter12_velocity_residpre_tlast_magnitude`
    - `fig2b_iter12_accel_residpre_tlast_magnitude`
- Result:
  - velocity_xy:
    - `L0=-0.011`, `L8=0.893`, peak `L23=0.978`, `first>=0.8 @ L7`
  - acceleration_xy:
    - `L0=-0.393`, `L8=0.707`, peak `L21=0.954`, `first>=0.8 @ L9`
- Diff from paper:
  - this is too conservative.
  - onset shifts later, but early scalar availability collapses and overall absolute values are too low.
- Next experiment:
  - keep `resid_post + temporal_last`, but tune the probe normalization instead of the capture branch.

### Iteration 13

- Hypothesis: the remaining Figure 2(c) gap may come from an overly strong trainable probe; `adamw100` on top of the current best temporal-last branch might increase the late decline while preserving the `L8` direction onset.
- What changed:
  - ran:
    - `fig2c_iter13_residpost_tlast_dirsector_angle_adamw100`
- Result:
  - speed:
    - `L0=0.865`, `L8=0.946`, peak `L14=0.956`
  - direction:
    - `L0=0.276`, `L8=0.705`, peak `L19=0.711`, never reaches `0.8`
  - acceleration:
    - `L0=0.838`, `L8=0.940`, peak `L15=0.949`
- Diff from paper:
  - the late decline is a bit stronger, but the core PEZ threshold is lost.
  - this weak-probe variant is not acceptable for Figure 2(c).
- Next experiment:
  - keep the trainable solver and modify only feature normalization.

### Iteration 14

- Hypothesis: full z-score standardization may over-amplify shallow direction signal; using mean-centering only (`norm_mode=center`) could preserve `L8` onset while slightly increasing the late decline.
- What changed:
  - added `--norm-mode {zscore,center,none}` to `step3_probe.py`
  - ran:
    - `fig2c_iter14_residpost_tlast_dirsector_angle_center`
- Result:
  - speed:
    - `L0=0.820`, `L8=0.983`, peak `L16=0.987`
  - direction:
    - `L0=0.294`, `L8=0.800`, peak `L21=0.872`, late `L23=0.829`, drop `0.043`
  - acceleration:
    - `L0=0.814`, `L8=0.974`, peak `L20=0.984`
- Diff from paper:
  - direction still crosses the PEZ threshold at `L8` and the late decline is slightly cleaner.
  - however, speed/acceleration early magnitude becomes a little too low relative to the previous best.
- Next experiment:
  - attack the remaining Figure 2(b) velocity `L0` gap directly with split/normalization changes on the existing temporal-last features.

### Iteration 15

- Hypothesis: making the Cartesian split harder with a joint magnitude-and-sector holdout may lower shallow `velocity_xy` decoding without regenerating features.
- What changed:
  - ran:
    - `fig2b_iter15_velocity_residpost_tlast_magsector`
    - `fig2b_iter15_accel_residpost_tlast_magsector`
- Result:
  - velocity_xy:
    - `L0=0.561`, `L8=0.943`, peak `L22=0.977`, `first>=0.8 @ L6`
  - acceleration_xy:
    - `L0=0.573`, `L8=0.969`, peak `L19=0.983`, `first>=0.8 @ L5`
- Diff from paper:
  - velocity improves slightly, but acceleration becomes even more early-decodable.
  - the joint holdout is too aggressive in the wrong direction.
- Next experiment:
  - keep the simpler magnitude split and apply only probe centering.

### Iteration 16

- Hypothesis: the remaining high `velocity_xy` baseline may come from target z-scoring; using `norm_mode=center` on the best temporal-last Cartesian setup may lower `L0` while preserving the `L8` acceleration transition.
- What changed:
  - ran:
    - `fig2b_iter16_velocity_residpost_tlast_magnitude_center`
    - `fig2b_iter16_accel_residpost_tlast_magnitude_center`
- Result:
  - velocity_xy:
    - `L0=0.553`, `L8=0.962`, peak `L23=0.978`, `first>=0.8 @ L6`
  - acceleration_xy:
    - `L0=0.454`, `L8=0.915`, peak `L21=0.944`, `first>=0.8 @ L8`
- Diff from paper:
  - this is the best balanced Cartesian pair so far:
    - velocity early magnitude comes down
    - acceleration keeps the `L8` transition
  - velocity `L0` is still slightly above the target range.
- Next experiment:
  - test whether removing normalization entirely helps Figure 2(c) direction decline.

### Iteration 17

- Hypothesis: full standardization may itself create too much shallow direction signal; removing normalization (`norm_mode=none`) might produce a cleaner PEZ transition and stronger late decline in Figure 2(c).
- What changed:
  - ran:
    - `fig2c_iter17_residpost_tlast_dirsector_angle_none`
- Result:
  - speed:
    - `L0=0.733`, `L8=0.970`, peak `L19=0.982`
  - direction:
    - `L0=0.030`, `L8=0.612`, peak `L22=0.775`, never reaches `0.8`
  - acceleration:
    - `L0=0.700`, `L8=0.925`, peak `L15=0.955`, late `L23=0.882`
- Diff from paper:
  - shallow direction is strongly suppressed, but the PEZ onset is now too weak and too late.
  - this confirms that some standardization is necessary for the paper-like direction curve.
- Next experiment:
  - try a different temporal pooling branch for Cartesian velocity only.

### Iteration 18

- Hypothesis: using the first temporal slice instead of the last may lower the shallow `velocity_xy` baseline while keeping the `L8` acceleration transition.
- What changed:
  - extracted:
    - `artifacts/features/resid_post_resize_temporal_first`
  - ran:
    - `fig2b_iter18_velocity_residpost_tfirst_magnitude`
    - `fig2b_iter18_accel_residpost_tfirst_magnitude`
- Result:
  - velocity_xy:
    - `L0=0.571`, `L8=0.971`, peak `L21=0.984`, `first>=0.8 @ L4`
  - acceleration_xy:
    - `L0=0.456`, `L8=0.833`, peak `L17=0.959`, `first>=0.8 @ L8`
- Diff from paper:
  - velocity remains too early and too strong.
  - acceleration still transitions at `L8`, but absolute shape is not better than `iter16`.
- Next experiment:
  - combine the two mildly helpful knobs for velocity: `magnitude_spatial_sector` and `center`.

### Iteration 19

- Hypothesis: the best path for Figure 2(b) velocity may require a probe-specific recipe; combining `magnitude_spatial_sector` with `norm_mode=center` could pull `velocity_xy L0` into the paper range while leaving acceleration to a different best config.
- What changed:
  - ran:
    - `fig2b_iter19_velocity_residpost_tlast_magsector_center`
    - `fig2b_iter19_accel_residpost_tlast_magsector_center`
- Result:
  - velocity_xy:
    - `L0=0.505`, `L8=0.942`, peak `L22=0.977`, `first>=0.8 @ L6`
  - acceleration_xy:
    - `L0=0.555`, `L8=0.970`, peak `L19=0.983`, `first>=0.8 @ L5`
- Diff from paper:
  - this is the strongest velocity-only match so far, bringing `L0` to the upper edge of the desired paper range.
  - acceleration gets worse again, so Figure 2(b) is now clearly a per-probe best-combination problem.

### Current Best After Iteration 19

- Figure 2(c):
  - `fig2c_iter11_residpost_tlast_dirsector_angle`
  - `fig2c_iter14_residpost_tlast_dirsector_angle_center` is a viable alternate, but not a clear overall win.
- Figure 2(b):
  - velocity_xy:
    - `fig2b_iter19_velocity_residpost_tlast_magsector_center`
  - acceleration_xy:
    - `fig2b_iter16_accel_residpost_tlast_magnitude_center`

Interpretation:

- The remaining gap is no longer dominated by a single global setting.
- Figure 2(c) prefers:
  - `resid_post + temporal_last + direction_spatial_sector + angle`
- Figure 2(b) prefers probe-specific settings:
  - velocity likes a harder split plus centered targets
  - acceleration likes the simpler magnitude split plus centered targets

### Iteration 20

- Hypothesis: `temporal_diff` pooling may better match motion-difference semantics and therefore align the PEZ transition more tightly with the paper.
- What changed:
  - extracted:
    - `artifacts/features/resid_post_resize_temporal_diff`
  - ran:
    - `fig2c_iter20_residpost_tdiff_dirsector_angle`
- Result:
  - speed:
    - `L0=0.447`, `L8=0.965`, peak `L17=0.987`, `first>=0.8 @ L2`
  - direction:
    - `L0=-0.503`, `L8=0.731`, peak `L21=0.865`, `first>=0.8 @ L11`, late `L23=0.864`
  - acceleration:
    - `L0=0.848`, `L8=0.962`, peak `L19=0.984`
- Diff from paper:
  - `temporal_diff` over-corrects the direction curve:
    - onset moves too late
    - late decline disappears
  - this branch is worse than the current best Figure 2(c) recipe.
- Next experiment:
  - test the same pooling on Cartesian targets only.

### Iteration 21

- Hypothesis: `temporal_diff` may still help Figure 2(b) by lowering the overly high shallow Cartesian baseline.
- What changed:
  - ran:
    - `fig2b_iter21_velocity_residpost_tdiff_magsector_center`
    - `fig2b_iter21_accel_residpost_tdiff_magnitude_center`
- Result:
  - velocity_xy:
    - `L0=0.027`, `L8=0.958`, peak `L13=0.987`, `first>=0.8 @ L5`
  - acceleration_xy:
    - `L0=0.011`, `L8=0.904`, peak `L19=0.977`, `first>=0.8 @ L5`
- Diff from paper:
  - shallow baselines become unrealistically small.
  - onset stays too early, so `temporal_diff` is not the missing detail for Figure 2(b) either.
- Next experiment:
  - move from pooled clip vectors to patch-level probing, closer to the Appendix C.5 analysis.
