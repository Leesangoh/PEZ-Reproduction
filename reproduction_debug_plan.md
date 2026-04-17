# PEZ Reproduction Debug Plan

## Goal

Reproduce Figure 2(c) from the PEZ paper as faithfully as possible, while
isolating why the current pipeline still fails to match the published PEZ
signature.

Current status:

- Reproduction is still not achieved under the current 5-fold setups.
- `resid_pre` is clearly better than `resid_post`.
- `direction_target=angle` is currently the most paper-like of the completed
  7 ablations, but it is still not a true match.
- The strongest remaining hypothesis is evaluation-protocol mismatch rather
  than a pure data-generation problem.

## Working Diagnosis

The most likely causes, in descending priority:

1. Grouped 5-fold CV is underspecified in the paper.
2. Current splits do not enforce true spatial generalization.
3. Direction target parameterization is ambiguous (`angle` vs `sin/cos`).
4. Residual readout mismatch existed earlier, but now explains only part of the gap.
5. Data/rendering mismatch may still exist, but is no longer the leading explanation.

The paper's Appendix C.5 strongly suggests that the real transition is from
local/retinotopic direction signals to global position-invariant direction.
That means naive mean-pooled 5-fold probing may reveal shallow local direction
information too early unless the split actually tests spatial generalization.

## Fixed Constraints For Future Exploration

These should remain fixed unless explicitly tested as an ablation:

- use `5-fold` validation
- use `resid_pre` as the default feature capture
- keep the current paper-faithful synthetic dataset as the default data source
- keep V-JEPA v2-L as the default model for Figure 2(c)
- compare all future runs against the current 7-run ablation baseline

## Execution Plan

### Step 1. Freeze the Current Baseline

Use the current 7-run ablation results as the reference baseline:

- `base_pre_resize_position_sincos_gpu`
- `ablate_group_condition_gpu`
- `ablate_group_video_gpu`
- `ablate_group_direction_gpu`
- `ablate_direction_target_angle_gpu`
- `ablate_preproc_eval_gpu`
- `ablate_residual_capture_post_gpu`

Reference artifacts:

- `artifacts/results/ablation_summary.csv`
- `artifacts/results/figure2c_ablation_overlay.png`
- `artifacts/results/best_config.json`

Purpose:

- avoid re-litigating already-explored axes
- measure every future experiment against a fixed anchor

### Step 2. Build True Spatial-Holdout Splits

Current `position` grouping only groups by `_pos0..6` index. Since start
positions are resampled independently for each `(direction, condition)` pair,
this is not a real spatial holdout.

Need to create and test 5-fold splits that actually hold out spatial regions:

1. pixel-region holdout
2. quadrant/sector holdout
3. clustered spatial holdout using actual `(pos_x_px, pos_y_px)`

Success criterion:

- train and validation see different spatial regions in the image plane

Main question:

- does true spatial holdout move direction emergence closer to Layer 8?

### Step 3. Re-run Figure 2(c) Mean-Pooled Probes Under Spatial Holdout

For each new spatial-holdout split:

- keep `resid_pre`
- keep `5-fold`
- run both direction targets:
  - `sincos`
  - `angle`

Track:

- direction `L0`
- direction `L1-L6 mean`
- first layer with `R^2 >= 0.8`
- direction peak layer

Main question:

- does the paper-like emergence appear once spatial generalization is enforced?

### Step 4. Reproduce Appendix C.5 More Directly

The paper's explanation of direction emergence is local-to-global.
This must be tested directly.

Need to run:

1. per-patch direction probe vs layer
2. train-on-one-region / test-on-unseen-region transfer probe
3. per-patch decoding heatmaps at representative layers:
   - layer 0
   - layer 7
   - layer 8
   - later layer such as 15

Main question:

- is Layer 7->8 a real transition from local direction to global direction?

If yes, that explains why naive mean-pooled probing under weak spatial
constraints fails to show the published PEZ curve cleanly.

### Step 5. Lock Down Direction Target Parameterization

Under the same spatial-holdout split, compare:

1. `angle` scalar
2. `sin/cos`
3. optional auxiliary comparison: `vx/vy`

Main question:

- is the published Figure 2(c) behavior sensitive to the target parameterization?

Interpretation:

- if only `angle` yields PEZ-like emergence, the paper may have effectively used
  a scalar-angle implementation even if not stated explicitly

### Step 6. Re-check Probe Strength Under Spatial Holdout

Once spatial-holdout splits are working, compare:

1. current trainable 20-HP sweep
2. weaker `AdamW100`
3. ridge

Main question:

- is PEZ only visible under weaker probes, or does it survive paper-faithful probing
  once the split is corrected?

Interpretation:

- PEZ only under weak probes -> hidden probe-training detail is likely important
- PEZ also under paper-faithful probing -> split mismatch was the main blocker

### Step 7. Final Layer-Definition Sanity Check

After the major split/target questions are settled, do one final layer-definition check:

1. current `resid_pre`
2. strict `patch_embed + pre-block states`
3. strict `post-block states`

Purpose:

- document exactly which internal representation aligns best with the paper's
  layer numbering

This is lower priority because it already explains only part of the mismatch.

### Step 8. Only Then Re-open Data/Rendering Causes

Only after Steps 2-7:

- revisit GT definition
- revisit target extraction details
- revisit rendering/physics hidden mismatches
- revisit any remaining checkpoint/inference-path differences

Reason:

- current evidence says protocol mismatch is more likely than data mismatch

## Decision Tree

### Case A. Spatial holdout restores a Layer-8 PEZ transition

Conclusion:

- the main failure was split/protocol mismatch

### Case B. Only weak probes recover PEZ-like behavior

Conclusion:

- hidden probe-training details matter

### Case C. Even spatial holdout + per-patch analysis fails

Conclusion:

- the paper likely omits a critical implementation detail, or public
  information is insufficient to reproduce Figure 2(c) exactly

## Recommended Immediate Order

The next execution order should be:

1. Step 2: true spatial-holdout splits
2. Step 3: mean-pooled Figure 2(c) re-runs under those splits
3. Step 4: Appendix C.5 local-to-global tests
4. Step 5: target parameterization comparison
5. Step 6: probe-strength comparison
6. Step 7: final layer-definition sanity check
7. Step 8: data/rendering only if still needed

## Minimal Summary

The current best explanation is not "the data is wrong" but:

- the published Figure 2(c) most likely depends on an evaluation protocol that
  more strongly tests global, position-invariant direction than the currently
  implemented 5-fold mean-pooled probes do.

## Execution Log

### Step 2 Result

Status: done

Artifacts:

- `artifacts/results/step2_spatial_split_audit.json`

Implemented true spatial-holdout groupings:

1. `pixel_region`
2. `spatial_sector`
3. `spatial_cluster`

Yes/No answer:

- Yes: all three splits are valid 5-group spatial holdouts built from real
  `(pos_x_px, pos_y_px)` coordinates rather than `_pos index` labels.

Observed audit outcome:

- all three methods produce 5 non-empty groups
- fold-level target variance remains non-zero for:
  - `speed`
  - `direction`
  - `acceleration`
- this removes the earlier `condition` split pathology for acceleration

Decision-tree update:

- No branch is resolved yet.
- Step 2 establishes the required spatial-holdout infrastructure for testing
  whether protocol mismatch is the main blocker.

### Step 3 Result

Status: done

Artifacts:

- `artifacts/results/step3_spatial_holdout_summary.csv`
- `artifacts/results/results_step3_pixel_region_sincos.csv`
- `artifacts/results/results_step3_pixel_region_angle.csv`
- `artifacts/results/results_step3_spatial_sector_sincos.csv`
- `artifacts/results/results_step3_spatial_sector_angle.csv`
- `artifacts/results/results_step3_spatial_cluster_sincos.csv`
- `artifacts/results/results_step3_spatial_cluster_angle.csv`

Yes/No answer:

- Yes, partially: true spatial holdout materially changes the curve, but only
  when `direction_target=angle`.

Observed outcome:

- `sincos` remains non-paper-like under all three spatial-holdout splits:
  - `first R^2 >= 0.8` stays at layer 2
  - direction still becomes decodable far too early
- `angle` changes the picture:
  - `pixel_region`: first `R^2 >= 0.8` at layer 8, peak at layer 14
  - `spatial_sector`: first `R^2 >= 0.8` at layer 8, peak at layer 12
  - `spatial_cluster`: first `R^2 >= 0.8` at layer 8, peak at layer 12

Interpretation:

- This is the first strong evidence that protocol mismatch is indeed a major
  blocker.
- However, the result is still not a full Figure 2(c) reproduction because:
  - early-layer direction is not near-zero throughout the pre-PEZ range
  - the direction peak is still too late (`12-14` instead of the sharp PEZ-layer
    behavior one would want)

Decision-tree update:

- Case A is now partially supported:
  - spatial holdout helps substantially
  - but it is not sufficient by itself
- This points directly to Step 4 and Step 5:
  - local-to-global direction analysis
  - target parameterization analysis

### Step 4 Result

Status: done

Artifacts:

- `artifacts/results/step4_local_patch_curve.csv`
- `artifacts/results/step4_cross_region_transfer_curve.csv`
- `artifacts/results/step4_local_global_curves.png`
- `artifacts/results/step4_patch_heatmaps.png`
- `artifacts/results/step4_summary.json`

Experimental design:

- `direction_target=angle`
- `resid_pre`
- velocity dataset only
- local trajectory-following patch feature:
  - for each tubelet, follow the patch containing the ball trajectory
  - average along the trajectory
- cross-region transfer:
  - train on one spatial sector
  - test on unseen spatial sectors

Yes/No answer:

- No: we do not recover a sharp Layer 7->8 local-to-global transition.

Observed outcome:

- local patch probe:
  - layer 5: `R^2 ≈ 0.29`
  - layer 7: `R^2 ≈ 0.37`
  - layer 8: `R^2 ≈ 0.37`
  - layer 10: `R^2 ≈ 0.73`
  - layer 12: `R^2 ≈ 0.82` (peak)
- cross-region transfer:
  - layer 7: `R^2 ≈ -0.30`
  - layer 8: `R^2 ≈ -0.33`
  - layer 10: `R^2 ≈ 0.36`
  - layer 12: `R^2 ≈ 0.52` (peak)

Interpretation:

- This is qualitatively consistent with a local-to-global story:
  - local patch decoding becomes positive earlier
  - cross-region transfer lags behind
- But the transition is shifted late:
  - around `10-12`, not `7-8`

Decision-tree update:

- Case A remains only partially supported.
- Spatial holdout plus local/global diagnostics do not yet reproduce the paper's
  claimed Layer-8 transition.
- This increases the likelihood that target parameterization and/or probe
  implementation details still matter.

### Step 5 Result

Status: done

Artifacts:

- `artifacts/results/step5_target_parameterization_summary.csv`
- `artifacts/results/results_step5_spatial_sector_vxy.csv`

Matched split used:

- `spatial_sector`

Compared targets:

1. `angle`
2. `sincos`
3. `vx/vy` (auxiliary cartesian comparison)

Yes/No answer:

- Yes: the observed PEZ-like behavior is highly sensitive to target
  parameterization.

Observed outcome under the same `spatial_sector` split:

- `angle`
  - first `R^2 >= 0.8` at layer 8
  - peak at layer 12
- `sincos`
  - first `R^2 >= 0.8` at layer 2
  - peak at layer 16
- `vx/vy`
  - first `R^2 >= 0.8` at layer 3
  - peak at layer 12

Interpretation:

- Only `angle` reproduces the Layer-8 onset.
- `sincos` and `vx/vy` both reveal much earlier direction decodability.
- Therefore, the paper-like emergence pattern is not robust across reasonable
  target parameterizations.

Decision-tree update:

- Case A is strengthened only for the specific `angle` target.
- A new constraint is now clear:
  - reproducing Figure 2(c) appears to require a very particular direction
    target parameterization.

### Step 6 Result

Status: done

Artifacts:

- `artifacts/results/results_step6_spatial_sector_angle_adamw100.csv`
- `artifacts/results/results_step6_spatial_sector_angle_ridge.csv`
- `artifacts/results/step6_probe_strength_summary.csv`

Matched setup:

- split: `spatial_sector`
- target: `angle`
- feature: `resid_pre`

Compared solvers:

1. `trainable` (20-HP sweep)
2. `adamw100`
3. `ridge`

Yes/No answer:

- No: the apparent Layer-8 onset is not specific to weak probes.

Observed outcome:

- `trainable`
  - first `R^2 >= 0.8` at layer 8
  - peak at layer 12
- `ridge`
  - first `R^2 >= 0.8` at layer 8
  - peak at layer 14
- `adamw100`
  - first `R^2 >= 0.8` at layer 9
  - peak at layer 13

Interpretation:

- Probe strength changes the curve modestly.
- But the major qualitative effect from Step 5 survives:
  - with the right split and `angle` target, even the paper-style `trainable`
    and a closed-form `ridge` probe both recover a Layer-8 onset
- Therefore the remaining reproduction gap is not primarily a
  `weak-probe-only` artifact.

Decision-tree update:

- Case B is not supported as the main explanation.
- The dominant remaining issues are now:
  - exact layer-definition choice
  - target-definition ambiguity
  - possibly hidden implementation details beyond the public Appendix B text

### Step 7 Result

Status: done

Artifacts:

- `artifacts/results/results_step7_spatial_sector_angle_resid_post.csv`
- `artifacts/results/step7_layer_definition_summary.csv`

Matched setup:

- split: `spatial_sector`
- target: `angle`
- solver: `trainable`

Compared representations:

1. current `resid_pre`
2. strict `resid_post`

Interpretation of current `resid_pre`:

- current `resid_pre` is the practical implementation of
  `patch_embed + pre-block states`

Yes/No answer:

- Yes: the current `resid_pre` definition is clearly the correct side of the
  layer-indexing ambiguity for reproducing Figure 2(c).

Observed outcome:

- current `resid_pre`
  - `L0 direction R^2 ≈ -0.004`
  - first `R^2 >= 0.8` at layer 8
  - peak at layer 12
- strict `resid_post`
  - `L0 direction R^2 ≈ 0.613`
  - first `R^2 >= 0.8` at layer 7
  - peak at layer 11

Interpretation:

- `resid_post` makes early direction decoding far too strong.
- So the earlier off-by-one suspicion has effectively been resolved:
  - the paper-aligned side is `resid_pre`, not `resid_post`
- The remaining mismatch is therefore not primarily a layer-capture bug anymore.

Decision-tree update:

- Layer definition is no longer the leading unresolved cause.
- The main remaining explanations are now:
  - target-definition ambiguity
  - hidden evaluation details not stated in the paper
  - residual data/rendering mismatch only if the above fail

### Step 8 Result

Status: done

Artifacts:

- `artifacts/results/results_step8_spatial_sector_angle_eval_preproc.csv`
- `artifacts/results/step8_final_checks_summary.csv`

Final check performed:

- keep the best current protocol:
  - `spatial_sector`
  - `angle`
  - `resid_pre`
  - `trainable`
- change only preprocessing:
  - `resize`
  - `eval_preproc`

Yes/No answer:

- No: official-like preprocessing does not close the remaining gap.

Observed outcome:

- `resize` best branch:
  - first `R^2 >= 0.8` at layer 8
  - peak at layer 12
- `eval_preproc` check:
  - first `R^2 >= 0.8` at layer 6
  - peak at layer 15

Interpretation:

- preprocessing does not explain the remaining mismatch
- if anything, `eval_preproc` makes the curve less clean for the PEZ narrative

## Final Decision

After Steps 2-8, the most defensible conclusion is:

- We can partially recover a PEZ-like onset only under a narrow protocol:
  - `resid_pre`
  - true spatial holdout
  - `angle` target
- But we still do **not** reproduce the full Figure 2(c) pattern:
  - onset can be moved to layer 8
  - however the peak remains late (`12-15`)
  - the sharp Layer-8 local-to-global transition is not recovered
  - the late-layer weakening seen in the paper is also not faithfully reproduced

Final decision-tree placement:

- Case A: partially supported
- Case B: not supported as the main explanation
- Case C: effectively the current conclusion

Meaning:

- the public paper description is not sufficient to reproduce Figure 2(c)
  exactly
- the strongest remaining explanation is hidden evaluation detail, especially:
  - exact grouping/split semantics
  - exact direction-target implementation
  - possibly other unpublished analysis choices in the internal figure pipeline
