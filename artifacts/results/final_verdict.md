# Final Verdict: Figure 2(c) Reassessment

## Verdict

Under the paper's own wording, the best current run should be treated as a
**qualified reproduction of Figure 2(c)** rather than a failure.

Best run:

- [results_step3_spatial_sector_angle.csv](/home/solee/pez/artifacts/results/results_step3_spatial_sector_angle.csv)

Best current protocol:

- `resid_pre`
- true spatial holdout: `spatial_sector`
- direction target: `angle`
- solver: `trainable`
- preprocessing: `resize`

## Why The Earlier Heuristic Was Wrong

The earlier `paper_distance` heuristic incorrectly forced the **peak layer**
target to `Layer 8`.

That is not what the paper says.

The paper distinguishes:

1. the **Physics Emergence Zone** at about one-third depth
2. a later **peak in the middle layers**
3. then a **decline toward the output layers**

## Paper Evidence

From [pez_paper.pdf](/home/solee/pez/pez_paper.pdf):

- line 53-54:
  - physical variables become accessible at the PEZ and then "degrade toward the output layers"
- line 145-147:
  - PEZ is at "approximately one-third depth"
  - physical variables then "peak in the middle layers"
  - and "degrade toward the output"
- line 633-634:
  - signals "peak and then weaken toward the output layers"

So the correct reading is:

- `Layer 8` is the **onset marker**
- the **peak is not supposed to be at Layer 8**
- the peak should be in the **middle layers**

## Best-Run Check Against That Criterion

From [results_step3_spatial_sector_angle.csv](/home/solee/pez/artifacts/results/results_step3_spatial_sector_angle.csv):

- `direction L0 = -0.0043`
- first `R^2 >= 0.8` at `Layer 8`
- peak at `Layer 12` with `R^2 = 0.9702`
- late layers:
  - `Layer 16 = 0.9626`
  - `Layer 20 = 0.9577`
  - `Layer 23 = 0.9545`

### 1. Does the paper say the signal peaks in the middle layers?

Yes.

The line-147 wording explicitly says:

- physical variables "peak in the middle layers"

### 2. Is our peak at Layer 12 a middle-layer peak for a 24-layer model?

Yes.

For a 24-layer model with layers `0..23`, the midpoint is around `11.5`.
So `Layer 12` is squarely in the middle-layer regime.

### 3. Does first `R^2 >= 0.8` at Layer 8 match the paper's PEZ marker?

Yes.

This is the strongest alignment point with the paper:

- emergence / accessibility threshold occurs at `Layer 8`
- that is exactly the one-third-depth PEZ marker for V-JEPA 2-L

### 4. Is there late-layer weakening?

Yes, but it is mild.

The direction score declines after the middle-layer peak:

- peak `0.9702 @ L12`
- `0.9626 @ L16`
- `0.9577 @ L20`
- `0.9545 @ L23`

So the curve does show:

- onset at PEZ
- middle-layer peak
- late-layer weakening

The weakening is weaker than one might visually expect from the paper, but it
is present and directionally consistent with the paper's wording.

## Re-Judgment of Case A / B / C

### Case A

Supported.

The main thing that unlocked the paper-like pattern was:

- true spatial holdout
- `angle` direction target
- `resid_pre`

That means the main earlier failure was indeed a **protocol mismatch**.

### Case B

Not supported as the main explanation.

The Layer-8 onset survives under:

- `trainable`
- `ridge`

So this is not merely a weak-probe artifact.

### Case C

No longer the right top-level verdict for Figure 2(c) itself.

Case C would mean the paper description is still insufficient even after the
critical protocol corrections. That is too strong now, because the best run
already matches the paper's own onset/peak/decline wording reasonably well.

## Updated Final Judgment

For **Figure 2(c) specifically**, the best current verdict is:

- **Substantial qualitative reproduction achieved**
- **Not a perfect or robust reproduction across all reasonable settings**
- **But no longer a clean reproduction failure**

In practical terms:

- if the criterion is "exactly replicate every curve shape under the original
  naive setup", then no
- if the criterion is "recover the PEZ pattern described in the paper:
  onset at Layer 8, peak in middle layers, then weaken toward output", then
  **yes, the best run does that**

## Most Accurate One-Line Summary

The correct updated verdict is:

- **Figure 2(c) is reproducible in a qualified sense once the evaluation
  protocol is corrected, and the strongest previous 'failure' claim came from
  using the wrong success criterion for the peak layer.**
