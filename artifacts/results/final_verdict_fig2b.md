# Final Verdict: Figure 2(b) Cartesian Reproduction

## Verdict

Figure 2(b) is **not reproduced at the same level as Figure 2(c)**.

Best current setup:

- `resid_pre`
- `5-fold`
- true spatial holdout
- `trainable` 20-HP sweep

Best grouping overall:

- `spatial_sector`

Main outputs:

- [fig2b_ablation_summary.csv](/home/solee/pez/artifacts/results/fig2b_ablation_summary.csv)
- [fig2b_overlay.png](/home/solee/pez/artifacts/results/fig2b_overlay.png)
- [results_fig2b_velocity_xy_spatial_sector.csv](/home/solee/pez/artifacts/results/results_fig2b_velocity_xy_spatial_sector.csv)
- [results_fig2b_acceleration_xy_spatial_sector.csv](/home/solee/pez/artifacts/results/results_fig2b_acceleration_xy_spatial_sector.csv)

## Paper Expectation

The requested Figure 2(b) reading was:

- `velocity_xy`: decodable from the earliest layers
- `acceleration_xy`: jump at the PEZ
- peak in the middle layers
- weaken toward the output

## What We Actually Get

### Velocity XY

Best spatial-sector run:

- `L0 = 0.0717`
- `L1 = 0.6392`
- `L3 = 0.8045`
- `L8 = 0.9537`
- peak `0.9864 @ L12`
- `L23 = 0.9685`

Interpretation:

- velocity becomes decodable very early
- but **not from Layer 0 in the strong sense expected**
- instead it rises sharply by `L1-L3`

### Acceleration XY

Best spatial-sector run:

- `L0 = -0.0699`
- `L1 = 0.6299`
- `L5 = 0.8057`
- `L8 = 0.9391`
- peak `0.9799 @ L12`
- `L23 = 0.9717`

Interpretation:

- acceleration does show a strong rise
- but it happens **too early**
- the first `R^2 >= 0.8` appears at `L5`, not at the PEZ marker `L8`

## What Matches The Paper

- both variables peak in the middle layers (`L12-L13`)
- both variables weaken slightly toward the output layers
- spatial-holdout groupings are stable across the three variants

## What Does Not Match

- `velocity_xy` is not strongly present at `L0`
- `acceleration_xy` does not wait for the PEZ; it becomes highly decodable by `L5`

These are the two core reasons the Cartesian result should not be called a
qualified reproduction.

## Bottom Line

The current Cartesian result is best described as:

- **partial qualitative match**
- **not a full Figure 2(b) reproduction**

More specifically:

- the overall curve shape is plausible
- but the timing of early availability vs PEZ-linked emergence is off for both
  variables in opposite directions

That makes Figure 2(b) materially weaker than the best Figure 2(c) result.
