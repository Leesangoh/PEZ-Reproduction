# Figure 2(b) Failure Analysis

## Verdict

Figure 2(b) is **not reproduced** in the current setup. The strongest mismatch is that:

- `velocity_xy` reaches `R^2 >= 0.8` by **Layer 3**
- `acceleration_xy` reaches `R^2 >= 0.8` by **Layer 5**

Best current run:

- `velocity_xy`: [results_fig2b_velocity_xy_spatial_sector.csv](/home/solee/pez/artifacts/results/results_fig2b_velocity_xy_spatial_sector.csv)
- `acceleration_xy`: [results_fig2b_acceleration_xy_spatial_sector.csv](/home/solee/pez/artifacts/results/results_fig2b_acceleration_xy_spatial_sector.csv)

Paper Figure 2(b) instead describes a **transition at the Physics Emergence Zone** around Layer 8.

## What The Current Data/Probe Are Doing

The current reproduction makes Cartesian targets unusually easy:

- Targets are **constant per video** and read at `frame_idx = 0`
- `velocity_xy` uses screen-space `(vx_px, vy_px)`
- `acceleration_xy` uses screen-space `(ax_px, ay_px)`
- There are only:
  - **56 unique** `(vx_px, vy_px)` pairs in the velocity dataset
  - **40 unique** `(ax_px, ay_px)` pairs in the acceleration dataset
- The scene is a **single ball**, fixed camera, fixed background, no distractors, no occlusion, no camera motion
- Spatial holdout changes the start position, but the target itself is almost position-independent

This means the probe is closer to decoding a small set of global motion templates than to solving a hard physical reasoning problem.

## Most Likely Failure Causes

### 1. Cartesian targets are too linearly aligned with early visual motion features

Why this is plausible:

- `(vx, vy)` and `(ax, ay)` do not have angle wraparound or polar decomposition ambiguity
- Early spatiotemporal filters can directly align with image-axis motion
- In our runs, the jump is immediate:
  - `velocity_xy`: `L0=0.0717`, `L1=0.6392`, `L3=0.8045`
  - `acceleration_xy`: `L0=-0.0699`, `L1=0.6299`, `L5=0.8057`

Why this hurts reproduction:

- Figure 2(b) in the paper still shows a PEZ transition, but our setup lets the model read the Cartesian code before Layer 8.

How to test:

- Hold out **unseen directions** or **unseen magnitudes** instead of only unseen positions
- Compare `velocity_xy` against a harder target such as:
  - normalized direction vector
  - future displacement after an offset horizon
  - frame-averaged vs frame-0 target definitions

### 2. The synthetic scene is visually too simple, so low layers already solve the task

Why this is plausible:

- The ball is the only moving object
- Camera and background are fixed
- Motion direction maps almost directly to the local streak / displacement of a single object
- No clutter, no occlusion, no competing motion sources

Why this hurts reproduction:

- In a simple single-object scene, patch embedding and the first few blocks may already carry enough motion information for Cartesian decoding.

How to test:

- Add controlled nuisance variation:
  - textured floor/background
  - lighting changes
  - distractor objects
  - slight camera jitter
- Re-run the same probe and check whether `first R^2 >= 0.8` shifts later.

### 3. Spatial holdout is not a hard enough split for Cartesian variables

Why this is plausible:

- The three spatial holdouts are almost identical in result:
  - `velocity_xy` first `>= 0.8`: always **Layer 3**
  - `acceleration_xy` first `>= 0.8`: always **Layer 5**
- Start position changes, but `(vx, vy)` and `(ax, ay)` remain the same discrete set of target vectors
- Correlation between start position and target is near zero, so holding out positions does not force new motion generalization

Why this hurts reproduction:

- The split may still allow the probe to learn a generic motion-to-vector map that transfers trivially across position.

How to test:

- Replace spatial holdout with:
  - leave-two-directions-out 5-fold
  - leave-one-speed-bin-out / leave-one-acceleration-bin-out
  - joint spatial + direction holdout
- If onset moves toward Layer 8 only under these harder splits, the current split is the main reason Figure 2(b) fails.

### 4. Our target definition is screen-space and episode-constant, which may be easier than the paper’s effective task

Why this is plausible:

- We probe `frame_idx = 0` targets, not a time-varying per-frame or per-window Cartesian signal
- `acceleration_xy` is computed as the **mean screen-space delta of velocity** over the clip
- That turns the task into classification/regression over a stable per-video label, not a sequence-dependent decoding problem

Why this hurts reproduction:

- A stable per-video target lets early layers succeed once they have any coarse motion evidence.

How to test:

- Compare:
  - `frame 0` target
  - clip mean target
  - mid-clip target
  - per-frame / per-window target with grouped CV over windows
- If only the episode-constant target saturates early, this is a major cause.

### 5. Mean pooling plus a strong probe may collapse local motion cues too effectively

Why this is plausible:

- Current setup uses spatiotemporal mean pooling over the full clip
- The trainable 20-HP sweep is a strong readout
- Mean pooling can turn many weak local motion signals into a very linearly separable global summary

Why this hurts reproduction:

- The paper’s Figure 2(b) may depend on a stricter measurement of where the representation becomes globally available, whereas our pipeline may expose it too early.

How to test:

- Re-run Figure 2(b) with:
  - ridge
  - weaker AdamW100
  - per-patch probes
  - region-transfer probes
- If early saturation survives all of these, the issue is more likely in the data than the probe.

## Current Ranking Of Causes

Most likely:

1. Cartesian target is intrinsically too easy in this dataset
2. Spatial holdout is too weak for Cartesian vectors
3. Scene simplicity lets early layers solve the task

Secondary:

4. Episode-constant target definition may simplify the problem
5. Strong mean-pooled readout may amplify shallow cues

## Immediate Next Checks

If we want to push Figure 2(b) further, the highest-value experiments are:

1. **Leave-directions-out Cartesian probe**
   - tests whether the current split is too easy
2. **Leave-magnitudes-out Cartesian probe**
   - tests whether the probe is just memorizing the discrete velocity/acceleration set
3. **Per-window Cartesian target**
   - tests whether episode-constant labels are the issue
4. **Visual complexity ablation**
   - tests whether the scene is too easy
5. **Per-patch / region-transfer probe**
   - tests whether early success is local rather than global

## Bottom Line

The current Figure 2(b) failure is most plausibly **not** a single implementation bug. It is more likely that our Cartesian task is simply too easy under the current synthetic setup:

- discrete low-cardinality Cartesian labels
- constant per-video targets
- single-object, fixed-camera scene
- spatial holdout that does not hold out target directions or magnitudes

Under those conditions, early layers can already linearly decode `(vx, vy)` and `(ax, ay)` far before the paper’s Layer 8 transition.
