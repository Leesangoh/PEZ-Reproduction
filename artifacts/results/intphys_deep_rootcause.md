# IntPhys Figure 1 Deep Root-Cause Analysis

## Scope

This document analyzes why the original IntPhys reproduction initially failed and what changed after checking the paper and the public IntPhys benchmark more carefully.

Goal:

- determine whether the mismatch came from dataset choice, metric choice, feature extraction, frame sampling, or probe implementation
- identify which changes are required to align our implementation with Figure 1 of the PEZ paper

## 1. Paper Setup Re-check

### Figure 1 visual evidence

From the rendered PDF page:

- panel titles:
  - `IntPhys: Linear Probe`
  - `IntPhys: Attentive MLP Probe`
- x-axis: `Layer Fraction`
- y-axis: `Test Accuracy (%)`
- chance line: `Chance (50%)`
- shaded region: one-third emergence zone

Rendered page:

- [/tmp/pez_intphys_fig1_hi-04.png](/tmp/pez_intphys_fig1_hi-04.png)

### Main text

The paper states that:

- the task is to distinguish physically possible vs impossible sequences
- probes are trained as a **binary classification task**
- the videos are **matched possible/impossible pairs**
- possible and impossible videos differ only at a single **breakpoint** frame
- the transition is from near chance (`~50%`) to high performance (`~85–95%`) around one-third depth
- representations peak in the middle third and degrade toward the output

Important implication:

- the figure itself clearly reports **accuracy**
- but the text alone does **not** specify whether this accuracy is clip-level accuracy or scene-level relative accuracy derived from matched sets

### Appendix / probe details

Appendix B says:

- all experiments use linear probes `f(h_l)=Wh+b`
- 20 hyperparameter sweep
  - `lr = {1e-4, 3e-4, 1e-3, 3e-3, 5e-3}`
  - `wd = {0.01, 0.1, 0.4, 0.8}`
- best model chosen by validation performance
- `5-fold grouped cross-validation`
- report `mean ± std`

Appendix C.1 says:

- Figure 6 / Figure 7 are full IntPhys linear-probe results across models
- Figure 10 shows the same one-third signature across the IntPhys sub-principles

Appendix C.11 says:

- for orthogonal-probe dimensionality experiments, IntPhys uses **binary logistic regression** with cross-entropy
- that section uses an `80/20` split and is **not** the main Figure 1 protocol

### What the paper does *not* specify

The PEZ paper does **not** specify:

- whether Figure 1 uses the IntPhys **dev** set or another labeled split
- whether the reported accuracy is:
  - clip-level binary accuracy
  - scene-level grouped relative accuracy
- exact temporal sampling from the 100-frame IntPhys clips
- exact spatial preprocessing for IntPhys
- exact layer capture convention for IntPhys (`resid_pre`, `resid_post`, post-norm, etc.)

## 2. Current Implementation vs Paper

### Our original implementation

Original full-dev implementation:

- data: public IntPhys `dev`
- total clips: `360`
- labels: `status.json -> header.is_possible`
- split: `5-fold GroupKFold`, grouped by scene quadruplet (`block/scene_id`)
- features: `resid_pre`
- pooling: mean over space-time tokens
- frame sampling: `16` frames uniformly sampled from the `100` RGB frames
- metric reported:
  - clip-level binary accuracy
  - AUC

### Direct mismatches / ambiguities

1. **Metric mismatch**
   - paper figure says `Test Accuracy (%)`
   - public IntPhys benchmark is officially evaluated with **scene-group relative / absolute** metrics
   - our first implementation only used clip-level binary accuracy

2. **Temporal sampling mismatch**
   - IntPhys clips have `100` frames at `288x288`
   - our first implementation used only `16` uniformly sampled frames
   - the paper does not say this is what it used
   - since impossible videos differ at a single breakpoint frame, temporal sampling matters

3. **Public split mismatch possibility**
   - public `dev` has only `30 scenes/block`
   - the paper does not explicitly say it used only this public dev split
   - it only says `IntPhys dataset`

4. **Test-set supervision is impossible**
   - the public test videos can be downloaded
   - but test labels are **secret**
   - therefore a supervised layer-wise linear probe cannot be trained/evaluated on public test in the same way as Figure 1

## 3. Evidence For / Against The Pair-wise Metric Hypothesis

### Public benchmark evidence

The IntPhys public challenge uses grouped scene evaluation:

- each scene has a quadruplet of 4 movies
- public docs say test answers are plausibility scores per movie
- `score.py` computes:
  - `relative` error from the grouped possible-vs-impossible sums
  - `absolute` error from AUC

This means that the **benchmark-native evaluation is grouped**, not plain clip accuracy.

### Re-run 1: cached full-dev features, clip-accuracy-selected model

Run:

- [results_intphys_possible_impossible_full_metrics16.csv](/home/solee/pez/artifacts/results/results_intphys_possible_impossible_full_metrics16.csv)

Key numbers:

- clip accuracy:
  - `L0 = 53.6%`
  - `L8 = 73.9%`
  - `peak = 77.2% @ L18`
- scene-relative accuracy:
  - `L0 = 52.2%`
  - `L6 = 91.1%`
  - `L7 = 97.8%`
  - `L8 = 97.8%`
  - `peak = 100%`

Interpretation:

- if we judge the same predictions by **scene-relative accuracy**, the one-third emergence pattern largely appears
- if we judge by **clip accuracy**, it does not

### Re-run 2: cached full-dev features, relative-accuracy-selected model

Run:

- [results_intphys_possible_impossible_full_select_relative.csv](/home/solee/pez/artifacts/results/results_intphys_possible_impossible_full_select_relative.csv)

Key numbers:

- clip accuracy:
  - still only `~73–77%` in the middle / late layers
- scene-relative accuracy:
  - `L5 = 88.9%`
  - `L6 = 95.6%`
  - `L7 = 98.9%`
  - `L8 = 100%`
  - plateau near `100%` through the middle / late layers

Interpretation:

- the pair-wise / grouped metric hypothesis is **strongly supported**
- the original failure was largely caused by evaluating the wrong notion of accuracy

## 4. Evidence For / Against The Temporal Sampling Hypothesis

### Why it is plausible

The paper explicitly says:

- possible and impossible videos differ at a single breakpoint frame

So reducing a `100`-frame IntPhys clip to `16` sampled frames may weaken the signal.

### Direct frame-difference evidence

For example scene groups:

- `O1/01`: strongest difference around frames `37–58`
- `O2/01`: strongest difference around frames `70–79`
- `O3/01`: strongest difference around frames `4–15`

This confirms that the discriminative breakpoint timing varies substantially across scenes.

Implication:

- a fixed sparse sample of 16 frames is not guaranteed to represent all scenes equally well

### 64-frame rerun

Full-dev `64`-frame rerun:

- [results_intphys_possible_impossible_full64_select_relative.csv](/home/solee/pez/artifacts/results/results_intphys_possible_impossible_full64_select_relative.csv)
- [summary_intphys_possible_impossible_full64_select_relative.json](/home/solee/pez/artifacts/results/summary_intphys_possible_impossible_full64_select_relative.json)

Key numbers:

- clip accuracy:
  - `L0 = 52.5%`
  - `L8 = 71.7%`
  - `peak = 73.3% @ L9`
- scene-relative accuracy:
  - `L5 = 93.3%`
  - `L6 = 96.7%`
  - `L7 = 100%`
  - `L8 = 100%`
  - sustained near `100%` through most of the middle and late layers

Interpretation:

- denser temporal sampling does **not** rescue clip accuracy into the paper's `85–95%` range
- but it preserves the same grouped relative-accuracy PEZ transition
- therefore temporal sampling is **secondary**, not primary

Comparison figure:

- [figure_intphys_rootcause_metric_vs_sampling.png](/home/solee/pez/artifacts/results/figure_intphys_rootcause_metric_vs_sampling.png)

At the time of this document update, the run may still be in progress or pending final summary.

## 5. Test Set Feasibility Check

Public IntPhys docs show:

- `test.O1.tar.gz`, `test.O2.tar.gz`, `test.O3.tar.gz` are public downloads
- but `ground truth for the test dataset is not provided and is kept secret for evaluation`

Therefore:

- raw test videos are accessible
- labels are not
- so **supervised linear probing on the public test set is impossible**

This means a paper-faithful supervised Figure 1 reproduction must rely on a labeled split such as public `dev`, or on a private labeled split not available publicly.

## 6. Root Cause Verdict

### Current best verdict

The original IntPhys reproduction failure was **primarily an evaluation mismatch**, not a dataset mismatch.

Most important root causes:

1. **Wrong metric**
   - we first used clip-level binary accuracy
   - the public IntPhys benchmark is naturally scene-grouped
   - scene-relative accuracy produces a PEZ-like Layer-7/8 jump

2. **Underspecified paper wording**
   - the paper says `binary classification task` and shows `Test Accuracy (%)`
   - but it does not state whether that accuracy is clip-level or grouped relative accuracy
   - this ambiguity is enough to change the conclusion completely

3. **Temporal sampling is likely secondary but real**
   - IntPhys videos differ at a variable breakpoint frame
   - sparse 16-frame sampling may blur the task
   - 64-frame rerun is the right follow-up check

4. **Public test set cannot solve this**
   - test labels are private
   - so moving from dev to test does not fix the supervised probe mismatch

## 7. Final Verdict

### Reproduction status

**Partial**

Why:

- under clip-level accuracy, Figure 1 was **not** reproduced
- under scene-relative grouped accuracy, the PEZ-style jump **does** appear around `L7/L8`
- but the paper does not explicitly say which of these two notions of accuracy it uses

### Numbered failure causes with evidence

1. **Metric mismatch**
   - evidence: `L8 clip accuracy = 73.9%` vs `L8 relative accuracy = 97.8–100%` on the same cached full-dev features

2. **Paper ambiguity**
   - evidence: text says `binary classification task`, figure says `Test Accuracy (%)`, but nowhere specifies grouped-vs-clip evaluation

3. **Temporal sampling mismatch**
   - evidence: sampled scene pairs show the strongest possible/impossible divergence at very different frame indices (`~8`, `~58`, `~75`)

4. **Public test-set unusability for supervised probing**
   - evidence: official IntPhys docs state test ground truth is secret

### Additional experiments still needed

1. optionally test whether `resid_post` changes the grouped metric curve
2. if needed, generate an overlay comparing:
   - paper Figure 1 curve (approximate digitization)
   - our clip-accuracy curve
   - our scene-relative curve
