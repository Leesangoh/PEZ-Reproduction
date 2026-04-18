# Possible/Impossible Physics Reproduction

## Paper Target

The paper's possible/impossible physics task is the one shown in **Figure 1**.

- Figure: [Figure 1 page render](/tmp/pez_intphys_fig1-04.png)
- Task: `IntPhys` possible vs impossible discrimination
- Probe type in the main panel:
  - `IntPhys: Linear Probe`
  - `IntPhys: Attentive MLP Probe`

From the paper text:

- Section 4.2 states the possible/impossible task shows a **one-third emergence signature**
- The model transitions from chance (`~50%`) to high performance (`~85–95%`) around the **Physics Emergence Zone**
- The paper also says these representations **peak in the middle third and degrade toward the output**

## What The Paper Says About The Dataset

The PEZ paper itself gives only a concise description:

- dataset: `IntPhys`
- task: binary classification between matched possible/impossible video pairs
- impossible videos violate:
  - object permanence
  - shape constancy
  - spatiotemporal continuity
- crucial detail:
  - possible and impossible videos differ only at a single **breakpoint** frame

Local paper references:

- [pez_paper.pdf](/home/solee/pez/pez_paper.pdf)
- lines around Section 4.1/4.2:
  - probe on `IntPhys`
  - binary classification
  - matched possible/impossible pairs
  - single breakpoint frame

Appendix status:

- Appendix `A.1.1 Intuitive Physics` contains only a dataset example figure
- unlike the synthetic ball task, the PEZ appendix **does not fully specify generation details**

## Public IntPhys Resources

Using the public IntPhys documentation and starting kit:

- public dev split exists with labels
- public train split exists, but is large
- public train video generation code exists
- test generation code is private

Key facts from the public IntPhys docs:

- dev split: `3 GB`
- train split: `15k scenes`
- total dataset size: `~250 GB`
- evaluation is based on matched quadruplets

Local artifacts downloaded during this task:

- `/home/solee/pez/artifacts/intphys/starting_kit.zip`
- `/home/solee/pez/artifacts/intphys/dev.tar.gz`
- extracted dev tree:
  - `/home/solee/pez/artifacts/intphys/dev`

## Implementability Verdict

### Exact PEZ-style reproduction

Not fully implementable in this session with paper-level fidelity, because:

1. the PEZ paper does **not** specify the exact preprocessing / frame sampling / split for IntPhys probes
2. exact reproduction would likely require the larger labeled setup the authors used, not just the public dev split
3. the current `/isaac-sim/python.sh` environment is **CPU-only** here
4. full dev extraction is possible, but slow enough on CPU that it is impractical to complete a faithful full run in this turn

### Practical best-effort reproduction

Implementable:

- use the public IntPhys dev split
- use `status.json` labels
- use grouped 5-fold CV by scene quadruplet
- run a V-JEPA v2-L linear probe

This is what was attempted below.

## Best-Effort Reproduction Attempt

### Implementation

Added script:

- [step_intphys_probe.py](/home/solee/pez/step_intphys_probe.py)

Current pilot settings:

- model: `V-JEPA v2-L`
- feature: `resid_pre`
- preprocessing: `resize -> 256`
- clip sampling: `16` frames uniformly sampled from the `100` IntPhys frames
- split: `5-fold GroupKFold`
- grouping: scene quadruplet (`block/scene_id`)
- probe: trainable linear binary classifier with the same `20`-config `lr x wd` sweep pattern used elsewhere in this repo

### Pilot Result

Outputs:

- [results_intphys_possible_impossible.csv](/home/solee/pez/artifacts/results/results_intphys_possible_impossible.csv)
- [figure_intphys_possible_impossible.png](/home/solee/pez/artifacts/results/figure_intphys_possible_impossible.png)
- [summary_intphys_possible_impossible.json](/home/solee/pez/artifacts/results/summary_intphys_possible_impossible.json)

Pilot summary:

- `L0 accuracy = 62.5%`
- `L8 accuracy = 72.5%`
- `peak accuracy = 85.0% @ layer 15`
- `first layer >= 85% = layer 15`
- `late-layer accuracy = 77.5%`

## Interpretation

This pilot **did not** reproduce Figure 1.

Why:

- the paper's curve rises sharply around the one-third PEZ (`~Layer 8`)
- our pilot does **not** show that
- it only reaches `85%` at **Layer 15**

So under this public-dev subset pilot:

- possible/impossible physics is decodable
- but its emergence is **too late** to match the paper's Figure 1

## Full Dev Run

After the pilot, the same probe was rerun on the **full public dev split** using GPU:

- total clips: `360`
- blocks: `O1`, `O2`, `O3`
- grouping: matched scene quadruplet
- model/probe settings unchanged

Full-dev outputs:

- [results_intphys_possible_impossible_full.csv](/home/solee/pez/artifacts/results/results_intphys_possible_impossible_full.csv)
- [figure_intphys_possible_impossible_full.png](/home/solee/pez/artifacts/results/figure_intphys_possible_impossible_full.png)
- [summary_intphys_possible_impossible_full.json](/home/solee/pez/artifacts/results/summary_intphys_possible_impossible_full.json)

Full-dev summary:

- `L0 accuracy = 53.61%`
- `L7 accuracy = 73.89%`
- `L8 accuracy = 73.89%`
- `peak accuracy = 77.22% @ layer 18`
- `late accuracy = 73.61%`
- `accuracy std near L7-L9 = ~3.4% to 4.9%`

Interpretation:

- there is a real transition from near-chance early layers to much better mid/late performance
- but it is **not** the paper's sharp Figure-1 pattern
- the strongest jump is around `L6 -> L7`, and performance then plateaus rather than sharply peaking at the PEZ
- importantly, the curve never reaches the paper's reported `~85-95%` range on this public-dev setup

So the full-dev verdict is:

- **Figure 1 is not reproduced**

The pilot and the full-dev run agree on the same qualitative outcome:

- possible/impossible becomes more decodable in deeper layers
- but the public-dev reproduction does not show the paper's strong one-third emergence signature

## Most Likely Reasons For The Gap

1. this is only a public-dev subset pilot, not the full paper setup
2. frame sampling for IntPhys is underspecified in the paper
3. exact split semantics are underspecified in the paper
4. CPU-only execution forced a reduced run
5. the paper may have used a larger or different labeled IntPhys slice than the public dev split

## Bottom Line

- **Figure 1 target identified:** yes
- **Public IntPhys dev reproduction path exists:** yes
- **Exact paper-faithful reproduction completed:** no
- **Best-effort pilot attempted:** yes
- **Full public-dev run completed:** yes
- **Full-dev outcome:** not reproduced; the sharp Layer-8 PEZ rise did not appear
