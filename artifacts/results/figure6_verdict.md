# Figure 6 Verdict

## Status

`overall-only` reproduction completed for `Large`, `Huge`, and `Giant`.

Full paper Figure 6 is **not fully reproduced** because the local public IntPhys resources do not expose the subtask mapping needed for the lower three rows (`Object Permanence`, `Shape Constancy`, `Spatiotemporal Continuity`).

## Outputs

- [figure6_intphys_overall_compare.csv](/home/solee/pez/artifacts/results/figure6_intphys_overall_compare.csv)
- [figure6_intphys_overall_compare.png](/home/solee/pez/artifacts/results/figure6_intphys_overall_compare.png)
- Huge Kubric feature cache: [/home/solee/pez/artifacts/features/resid_pre_resize/vjepa2_H](/home/solee/pez/artifacts/features/resid_pre_resize/vjepa2_H)

## Model summaries

### Large

- clip acc: `L0=51.9%`, `L8=73.6%`, `peak=76.9% @ L18`
- relative acc: `L0=73.3%`, `L8=100.0%`, `peak=100.0%`

### Huge

- clip acc: `L0=51.7%`, `L8=59.7%`, `peak=79.7% @ L17`
- relative acc: `L0=62.2%`, `L8=96.7%`, `peak=100.0%`

### Giant

- clip acc: `L0=51.4%`, `L8=60.0%`, `peak=78.9% @ L37`
- relative acc: `L0=70.0%`, `L8=81.1%`, `peak=100.0%`

## Interpretation

- Under clip accuracy, all three models rise more slowly than the paper appendix plots and peak later than the PEZ marker.
- Under scene-relative accuracy, all three models show a strong one-third-style transition, but the exact plateau/decline profile still depends on the metric choice.
- Therefore the strongest reproducible statement is that Figure 6 is only matched as an **overall-row approximation**, not as the full appendix figure.