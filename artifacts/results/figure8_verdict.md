# Figure 8 Verdict

Status: `overall-only attentive reproduction partially completed (Large/Huge complete; Giant still running or unavailable)`.

Figure 8 in the paper reports full attentive-MLP results across model sizes. In local reproduction, we implemented a patch-preserving attentive probe on top of `temporal_last_patch` frozen features and evaluated the overall IntPhys possible/impossible task.

## Completed models
- **Large**: relative acc `L0=0.0%`, `L8=82.2%`, peak `88.9% @ L10`, late `78.9%`, first `>=80% @ L8`
- **Huge**: relative acc `L0=0.0%`, `L8=68.9%`, peak `86.7% @ L16`, late `75.6%`, first `>=80% @ L12`

## Outputs
- [figure8_intphys_overall_compare.csv](/home/solee/pez/artifacts/results/figure8_intphys_overall_compare.csv)
- [figure8_intphys_overall_compare.png](/home/solee/pez/artifacts/results/figure8_intphys_overall_compare.png)
- Attentive per-model CSV/PNG/summary under `artifacts/results/results_figure8_intphys_*`
