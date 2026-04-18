# Figure 8 Verdict

Status: `overall-only attentive reproduction completed for Large/Huge/Giant`.

Figure 8 in the paper reports full attentive-MLP results across model sizes. In local reproduction, we implemented a patch-preserving attentive probe on top of `temporal_last_patch` frozen features and evaluated the overall IntPhys possible/impossible task.

## Completed models
- **Large**: relative acc `L0=0.0%`, `L8=82.2%`, peak `88.9% @ L10`, late `78.9%`, first `>=80% @ L8`; clip acc `L0=50.0%`, `L8=50.6%`, peak `54.2%`
- **Huge**: relative acc `L0=0.0%`, `L8=68.9%`, peak `86.7% @ L16`, late `75.6%`, first `>=80% @ L12`; clip acc `L0=50.0%`, `L8=50.3%`, peak `53.3%`
- **Giant**: relative acc `L0=0.0%`, `L8=68.9%`, peak `87.8% @ L21`, late `81.1%`, first `>=80% @ L1`; clip acc `L0=50.0%`, `L8=52.5%`, peak `55.0%`

## Interpretation
- Large is the most paper-like: `L0=0`, `L8=82.2%`, peak `88.9%@L10`, first `>=80%@L8`.
- Huge and Giant are weaker in this overall-only local reproduction, with onsets later than Layer 8 (`L12` for Huge, `L1` artifact for Giant due to an early spike but main peak late).
- This should be interpreted as an overall-row partial match to paper Figure 8, not a full appendix-faithful recreation of every subtask panel.

## Outputs
- [figure8_intphys_overall_compare.csv](/home/solee/pez/artifacts/results/figure8_intphys_overall_compare.csv)
- [figure8_intphys_overall_compare.png](/home/solee/pez/artifacts/results/figure8_intphys_overall_compare.png)
- [figure8_verdict.md](/home/solee/pez/artifacts/results/figure8_verdict.md)
- Per-model CSV/PNG/summary under `artifacts/results/results_figure8_intphys_*`
