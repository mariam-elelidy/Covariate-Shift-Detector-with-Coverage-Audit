# Covariate Shift Detector with Coverage Audit

> *The conformal guarantee does not fail loudly. It continues reporting 90% confidence while achieving 2.7% coverage. Monitoring the guarantee requires monitoring the assumption it depends on — not monitoring the model output.*

---

## What this is

An MMD-based covariate shift detector that converts a theoretical assumption violation (non-exchangeability between calibration and test data) into a measurable alarm — with empirical evidence that the alarm fires **before** conformal coverage degrades below target.

This is the final layer in the reliability series: the deployment-layer question that precedes all statistical guarantees. Every repo before this one assumed the data distribution was stable. This repo measures whether that assumption holds.

Part of a series on measurable reliability in ML:
[Mathematical Reliability](https://github.com/mariam-elelidy/Mathematical-Reliability-for-ML-Predictions) · [Assumption Stress Harness](https://github.com/mariam-elelidy/Assumption-Stress-Harness) · [Influence & Stability](https://github.com/mariam-elelidy/Influence-Stability-Analysis-for-ML-Predictions) · [Calibration](https://github.com/mariam-elelidy/Calibration-as-a-Measurable-Reliability-Constraint) · [Selective Prediction](https://github.com/mariam-elelidy/Selective-Prediction-Under-Uncertainty)

---

## Core result (n=1500, d=6, seed=42)

| Shift (delta) | MMD² | Coverage | Alarm | Status |
|---|---|---|---|---|
| 0.0 | −0.00130 | **0.9333** | no | on target |
| 0.2 | 0.00119 | 0.9333 | no | on target |
| **0.4** | **0.00867** | 0.9400 | **ALARM** | early warning |
| 0.6 | 0.01938 | 0.8933 | ALARM | below target |
| 1.0 | 0.04213 | 0.8400 | ALARM | guarantee void |
| 2.0 | 0.06559 | 0.5100 | ALARM | guarantee void |
| 4.0 | 0.06652 | **0.0267** | ALARM | total failure |

**Spearman ρ(MMD², coverage) = −0.9857** — empirical, not assumed.  
**q = 1.1953 at every shift level.** The model never knows it is failing.

---

## Quick start

```bash
pip install numpy scipy

python covariate_shift_detector.py                       # defaults
python covariate_shift_detector.py --max-shift 5.0 --n-perm 1000
python covariate_shift_detector.py --sigma-scale 0.5    # less sensitive kernel
```

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `--n` | 1500 | Dataset size |
| `--d` | 6 | Feature dimension |
| `--alpha` | 0.10 | Conformal miscoverage level |
| `--seed` | 42 | Random seed |
| `--max-shift` | 4.0 | Maximum shift magnitude to sweep |
| `--n-perm` | 500 | Permutations for null calibration |
| `--sigma-scale` | 0.3 | Scale factor on median heuristic for RBF sigma |

---

## How it works

```
Training reference set (X_tr[:200])
        │
        ├──► Sigma selection: σ = 0.3 × median pairwise distance
        │
        ├──► Null calibration: 500 random splits of pooled data
        │    → threshold τ = 95th percentile of MMD² null distribution
        │
        └──► Conformal calibration: q on in-distribution data

Deployment monitoring — per incoming batch:
        │
        ├──► Compute MMD²(X_tr_ref, X_batch)
        │
        ├──► If MMD² > τ: ALARM — exchangeability assumption void
        │    │
        │    └──► Coverage audit: report expected coverage degradation
        │
        └──► If MMD² ≤ τ: no alarm — conformal guarantee holds
```

---

## The silent failure (why this matters)

At delta = 4.0:
- Model reports intervals of width **2.391** (unchanged since calibration)
- Model is wrong by **6.178 units** on average
- Only **2.67%** of predictions are within their intervals
- **No internal signal** that anything has changed

q is calibrated once on in-distribution data. It stays frozen. The model has no mechanism to detect distributional drift — it will continue reporting 90% confidence intervals regardless of how far deployment data has shifted.

The alarm from this detector is the only signal that the conformal guarantee has become void.

---

## Key findings

**Alarm fires before coverage fails.** At delta = 0.4: alarm = YES, coverage = 0.940 (still above target). At delta = 1.0: coverage = 0.840 (below target). Early-detection window: delta 0.4 → 1.0.

**ρ = −0.986 is empirical, not theoretical.** The Assumption Stress Harness showed structurally that covariate shift degrades coverage. This repo measures the relationship with Spearman ρ = −0.986 across a continuous shift range.

**MMD saturates; coverage does not.** MMD² plateaus at ~0.0665 starting at delta = 2.5. Coverage continues to collapse to 0.027 at delta = 4.0. At large shifts, MMD signals "severe violation" but cannot quantify how much coverage remains.

**Sigma is a design decision.** Full median (sigma = 3.27): small shifts missed, no early warning. Scaled (sigma = 0.98): delta = 0.4 detected at p < 0.001. Sigma must be fixed before deployment and documented alongside the alarm threshold.

---

## Outputs

| Output | What it answers |
|---|---|
| Null distribution stats + threshold | "What does no-shift MMD² look like? When is the alarm calibrated to fire?" |
| MMD–coverage sweep table | "As shift increases, how does MMD² change and how does coverage degrade?" |
| ρ(MMD², coverage) | "How predictive is the MMD alarm of actual coverage loss?" |
| q across shift levels | "Does the model know it is failing?" (answer: no) |
| Multi-level alarm table | "When does the alarm fire at different significance levels?" |
| Final tensor `[delta, MMD², coverage, q, bias, alarm]` | Machine-readable sweep for analysis |

---

## Connection to the series

| Artifact | Reliability layer |
|---|---|
| Mathematical reliability | Coverage guarantee under ideal conditions |
| Assumption stress harness | Coverage under assumption violations |
| Influence & stability | Training data composition reliability |
| Calibration decomposition | Probability reliability |
| Selective prediction | Prediction usefulness and subgroup reliability |
| **This artifact** | Deployment-layer assumption monitoring |

Each layer is necessary. None is sufficient alone.

---

## Repository layout

```
├── README.md                      ← this file
├── covariate_shift_detector.py    ← implementation
├── output.txt                     ← annotated run output
└── writeup.md                     ← full technical writeup
```

---

## References

- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *JMLR*, 13, 723–773.
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *NeurIPS*.

---

*Final repo in the Mathematical Reliability for ML series.*
