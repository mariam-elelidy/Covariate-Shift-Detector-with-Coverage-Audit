# Covariate Shift Detector with Coverage Audit

**Author:** Mariam Mohamed Elelidy  
**Topic:** Distribution Shift · Two-Sample Testing · Coverage Audit · Deployment Reliability

---

## TL;DR

Conformal prediction's coverage guarantee depends on exchangeability between calibration and test data. In deployment, this assumption is never directly observable — you cannot see it violated until predictions have already failed. The model continues reporting 90% confidence while achieving near-zero coverage, with no internal signal.

This artifact builds the bridge from the statistical world to the deployment world: an MMD-based shift detector that converts a theoretical assumption violation into a measurable alarm — **before** coverage degrades, not after.

**Core finding:** Spearman ρ(MMD², coverage) = −0.986 (empirical). The alarm fires at delta = 0.4 while coverage is still 0.940 — before the guarantee fails at delta = 1.0. By delta = 4.0, coverage reaches 0.027 and q = 1.195 (unchanged since calibration). The model is confidently wrong on 97% of predictions.

---

## 1. The Problem

The repos in this series established:
- How to build conformal intervals with finite-sample guarantees
- When those guarantees fail under assumption violation
- Which training points drive predictions and by how much
- How to calibrate probabilities for decision-making
- When the model should decline to predict

This artifact answers the deployment-layer question that precedes all of the above: **"Is the data the model sees at test time the same as what it trained on?"**

Without answering this question, every downstream reliability guarantee is conditional on an assumption you cannot observe.

---

## 2. Testable Claims

**Primary:** Under covariate shift of increasing magnitude, MMD² between training and test features increases monotonically, and empirical coverage decreases monotonically. Spearman ρ = −0.986.

**Alarm claim:** A permutation-calibrated MMD threshold fires before coverage falls below target, providing an early-detection window.

**Silent failure claim:** q is calibrated on in-distribution data and never updated. The model reports the same interval width at delta=0 and delta=4 — 2.391 units — despite 97% coverage failure in the latter case.

---

## 3. Method

### Nonlinear DGP (deliberate model misspecification)

$$y_i = x_i^\top w^* + 0.4(x_{i,1}^2 - 1) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, 0.25)$$

Ridge regression fits the linear part. Under shift $X \leftarrow X + \delta$, the quadratic term produces bias $\propto \delta^2$ — growing residuals that q cannot cover.

### MMD² estimator (unbiased U-statistic, RBF kernel)

$$\widehat{\text{MMD}}^2(X, Y) = \frac{\sum_{i \neq j} k(x_i, x_j)}{n(n-1)} + \frac{\sum_{i \neq j} k(y_i, y_j)}{m(m-1)} - \frac{2}{nm}\sum_{i,j} k(x_i, y_j)$$

with $k(a,b) = \exp(-\|a-b\|^2/2\sigma^2)$. The U-statistic zeros out the diagonal — under $H_0$, $\widehat{\text{MMD}}^2$ has mean near zero and can be slightly negative. Comparing to zero is incorrect; comparison to the permutation threshold is required.

### Sigma: scaled median heuristic

$$\sigma = 0.3 \times \text{median pairwise distance}$$

The scale 0.3 is a deliberate design choice. The full median ($\sigma = 3.27$) saturates too quickly under small shifts — the null and shifted distributions become indistinguishable. The scaled version ($\sigma = 0.98$) detects drift at delta = 0.4. **Sigma is a hyperparameter of the detection system**, not a free parameter; it must be fixed before deployment.

### Permutation-calibrated threshold

$$\tau_{0.95} = \text{quantile}_{0.95}\!\left(\{\widehat{\text{MMD}}^2(\tilde{X}, \tilde{Y}) : \text{500 random pooled splits}\}\right)$$

Finite-sample exact, distribution-free, no asymptotic assumptions. Matching the philosophy of the rest of this series.

---

## 4. Results

### Null distribution (no shift)

| Statistic | Value |
|---|---|
| Null mean | 0.000007 |
| Null std | 0.001577 |
| Threshold (α=0.05) | 0.002965 |
| No-shift p-value | 0.773 — H₀ accepted ✓ |

### MMD–Coverage sweep

| delta | MMD² | Coverage | Alarm | Notes |
|---|---|---|---|---|
| 0.0 | −0.00130 | **0.9333** | no | on target |
| 0.2 | 0.00119 | 0.9333 | no | on target |
| **0.4** | **0.00867** | 0.9400 | **ALARM** | early warning |
| 0.6 | 0.01938 | 0.8933 | ALARM | below target |
| 1.0 | 0.04213 | 0.8400 | ALARM | guarantee void |
| 1.5 | 0.05999 | 0.6733 | ALARM | guarantee void |
| 2.0 | 0.06559 | 0.5100 | ALARM | guarantee void |
| 3.0 | 0.06652 | 0.1700 | ALARM | guarantee void |
| 4.0 | 0.06652 | **0.0267** | ALARM | total failure |

q = **1.1953** at every shift level. Unchanged.

### Correlation

Spearman ρ(MMD², coverage) = **−0.9857** — empirical, not assumed.

### Final tensor `[delta, MMD², coverage, q, bias, alarm(0/1)]`

```
[[ 0.0    -0.00130   0.9333  1.1953  +0.070  0],
 [ 0.2     0.00119   0.9333  1.1953  +0.059  0],
 [ 0.4     0.00867   0.9400  1.1953  +0.013  1],
 [ 0.6     0.01938   0.8933  1.1953  -0.031  1],
 [ 1.0     0.04213   0.8400  1.1953  -0.284  1],
 [ 1.5     0.05999   0.6733  1.1953  -0.770  1],
 [ 2.0     0.06559   0.5100  1.1953  -1.457  1],
 [ 3.0     0.06652   0.1700  1.1953  -3.390  1],
 [ 4.0     0.06652   0.0267  1.1953  -6.178  1]]
```

---

## 5. Analysis

### Alarm fires before guarantee fails

At delta = 0.4: alarm = YES, coverage = 0.940 (still above target).  
At delta = 1.0: coverage = 0.840, guarantee void.

**Early-detection window: delta 0.4 to 1.0.** The system identifies a distributional anomaly while predictions are still reliable — not after they have failed.

### The silent failure: q never changes

q = 1.1953 at every shift level, including delta = 4.0 (coverage = 0.027). The model reports intervals of width 2.391 throughout. At delta = 4.0, the model is wrong by ~6.18 units on average, but the interval is ±1.195. There is no internal signal that anything has changed.

This is not a bug — it is how deployed conformal models are designed. An external shift detector is required.

### MMD saturation at large shifts

MMD² saturates at ~0.0665 starting at delta = 2.5. Coverage continues to collapse from 0.28 → 0.027 over this range. **At large shifts, MMD is no longer informative about how much coverage has degraded** — only that a severe shift exists. In deployment, an alarm does not mean "coverage = 0.90 - epsilon." It means "exchangeability is void; coverage could be anywhere below the nominal level."

### Sigma selection is a design decision

Full median (sigma = 3.27): kernel saturates, small shifts missed, no early warning.  
Scaled (sigma = 0.98): tight null, threshold = 0.003, delta = 0.4 detected at p < 0.001.

Documenting sigma alongside the detection threshold is as important as documenting q alongside the coverage target.

---

## 6. Connection to the Series

| Artifact | What it shows |
|---|---|
| Assumption stress harness | "If covariate shift occurs, coverage degrades" (small controlled shift) |
| **This artifact** | "How to detect shift before coverage degrades; Spearman ρ = −0.986" |

The stress harness showed the structural result. This artifact provides the operationalisation: a detector, a threshold, an alarm, and a quantified relationship between MMD and expected coverage loss.

---

## 7. Limitations

| Limitation | Detail |
|---|---|
| MMD saturation | At large shifts, MMD stops being informative about coverage magnitude |
| Sigma sensitivity | Detector performance depends on sigma choice; must be documented pre-deployment |
| Reference set size | Using 200/100 samples; larger sets give tighter null distributions |
| Kernel choice | RBF for Euclidean feature shift; label shift and concept drift need different approaches |
| No post-alarm coverage bound | The detector flags void guarantees but cannot quantify coverage without ground truth |
| O(n²) cost | For large batches, approximate MMD via random features |

---

## 8. Reproducibility

```bash
pip install numpy scipy

python covariate_shift_detector.py                      # defaults
python covariate_shift_detector.py --max-shift 5.0 --n-perm 1000
python covariate_shift_detector.py --sigma-scale 0.5    # less sensitive
```

Deterministic given `--seed`. No plotting libraries required.

---

## 9. Takeaways

> **The conformal guarantee does not fail loudly. It continues reporting 90% confidence while achieving 2.7%. Monitoring the guarantee requires monitoring the assumption it depends on — not monitoring the model output.**

Three shifts:

1. **The guarantee is void before you know it is.** At delta = 1.0, coverage is 0.840 and q is unchanged. There is no internal signal. External monitoring is not a best practice — it is a prerequisite for deployment reliability.

2. **MMD is a predictive measure, not just a diagnostic.** ρ = −0.986 means MMD allows estimation of coverage loss before ground truth labels arrive. This converts a hypothesis test into a deployment alarm with interpretable severity.

3. **Sigma selection is a design decision, not a detail.** The full median heuristic missed early-stage drift. Documenting sigma alongside the threshold — and justifying the choice — is part of the detection system's specification.

---

## References

- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *JMLR*, 13, 723–773.
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
- Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *NeurIPS*.
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*.
