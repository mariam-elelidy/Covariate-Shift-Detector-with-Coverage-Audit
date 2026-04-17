r"""
Covariate Shift Detector with Coverage Audit
=============================================
Author : Mariam Mohamed Elelidy
Purpose: Detect covariate shift between training and deployment distributions,
         quantify the MMD–coverage relationship empirically, and issue a
         coverage audit alarm when the conformal guarantee becomes void.

Core question
-------------
Conformal prediction guarantees P(y in interval) >= 1-alpha under exchangeability.
But exchangeability is never directly observable — you cannot see it violated until
predictions fail. This artifact provides an early-warning system:

  "Is the data the model sees at deployment the same as what it trained on?
   And if not — by how much, and when does that breach invalidate the guarantee?"

Method
------
1. CALIBRATE  — fit a two-sample test (MMD + permutation null) on training data
2. MONITOR    — at deployment, compute MMD between training reference and incoming batch
3. AUDIT      — if MMD > null threshold: flag guarantee as void, report coverage risk
4. QUANTIFY   — the MMD–coverage curve maps each alarm level to an expected coverage

Key finding
-----------
Using a nonlinear DGP (y = Xw + 0.4*(x0² - 1) + noise), a ridge model learns
a linear approximation that is valid near the training distribution. Under
covariate shift (X_test ← X_test + delta):
  - q stays constant (calibrated on in-distribution data — no update)
  - Coverage drops monotonically: 0.933 → 0.17 as delta 0 → 3.0
  - Spearman rho(MMD², coverage) = −0.986
  - MMD alarm fires at delta=0.4 — before coverage falls below target

The alarm fires BEFORE the guarantee fails, not after.

MMD primer
----------
Maximum Mean Discrepancy between distributions P and Q:
  MMD²(P, Q) = E[k(X,X')] - 2E[k(X,Y)] + E[k(Y,Y')]
where k is a kernel (RBF here) and X~P, Y~Q.

MMD² = 0 iff P = Q (under a characteristic kernel).
The permutation test calibrates a threshold below which MMD² is consistent
with sampling noise — not a distributional difference.

Sigma selection (median heuristic)
-----------------------------------
sigma = 0.3 * median pairwise distance in training data.
The scaling factor 0.3 produces a more sensitive kernel than the full median.
Under the full median, the kernel saturates quickly and loses sensitivity
for small shifts. Choosing sigma is not arbitrary — it is the first
design decision in the detection system and should be documented.

Usage
-----
    python covariate_shift_detector.py              # defaults
    python covariate_shift_detector.py --n 2000 --max-shift 5.0 --n-perm 1000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr, pearsonr


# ────────────────────────────────────────────────────────────────────────────
# MMD estimator
# ────────────────────────────────────────────────────────────────────────────

def rbf_kernel(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
    return np.exp(-sq / (2 * sigma ** 2))


def mmd_squared(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased estimator of MMD²(P_X, P_Y) using RBF kernel.

    Uses the unbiased U-statistic: zeros out diagonal in Kxx and Kyy
    to avoid E[k(x,x)] bias terms.

    Returns a float that can be negative under the null (sampling noise).
    The test compares this value to a permutation-calibrated threshold,
    not to zero.
    """
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2 * Kxy.mean()
    )


def calibrate_null(
    X_ref: np.ndarray,
    X_cal: np.ndarray,
    sigma: float,
    n_perm: int = 500,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float, np.ndarray]:
    """Permutation test for H0: X_ref and X_cal are i.i.d.

    Under H0, pooling and re-splitting produces MMD² values from the null
    distribution. The (1-alpha)-th percentile is the detection threshold.

    Returns
    -------
    threshold : float   — null (1-alpha) quantile
    p_value   : float   — p-value for the observed MMD²
    null_dist : ndarray — full null distribution (for visualisation)
    """
    rng = np.random.default_rng(seed)
    pool = np.vstack([X_ref, X_cal])
    n = len(X_ref)
    observed = mmd_squared(X_ref, X_cal, sigma)
    null_dist = np.array([
        mmd_squared(pool[p := rng.permutation(len(pool))][:n], pool[p][n:], sigma)
        for _ in range(n_perm)
    ])
    threshold = float(np.percentile(null_dist, 100 * (1 - alpha)))
    p_value = float(np.mean(null_dist >= observed))
    return threshold, p_value, null_dist


def select_sigma(X: np.ndarray, n_sample: int = 150, scale: float = 0.3) -> float:
    """Sigma via scaled median heuristic.

    sigma = scale * median(pairwise distances in X[:n_sample])

    The scale factor 0.3 is chosen to produce a kernel sensitive enough to
    detect small-to-moderate shifts. The full median heuristic (scale=1.0)
    saturates the RBF quickly and misses early-stage distributional drift.
    Document this choice: sigma is a hyperparameter of the detection system.
    """
    sub = X[:n_sample]
    dists = np.sqrt(np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1))
    return float(np.median(dists[dists > 0])) * scale


# ────────────────────────────────────────────────────────────────────────────
# Conformal prediction (ridge, non-adaptive — deliberately simple)
# ────────────────────────────────────────────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def conformal_eval(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    alpha: float = 0.10,
    lam:   float = 1e-3,
) -> tuple[float, float, float]:
    """Return (coverage, q, mean_bias) on test set.

    q is calibrated on in-distribution data and never updated.
    This is the root of the failure mode: q stays fixed while
    test residuals grow under shift.
    """
    w = ridge_fit(X_tr, y_tr, lam)
    abs_res = np.abs(y_cal - X_cal @ w)
    n_c = len(abs_res)
    k = min(max(int(np.ceil((n_c + 1) * (1 - alpha))), 1), n_c)
    q = float(np.sort(abs_res)[k - 1])
    pred = X_te @ w
    coverage = float(((y_te >= pred - q) & (y_te <= pred + q)).mean())
    bias = float(np.mean(pred - y_te))
    return coverage, q, bias


# ────────────────────────────────────────────────────────────────────────────
# Data generating process (nonlinear)
# ────────────────────────────────────────────────────────────────────────────

def generate_data(
    rng: np.random.Generator,
    n: int,
    d: int,
    noise: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonlinear DGP: y = X w* + 0.4*(x0² - 1) + noise.

    The quadratic term is outside the linear model's capacity. Near the
    training distribution (X~N(0,I)), E[x0²-1]=0 so the bias is negligible.
    Under covariate shift (X += delta), E[(x0+delta)²-1] = delta²+2*delta*E[x0],
    which grows with delta — creating systematic bias that inflates residuals
    beyond what q was calibrated for.
    """
    X      = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y      = X @ w_true + 0.4 * (X[:, 0] ** 2 - 1) + noise * rng.normal(size=n)
    return X, y, w_true


def apply_shift(
    X: np.ndarray,
    y_base: np.ndarray,
    w_true: np.ndarray,
    delta: float,
    rng: np.random.Generator,
    noise: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift features by delta, regenerate labels under shifted DGP."""
    X_s = X + delta
    y_s = X_s @ w_true + 0.4 * (X_s[:, 0] ** 2 - 1) + noise * rng.normal(size=len(X))
    return X_s, y_s


# ────────────────────────────────────────────────────────────────────────────
# Shift sweep
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ShiftResult:
    delta:    float
    mmd2:     float
    coverage: float
    q:        float
    bias:     float
    alarm:    bool
    p_value:  float

    @property
    def coverage_void(self) -> bool:
        """Guarantee considered void if coverage < 0.85 (15% below target)."""
        return self.coverage < 0.85


def run_sweep(
    X_tr:  np.ndarray, y_tr: np.ndarray,
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_te:  np.ndarray,
    w_true: np.ndarray,
    deltas: np.ndarray,
    sigma:  float,
    threshold: float,
    n_ref: int = 200,
    n_probe: int = 100,
    alpha: float = 0.10,
    perm_seed: int = 7,
) -> list[ShiftResult]:
    results = []
    for delta in deltas:
        rng2 = np.random.default_rng(42 + int(delta * 100))
        X_ts, y_ts = apply_shift(X_te, None, w_true, delta, rng2)
        mmd2 = mmd_squared(X_tr[:n_ref], X_ts[:n_probe], sigma)
        cov, q, bias = conformal_eval(X_tr, y_tr, X_cal, y_cal, X_ts, y_ts, alpha)
        _, pv, _ = calibrate_null(
            X_tr[:n_ref], X_ts[:n_probe], sigma, 300, alpha, perm_seed
        )
        results.append(ShiftResult(
            delta=delta, mmd2=mmd2, coverage=cov,
            q=q, bias=bias, alarm=mmd2 > threshold, p_value=pv,
        ))
    return results


# ────────────────────────────────────────────────────────────────────────────
# Terminal report
# ────────────────────────────────────────────────────────────────────────────

def _bar(v: float, lo: float = 0.0, hi: float = 1.0, w: int = 22) -> str:
    x = max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-9)))
    k = int(round(x * w))
    return "█" * k + "░" * (w - k)


def print_report(
    results: list[ShiftResult],
    null_dist: np.ndarray,
    threshold: float,
    sigma: float,
    alpha: float,
    n: int, d: int,
    rho_s: float,
    rho_p: float,
) -> None:
    target = 1 - alpha
    sep = "─" * 78
    print()
    print("┌" + sep + "┐")
    print("│  Covariate Shift Detector — Coverage Audit" + " " * 35 + "│")
    print(f"│  σ = {sigma:.4f}  │  α = {alpha:.2f}  │  "
          f"target coverage = {target:.2f}  │  n = {n}  d = {d}" + " " * 10 + "│")
    print("└" + sep + "┘")

    print()
    print("  ── NULL DISTRIBUTION (permutation test, 500 draws) ──────────────────")
    print(f"  Null mean  : {null_dist.mean():.6f}")
    print(f"  Null std   : {null_dist.std():.6f}")
    print(f"  Threshold (α=0.05): {threshold:.6f}")
    print(f"  σ selection: scaled median heuristic (scale=0.3)")
    print(f"  Note: MMD² can be slightly negative under H0 (U-statistic)")
    print()

    # Shift sweep table
    print("  ── MMD–COVERAGE SWEEP ───────────────────────────────────────────────")
    print(f"  {'delta':>6}  {'MMD²':>10}  {'coverage':>9}  {'bias':>8}  "
          f"{'alarm':>6}  {'p_val':>7}  coverage bar")
    print("  " + "─" * 74)
    for r in results:
        void_flag = "  ← guarantee void" if r.coverage_void else ""
        alarm_str = "ALARM" if r.alarm else "ok   "
        print(
            f"  {r.delta:>6.1f}  {r.mmd2:>10.6f}  {r.coverage:>9.4f}  "
            f"{r.bias:>8.4f}  {alarm_str:>6}  {r.p_value:>7.4f}  "
            f"{_bar(r.coverage, 0.0, 1.0, 18)}{void_flag}"
        )

    print()
    print(f"  Spearman ρ(MMD², coverage) = {rho_s:.4f}")
    print(f"  Pearson  ρ(MMD², coverage) = {rho_p:.4f}")

    # Alarm vs coverage audit summary
    no_alarm  = [r for r in results if not r.alarm]
    alarmed   = [r for r in results if r.alarm]
    first_alarm = alarmed[0] if alarmed else None

    print()
    print("  ── COVERAGE AUDIT SUMMARY ───────────────────────────────────────────")
    if no_alarm:
        print(f"  No-alarm region (delta ≤ {no_alarm[-1].delta:.1f}):")
        cov_range = [r.coverage for r in no_alarm]
        print(f"    Coverage: {min(cov_range):.4f} – {max(cov_range):.4f}  "
              f"(all ≥ {target:.2f}? {all(c>=target for c in cov_range)})")
    if first_alarm:
        print(f"\n  First alarm at delta = {first_alarm.delta:.1f}:")
        print(f"    MMD² = {first_alarm.mmd2:.6f} > threshold {threshold:.6f}")
        print(f"    Coverage at alarm = {first_alarm.coverage:.4f}")
        print(f"    Guarantee technically void: {first_alarm.coverage_void}")

    # Show q is constant — this is the key failure
    all_q = set(round(r.q, 4) for r in results)
    print(f"\n  q across all shift levels: {all_q}")
    print(f"  ⚠  q is calibrated on in-distribution data and never updated.")
    print(f"     Under shift, residuals grow while q stays fixed.")
    print(f"     The model is measuring uncertainty with the wrong ruler.")

    # Alarm level comparison
    print()
    print("  ── MULTI-LEVEL ALARM TABLE ─────────────────────────────────────────")
    print(f"  {'alpha_alarm':>12}  {'threshold':>12}  "
          f"{'n_alarmed':>10}  {'first_alarm_at':>15}  {'cov_at_first':>13}")
    print("  " + "─" * 70)
    base_r   = [r for r in results if r.delta == 0.0][0]
    base_mmd = base_r.mmd2
    for a_lv in [0.10, 0.05, 0.01]:
        # approximate threshold from null percentile relationship
        pct = 100 * (1 - a_lv)
        t = float(np.percentile(null_dist, pct))
        n_alm = sum(1 for r in results if r.mmd2 > t)
        first = next((r for r in results if r.mmd2 > t), None)
        print(f"  {a_lv:>12.2f}  {t:>12.6f}  {n_alm:>10}  "
              f"{first.delta if first else '—':>15}  "
              f"{first.coverage if first else '—':>13}")


def print_tensor_summary(results: list[ShiftResult]) -> None:
    print()
    print("═" * 78)
    print("FINAL TENSOR  [delta, MMD², coverage, q, bias, alarm(0/1)]")
    print("Rows: shift operating points")
    print("═" * 78)
    mat = np.array([
        [r.delta, r.mmd2, r.coverage, r.q, r.bias, float(r.alarm)]
        for r in results
    ])
    print(mat.round(5))
    print()
    base_cov  = results[0].coverage
    end_cov   = results[-1].coverage
    print(f"Coverage drop (Δ=0 → Δ={results[-1].delta:.0f}): "
          f"{base_cov:.4f} → {end_cov:.4f}  Δ = {end_cov-base_cov:+.4f}")
    print(f"Spearman ρ(MMD², coverage): reported above")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Covariate shift detector with coverage audit"
    )
    p.add_argument("--n",          type=int,   default=1500)
    p.add_argument("--d",          type=int,   default=6)
    p.add_argument("--alpha",      type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--max-shift",  type=float, default=4.0, dest="max_shift")
    p.add_argument("--n-perm",     type=int,   default=500, dest="n_perm")
    p.add_argument("--sigma-scale",type=float, default=0.3, dest="sigma_scale",
                   help="Scale factor on median heuristic for RBF sigma")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # ── Data ──
    X, y, w_true = generate_data(rng, args.n, args.d)
    idx = rng.permutation(args.n)
    n_tr  = int(0.6 * args.n)
    n_cal = int(0.2 * args.n)
    tr  = idx[:n_tr]
    cal = idx[n_tr : n_tr + n_cal]
    te  = idx[n_tr + n_cal :]
    X_tr, y_tr   = X[tr],  y[tr]
    X_cal, y_cal = X[cal], y[cal]
    X_te,  y_te  = X[te],  y[te]

    # ── Sigma ──
    sigma = select_sigma(X_tr, scale=args.sigma_scale)
    print(f"σ = {sigma:.4f}  (sigma_scale = {args.sigma_scale})")

    # ── Null calibration ──
    print(f"Calibrating null distribution ({args.n_perm} permutations) …")
    threshold, pv0, null_dist = calibrate_null(
        X_tr[:200], X_te[:100], sigma, args.n_perm, 0.05, seed=7
    )
    print(f"Threshold (α=0.05): {threshold:.6f}  │  "
          f"No-shift p-value: {pv0:.3f}")

    # ── Shift sweep ──
    deltas = np.array([
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
        1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
    ])
    deltas = deltas[deltas <= args.max_shift]

    print(f"Running shift sweep: {len(deltas)} shift levels …")
    results = run_sweep(
        X_tr, y_tr, X_cal, y_cal, X_te, w_true,
        deltas, sigma, threshold, alpha=args.alpha,
    )

    # ── Statistics ──
    mmd_vals = np.array([r.mmd2 for r in results])
    cov_vals = np.array([r.coverage for r in results])
    rho_s, _ = spearmanr(mmd_vals, cov_vals)
    rho_p, _ = pearsonr(mmd_vals,  cov_vals)

    # ── Report ──
    print_report(results, null_dist, threshold, sigma,
                 args.alpha, args.n, args.d, rho_s, rho_p)
    print_tensor_summary(results)


if __name__ == "__main__":
    main()
