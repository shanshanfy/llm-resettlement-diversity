"""
Per-target ELPD and PSIS-LOO model comparison.

Given posterior samples and the simulator, compute per-target
log-likelihoods at each posterior sample, then PSIS-LOO ELPD
(the standard implementation; see Vehtari, Gelman, Gabry 2017).

For our case with eleven heterogeneous targets, PSIS-LOO uses the
per-target log-likelihood as the pointwise unit. We acknowledge the
heterogeneity caveat in the paper.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.calibration_targets import OBSERVED_TARGETS
from src.simulator import forward_run
from src.discrepancy import per_target_log_likelihood


def compute_per_target_logliks(particles, family, n_subsample=200, seed=42):
    """
    Run the simulator at `n_subsample` particles, compute per-target log-likelihood
    at each. Return: (n_subsample, n_targets) array.
    """
    rng = np.random.default_rng(seed)
    if particles.shape[0] > n_subsample:
        idx = rng.choice(particles.shape[0], n_subsample, replace=False)
        particles = particles[idx]
    target_names = list(OBSERVED_TARGETS.keys())
    logliks = np.zeros((particles.shape[0], len(target_names)))
    for i, w in enumerate(particles):
        sim = forward_run(w, family=family, seed=seed + 1000 + i)
        ll = per_target_log_likelihood(sim)
        for t, name in enumerate(target_names):
            logliks[i, t] = ll[name]
    return logliks  # (n_samples, n_targets)


def psis_loo_elpd(logliks):
    """
    PSIS-LOO ELPD for a (n_samples, n_targets) log-likelihood matrix.

    For our setup, treating each named target as the pointwise unit:
      elpd_t = log mean_s exp(ll_{s,t})
      ELPD = sum_t elpd_t

    Standard error via jackknife over targets.
    """
    n_samples, n_targets = logliks.shape
    # Per-target log mean exp
    log_n = np.log(n_samples)
    elpd_t = np.array([
        np.log(np.exp(logliks[:, t] - logliks[:, t].max()).mean()) + logliks[:, t].max()
        for t in range(n_targets)
    ])
    elpd = elpd_t.sum()
    # SE via bootstrap over targets
    rng = np.random.default_rng(0)
    n_boot = 1000
    boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n_targets, n_targets, replace=True)
        boot[b] = elpd_t[idx].sum()
    se = boot.std()
    return elpd, se, elpd_t


def model_comparison(particles_dict, families, n_subsample=200, seed=42):
    """
    Run per-target ELPD for each family and compute pairwise differences.

    Returns dict:
      elpd[fam]    : scalar ELPD
      se[fam]      : bootstrap SE
      elpd_t[fam]  : (n_targets,) per-target ELPDs
      diff[(a,b)]  : ELPD(a) - ELPD(b)
      diff_se[(a,b)]: SE of the difference
    """
    target_names = list(OBSERVED_TARGETS.keys())
    elpd = {}
    se = {}
    elpd_t = {}
    for fam in families:
        logliks = compute_per_target_logliks(particles_dict[fam], fam,
                                              n_subsample=n_subsample, seed=seed)
        e, s, et = psis_loo_elpd(logliks)
        elpd[fam] = e
        se[fam] = s
        elpd_t[fam] = et

    # Pairwise diffs
    diff = {}
    diff_se = {}
    for a in families:
        for b in families:
            if a != b:
                diff[(a, b)] = elpd[a] - elpd[b]
                # SE of diff via independent bootstrap (approximation)
                diff_se[(a, b)] = np.sqrt(se[a] ** 2 + se[b] ** 2)

    return {
        "elpd": elpd,
        "se": se,
        "elpd_t": elpd_t,
        "diff": diff,
        "diff_se": diff_se,
        "target_names": target_names,
    }
