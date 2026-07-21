"""
Sensitivity analysis: zero out the ethnic-clustering weight (zeta) in
Pi3's posterior and recompute predictions.

Reports per-site KL divergence between the full and ablated posterior
predictives, and KL of each from Pi1's posterior predictive.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.sites import N_SITES
from src.simulator import forward_run
from src.decision_rules import PI_FAMILIES


def posterior_predictive_occupancy(particles, family, n_subsample=200, seed=42, ablate_zeta=False):
    """
    For each posterior particle, run forward simulator and record three-site
    occupancy. Return (n_subsample, N_SITES) array of occupancies.
    """
    rng = np.random.default_rng(seed)
    if particles.shape[0] > n_subsample:
        idx = rng.choice(particles.shape[0], n_subsample, replace=False)
        particles = particles[idx]

    # If ablating zeta, zero the zeta column (5th index, position 4 for Pi3)
    if ablate_zeta and family == "pi3":
        particles = particles.copy()
        zeta_idx = PI_FAMILIES["pi3"].index("zeta")
        particles[:, zeta_idx] = 0.0

    occ = np.zeros((particles.shape[0], N_SITES))
    for i, w in enumerate(particles):
        sim = forward_run(w, family=family, seed=seed + 2000 + i)
        occ[i] = sim["three_site_occupancy"]
    return occ


def kl_div_dirichlet(p_samples, q_samples, n_bins=20):
    """
    Approximate KL(p || q) between two posterior predictive distributions
    over a 3-simplex. We discretise into bins along each axis.
    """
    # Use a kernel-density-like binning
    p_samples = np.asarray(p_samples)
    q_samples = np.asarray(q_samples)
    # Bin into 1D for the focal site (e.g., distant site)
    return None  # we use per-site KL below


def per_site_kl(p_samples, q_samples, n_bins=30, eps=1e-6):
    """
    For each site index, compute KL between the marginal histograms of
    p and q. Returns (N_SITES,) array of KL values.
    """
    out = np.zeros(N_SITES)
    edges = np.linspace(0, 1, n_bins + 1)
    for s in range(N_SITES):
        p_hist, _ = np.histogram(p_samples[:, s], bins=edges, density=False)
        q_hist, _ = np.histogram(q_samples[:, s], bins=edges, density=False)
        p_hist = p_hist.astype(float) + eps
        q_hist = q_hist.astype(float) + eps
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()
        out[s] = np.sum(p_hist * np.log(p_hist / q_hist))
    return out


def sensitivity_table(particles_pi1, particles_pi3, n_subsample=200, seed=42):
    """
    Returns dict:
      occ_full      : (n, 3) occupancy under full Pi3
      occ_ablated   : (n, 3) occupancy under Pi3 with zeta=0
      occ_pi1       : (n, 3) occupancy under Pi1
      kl_full_vs_pi1   : (3,) per-site KL(full || Pi1)
      kl_ablated_vs_pi1: (3,) per-site KL(ablated || Pi1)
      reduction_factor : (3,) ratio of full / ablated KL
    """
    occ_full = posterior_predictive_occupancy(particles_pi3, "pi3",
                                              n_subsample, seed, ablate_zeta=False)
    occ_ablated = posterior_predictive_occupancy(particles_pi3, "pi3",
                                                  n_subsample, seed, ablate_zeta=True)
    occ_pi1 = posterior_predictive_occupancy(particles_pi1, "pi1",
                                              n_subsample, seed)

    kl_full = per_site_kl(occ_full, occ_pi1)
    kl_abl = per_site_kl(occ_ablated, occ_pi1)
    return {
        "occ_full_mean": occ_full.mean(axis=0),
        "occ_full_std": occ_full.std(axis=0),
        "occ_ablated_mean": occ_ablated.mean(axis=0),
        "occ_ablated_std": occ_ablated.std(axis=0),
        "occ_pi1_mean": occ_pi1.mean(axis=0),
        "occ_pi1_std": occ_pi1.std(axis=0),
        "kl_full_vs_pi1": kl_full,
        "kl_ablated_vs_pi1": kl_abl,
        "reduction_factor": kl_full / np.maximum(kl_abl, 1e-10),
    }
