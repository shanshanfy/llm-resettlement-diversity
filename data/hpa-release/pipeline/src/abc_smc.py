"""
ABC-SMC over the simulator.

Sequential Monte Carlo Approximate Bayesian Computation (Sisson, Fan,
Tanaka 2007), with adaptive 50th-percentile tolerance schedule.

Because each forward run is fast (~1 ms), we run ABC directly on the
simulator without a GP emulator. The emulator is built separately in
emulator.py for reproducibility comparison.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulator import forward_run
from src.discrepancy import discrepancy
from src.decision_rules import (
    sample_prior, family_dim, WEIGHT_PRIOR_RANGES, PI_FAMILIES,
)


def _prior_log_density(weights, family):
    """Log density of weights under the (uniform) prior. -inf if out of range."""
    weight_names = PI_FAMILIES[family]
    for w, name in zip(weights, weight_names):
        lo, hi = WEIGHT_PRIOR_RANGES[name]
        if not (lo <= w <= hi):
            return -np.inf
    return 0.0  # uniform within range


def _perturb(particle, kernel_cov, family, rng):
    """Perturb a particle via a Gaussian kernel; reject samples outside prior."""
    weight_names = PI_FAMILIES[family]
    for _ in range(50):  # at most 50 attempts
        proposal = particle + rng.multivariate_normal(np.zeros(len(particle)), kernel_cov)
        if all(WEIGHT_PRIOR_RANGES[n][0] <= w <= WEIGHT_PRIOR_RANGES[n][1]
               for w, n in zip(proposal, weight_names)):
            return proposal
    return particle  # give up; return original


def run_abc_smc(family="pi3", n_particles=2000, n_generations=6, seed=42, verbose=True):
    """
    Run ABC-SMC for the given decision-rule family.

    Returns dict:
      particles      : (n_particles, dim) final-generation weight samples
      weights        : (n_particles,) particle weights
      tolerances     : list of tolerance values per generation
      n_evals        : total simulator evaluations
      gen_acceptance : list of acceptance rates per generation
    """
    rng = np.random.default_rng(seed)
    dim = family_dim(family)

    # Generation 0: sample from prior, run simulator, record discrepancies
    if verbose:
        print(f"[{family}] Generation 0: sampling {n_particles} from prior...")
    particles = sample_prior(family, n_particles, seed=seed)
    discrepancies = np.zeros(n_particles)
    for i, w in enumerate(particles):
        sim = forward_run(w, family=family, seed=seed + 1000)
        discrepancies[i] = discrepancy(sim)
    weights_p = np.full(n_particles, 1.0 / n_particles)
    tolerances = [np.inf]
    n_evals = n_particles
    gen_acceptance = [1.0]

    # Generation t = 1, 2, ..., n_generations
    for t in range(1, n_generations):
        # Adaptive tolerance: 50th percentile of current discrepancies
        eps = np.percentile(discrepancies, 50)
        tolerances.append(eps)

        # Kernel covariance: 2x the empirical particle variance
        cov = 2.0 * np.cov(particles, rowvar=False, aweights=weights_p) + 1e-6 * np.eye(dim)

        new_particles = np.zeros_like(particles)
        new_disc = np.zeros(n_particles)
        new_weights = np.zeros(n_particles)
        n_accepted = 0
        n_attempts = 0
        max_attempts = n_particles * 100  # avoid infinite loops

        # Pre-compute inv_cov once per generation
        inv_cov = np.linalg.inv(cov)

        while n_accepted < n_particles and n_attempts < max_attempts:
            # sample particle index proportional to weights
            idx = rng.choice(n_particles, p=weights_p / weights_p.sum())
            proposal = _perturb(particles[idx], cov, family, rng)
            sim = forward_run(proposal, family=family, seed=seed + 1000 + n_attempts)
            d = discrepancy(sim)
            n_attempts += 1
            n_evals += 1
            if d <= eps:
                new_particles[n_accepted] = proposal
                new_disc[n_accepted] = d
                # Kernel-density-based importance weight (Beaumont 2009).
                offsets = particles - proposal[None, :]
                quad = np.einsum('ij,jk,ik->i', offsets, inv_cov, offsets)
                kernel_density = np.exp(-0.5 * quad)
                denom = np.sum(weights_p * kernel_density)
                new_weights[n_accepted] = 1.0 / max(denom, 1e-100)
                n_accepted += 1

        # Normalise weights (handle degenerate case)
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        else:
            new_weights = np.full(n_particles, 1.0 / n_particles)
        particles = new_particles
        weights_p = new_weights
        discrepancies = new_disc
        gen_acceptance.append(n_accepted / max(n_attempts, 1))

        if verbose:
            print(f"[{family}] Gen {t}: tol={eps:.3f}, accept_rate={gen_acceptance[-1]:.3f}, n_evals={n_evals}")

    return {
        "particles": particles,
        "weights": weights_p,
        "tolerances": tolerances,
        "n_evals": n_evals,
        "gen_acceptance": gen_acceptance,
        "final_discrepancies": discrepancies,
    }


def posterior_summary(result, family):
    """Return per-weight posterior mean and std."""
    p = result["particles"]
    w = result["weights"]
    mean = (p * w[:, None]).sum(axis=0)
    var = (((p - mean[None, :]) ** 2) * w[:, None]).sum(axis=0)
    std = np.sqrt(var)
    weight_names = PI_FAMILIES[family]
    return {n: (m, s) for n, m, s in zip(weight_names, mean, std)}


if __name__ == "__main__":
    # Quick test with small n_particles
    result = run_abc_smc(family="pi3", n_particles=200, n_generations=3, seed=42)
    print("Posterior summary:")
    summary = posterior_summary(result, "pi3")
    for k, (m, s) in summary.items():
        print(f"  {k}: {m:.3f} +/- {s:.3f}")
