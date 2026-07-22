import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.simulator import forward_run
from src.discrepancy import discrepancy
from src.decision_rules import sample_prior, family_dim, WEIGHT_PRIOR_RANGES, PI_FAMILIES

def _prior_log_density(weights, family):
    weight_names = PI_FAMILIES[family]
    for w, name in zip(weights, weight_names):
        lo, hi = WEIGHT_PRIOR_RANGES[name]
        if not lo <= w <= hi:
            return -np.inf
    return 0.0

def _perturb(particle, kernel_cov, family, rng):
    weight_names = PI_FAMILIES[family]
    for _ in range(50):
        proposal = particle + rng.multivariate_normal(np.zeros(len(particle)), kernel_cov)
        if all((WEIGHT_PRIOR_RANGES[n][0] <= w <= WEIGHT_PRIOR_RANGES[n][1] for w, n in zip(proposal, weight_names))):
            return proposal
    return particle

def run_abc_smc(family='pi3', n_particles=2000, n_generations=6, seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    dim = family_dim(family)
    if verbose:
        print(f'[{family}] Generation 0: sampling {n_particles} from prior...')
    particles = sample_prior(family, n_particles, seed=seed)
    discrepancies = np.zeros(n_particles)
    for i, w in enumerate(particles):
        sim = forward_run(w, family=family, seed=seed + 1000)
        discrepancies[i] = discrepancy(sim)
    weights_p = np.full(n_particles, 1.0 / n_particles)
    tolerances = [np.inf]
    n_evals = n_particles
    gen_acceptance = [1.0]
    for t in range(1, n_generations):
        eps = np.percentile(discrepancies, 50)
        tolerances.append(eps)
        cov = 2.0 * np.cov(particles, rowvar=False, aweights=weights_p) + 1e-06 * np.eye(dim)
        new_particles = np.zeros_like(particles)
        new_disc = np.zeros(n_particles)
        new_weights = np.zeros(n_particles)
        n_accepted = 0
        n_attempts = 0
        max_attempts = n_particles * 100
        inv_cov = np.linalg.inv(cov)
        while n_accepted < n_particles and n_attempts < max_attempts:
            idx = rng.choice(n_particles, p=weights_p / weights_p.sum())
            proposal = _perturb(particles[idx], cov, family, rng)
            sim = forward_run(proposal, family=family, seed=seed + 1000 + n_attempts)
            d = discrepancy(sim)
            n_attempts += 1
            n_evals += 1
            if d <= eps:
                new_particles[n_accepted] = proposal
                new_disc[n_accepted] = d
                offsets = particles - proposal[None, :]
                quad = np.einsum('ij,jk,ik->i', offsets, inv_cov, offsets)
                kernel_density = np.exp(-0.5 * quad)
                denom = np.sum(weights_p * kernel_density)
                new_weights[n_accepted] = 1.0 / max(denom, 1e-100)
                n_accepted += 1
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        else:
            new_weights = np.full(n_particles, 1.0 / n_particles)
        particles = new_particles
        weights_p = new_weights
        discrepancies = new_disc
        gen_acceptance.append(n_accepted / max(n_attempts, 1))
        if verbose:
            print(f'[{family}] Gen {t}: tol={eps:.3f}, accept_rate={gen_acceptance[-1]:.3f}, n_evals={n_evals}')
    return {'particles': particles, 'weights': weights_p, 'tolerances': tolerances, 'n_evals': n_evals, 'gen_acceptance': gen_acceptance, 'final_discrepancies': discrepancies}

def posterior_summary(result, family):
    p = result['particles']
    w = result['weights']
    mean = (p * w[:, None]).sum(axis=0)
    var = ((p - mean[None, :]) ** 2 * w[:, None]).sum(axis=0)
    std = np.sqrt(var)
    weight_names = PI_FAMILIES[family]
    return {n: (m, s) for n, m, s in zip(weight_names, mean, std)}
if __name__ == '__main__':
    result = run_abc_smc(family='pi3', n_particles=200, n_generations=3, seed=42)
    print('Posterior summary:')
    summary = posterior_summary(result, 'pi3')
    for k, (m, s) in summary.items():
        print(f'  {k}: {m:.3f} +/- {s:.3f}')
