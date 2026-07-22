import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sites import SITES, N_SITES

def _per_household_distance(pop):
    site_xy = np.array([[20.0, 20.0], [60.0, 20.0], [40.0, -150.0]])
    n_hh = pop['pre_disaster_xy'].shape[0]
    diff = site_xy[None, :, :] - pop['pre_disaster_xy'][:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    site_published_km = np.array([s.distance_km for s in SITES])
    return np.tile(site_published_km, (n_hh, 1))

def _per_household_job_risk(pop):
    role_mult = np.array([1.5, 1.5, 1.0, 0.6, 0.2, 0.2])
    base = np.array([s.job_risk_baseline for s in SITES])
    return role_mult[pop['role']][:, None] * base[None, :]

def _per_household_climate_mismatch(pop):
    cm = np.array([s.climate_match for s in SITES])
    return (1.0 - cm)[None, :] * np.ones(pop['role'].shape[0])[:, None]

def _per_household_recurrence_risk(pop):
    return np.clip(0.4 + 0.6 * pop['asset_loss'], 0.0, 1.0)

def _per_household_shelter_quality(pop):
    return np.array([s.amenity for s in SITES])[None, :] * np.ones(pop['role'].shape[0])[:, None]

def _per_household_ethnic_clustering(pop):
    from data.calibration_targets import OBSERVED_TARGETS
    site_ethnic = OBSERVED_TARGETS['ethnic_mix_per_site']
    n_hh = pop['ethnicity'].shape[0]
    own_eth_share = site_ethnic[np.arange(N_SITES)[None, :].repeat(n_hh, axis=0), pop['ethnicity'][:, None]]
    return own_eth_share

def _per_household_livelihood_continuity(pop):
    cont_table = np.array([[0.85, 0.5, 0.05], [0.8, 0.45, 0.05], [0.7, 0.65, 0.55], [0.6, 0.55, 0.5], [0.85, 0.85, 0.8], [0.85, 0.85, 0.8]])
    return cont_table[pop['role']]

def _per_household_network_density(pop):
    village_aff = np.zeros(pop['village_id'].shape[0])
    village_aff[pop['village_id'] < 4] = 1.0
    site_density = np.array([0.7, 0.4, 0.05])
    return village_aff[:, None] * site_density[None, :] + 0.1

def utility_pi1(pop, weights):
    alpha, beta, gamma, delta = weights
    site_features = np.array([[s.housing, s.services] for s in SITES])
    n_hh = pop['role'].shape[0]
    housing = np.tile(site_features[:, 0], (n_hh, 1))
    services = np.tile(site_features[:, 1], (n_hh, 1))
    dist = _per_household_distance(pop)
    job_risk = _per_household_job_risk(pop)
    dist_n = dist / 200.0
    return alpha * housing + beta * services - gamma * dist_n - delta * job_risk

def utility_pi2(pop, weights):
    alpha, beta, gamma, delta, rho1, rho2 = weights
    u1 = utility_pi1(pop, [alpha, beta, gamma, delta])
    rec = _per_household_recurrence_risk(pop)[:, None]
    shelter = _per_household_shelter_quality(pop)
    return u1 + rho1 * rec * np.ones((1, N_SITES)) + rho2 * shelter

def utility_pi3(pop, weights):
    alpha, beta, gamma, delta, zeta, eta, theta = weights
    u1 = utility_pi1(pop, [alpha, beta, gamma, delta])
    ethnic = _per_household_ethnic_clustering(pop)
    livelihood = _per_household_livelihood_continuity(pop)
    network = _per_household_network_density(pop)
    return u1 + zeta * ethnic + eta * livelihood + theta * network

def choose_site(utilities, temperature=0.05, seed=42):
    rng = np.random.default_rng(seed)
    u = utilities / temperature
    u = u - u.max(axis=1, keepdims=True)
    p = np.exp(u)
    p = p / p.sum(axis=1, keepdims=True)
    cum = np.cumsum(p, axis=1)
    r = rng.random(p.shape[0])[:, None]
    return (cum > r).argmax(axis=1)
WEIGHT_PRIOR_RANGES = {'alpha': (0.0, 3.0), 'beta': (0.0, 3.0), 'gamma': (0.0, 5.0), 'delta': (0.0, 3.0), 'rho1': (0.0, 3.0), 'rho2': (0.0, 3.0), 'zeta': (0.0, 3.0), 'eta': (0.0, 3.0), 'theta': (0.0, 3.0)}
PI_FAMILIES = {'pi1': ['alpha', 'beta', 'gamma', 'delta'], 'pi2': ['alpha', 'beta', 'gamma', 'delta', 'rho1', 'rho2'], 'pi3': ['alpha', 'beta', 'gamma', 'delta', 'zeta', 'eta', 'theta']}

def sample_prior(family, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    weight_names = PI_FAMILIES[family]
    out = np.empty((n_samples, len(weight_names)))
    for i, w in enumerate(weight_names):
        lo, hi = WEIGHT_PRIOR_RANGES[w]
        out[:, i] = rng.uniform(lo, hi, n_samples)
    return out

def family_dim(family):
    return len(PI_FAMILIES[family])
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from agents import spawn_population, apply_hazard
    pop = spawn_population(500)
    pop = apply_hazard(pop)
    survivors = {k: v[pop['survived']] if hasattr(v, 'shape') and v.shape and (v.shape[0] == 500) else v for k, v in pop.items()}
    w1 = sample_prior('pi1', 1)[0]
    w2 = sample_prior('pi2', 1)[0]
    w3 = sample_prior('pi3', 1)[0]
    u1 = utility_pi1(survivors, w1)
    u2 = utility_pi2(survivors, w2)
    u3 = utility_pi3(survivors, w3)
    print(f'Pi1 weights: {w1}')
    print(f'  utility shape: {u1.shape}, mean per site: {u1.mean(axis=0)}')
    print(f'Pi2 weights: {w2}')
    print(f'  utility shape: {u2.shape}, mean per site: {u2.mean(axis=0)}')
    print(f'Pi3 weights: {w3}')
    print(f'  utility shape: {u3.shape}, mean per site: {u3.mean(axis=0)}')
    choices1 = choose_site(u1)
    choices3 = choose_site(u3)
    print(f'Pi1 occupancy: {np.bincount(choices1, minlength=3) / len(choices1)}')
    print(f'Pi3 occupancy: {np.bincount(choices3, minlength=3) / len(choices3)}')
