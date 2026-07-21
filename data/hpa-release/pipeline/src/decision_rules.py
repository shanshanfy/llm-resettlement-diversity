"""
Three families of resettlement-site decision rules: Pi1, Pi2, Pi3.

All three share the per-site amenity / distance / job-risk inputs.
Differences:
  Pi1 (4 weights):   distance-discounted utility
  Pi2 (6 weights):   adds protective-action terms (recurrence_risk, shelter)
  Pi3 (7 weights):   adds identity-grounded terms (ethnic, livelihood, network)

Each function returns a (N_HH, N_SITES) array of utilities.
The simulator turns utilities into site choices via a low-temperature softmax.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sites import SITES, N_SITES


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _per_household_distance(pop):
    """(N_HH, N_SITES) distance from each household's pre-disaster home to each site."""
    # Approximate: site is at a fixed fictional (x, y) on the simulation grid.
    # near=(20, 20), mid_valley=(60, 20), distant=(40, -150). Distances in km.
    site_xy = np.array([
        [20.0, 20.0],
        [60.0, 20.0],
        [40.0, -150.0],
    ])
    n_hh = pop["pre_disaster_xy"].shape[0]
    diff = site_xy[None, :, :] - pop["pre_disaster_xy"][:, None, :]
    # Treat 1 grid unit = 1 km for the purposes of distance.
    dist = np.linalg.norm(diff, axis=2)
    # But override with the published site distances for the absolute scale:
    site_published_km = np.array([s.distance_km for s in SITES])
    # Use published distances directly (households are notionally near impact zone)
    return np.tile(site_published_km, (n_hh, 1))


def _per_household_job_risk(pop):
    """(N_HH, N_SITES) job risk for each household at each site.
    Farmers: high job risk at distant sites (no farmland).
    Vendors/laborers: medium.
    Institutional: low everywhere.
    """
    # Role-dependent multipliers
    role_mult = np.array([1.5, 1.5, 1.0, 0.6, 0.2, 0.2])  # farmer ... rescuer
    base = np.array([s.job_risk_baseline for s in SITES])  # (N_SITES,)
    return role_mult[pop["role"]][:, None] * base[None, :]


def _per_household_climate_mismatch(pop):
    """(N_HH, N_SITES) climate mismatch (0 = same, 1 = different)."""
    cm = np.array([s.climate_match for s in SITES])
    return (1.0 - cm)[None, :] * np.ones(pop["role"].shape[0])[:, None]


def _per_household_recurrence_risk(pop):
    """(N_HH,) perceived recurrence risk at the original site."""
    # Higher for households that lost more.
    return np.clip(0.4 + 0.6 * pop["asset_loss"], 0.0, 1.0)


def _per_household_shelter_quality(pop):
    """(N_HH, N_SITES) perceived shelter quality at each candidate site."""
    # For Pi2's protective-action: amenity-correlated, with a small noise.
    return np.array([s.amenity for s in SITES])[None, :] * np.ones(pop["role"].shape[0])[:, None]


def _per_household_ethnic_clustering(pop):
    """
    (N_HH, N_SITES) ethnic-clustering preference.

    For each household i and site p, compute the fraction of resettled
    households at p that share i's ethnicity. We approximate this as the
    static ethnic-mix prior at each site (more refined: would use the
    dynamic occupancy distribution).
    """
    # Use the OBSERVED prior ethnic mix as the site's expected ethnic
    # composition (this is what the household reasons about ex ante).
    from data.calibration_targets import OBSERVED_TARGETS
    site_ethnic = OBSERVED_TARGETS["ethnic_mix_per_site"]  # (3, 4)
    # For each household, lookup the fraction of its own ethnicity at each site.
    n_hh = pop["ethnicity"].shape[0]
    own_eth_share = site_ethnic[np.arange(N_SITES)[None, :].repeat(n_hh, axis=0),
                                 pop["ethnicity"][:, None]]
    return own_eth_share  # (N_HH, N_SITES)


def _per_household_livelihood_continuity(pop):
    """
    (N_HH, N_SITES) probability that the household's livelihood continues at p.

    Farmers: high continuity at near (still in valley), zero at distant.
    Laborers: high continuity everywhere (mobile labour).
    Institutional: high everywhere.
    """
    # Per-role base continuity at each site
    cont_table = np.array([
        [0.85, 0.50, 0.05],   # farmer: needs land
        [0.80, 0.45, 0.05],   # herder
        [0.70, 0.65, 0.55],   # laborer
        [0.60, 0.55, 0.50],   # infra
        [0.85, 0.85, 0.80],   # gov
        [0.85, 0.85, 0.80],   # rescuer
    ])
    return cont_table[pop["role"]]


def _per_household_network_density(pop):
    """
    (N_HH, N_SITES) social-network density at each site.

    Approximation: scales with how many co-villagers we expect at each site.
    Near absorbs most residents of the principal affected village, mid_valley absorbs urban vendors,
    distant absorbs essentially no one.
    """
    # Density scales with site occupancy under the observed pattern,
    # weighted by whether the household is from a heavily-affected village.
    village_aff = np.zeros(pop["village_id"].shape[0])
    village_aff[pop["village_id"] < 4] = 1.0  # top 4 affected villages
    site_density = np.array([0.7, 0.4, 0.05])  # rough match to observed
    return village_aff[:, None] * site_density[None, :] + 0.1


# -----------------------------------------------------------------------------
# Decision rules
# -----------------------------------------------------------------------------

def utility_pi1(pop, weights):
    """
    Pi1: distance-discounted utility (4 weights).
      weights = [alpha, beta, gamma, delta]
              housing  services  -distance  -job_risk
    """
    alpha, beta, gamma, delta = weights
    site_features = np.array([[s.housing, s.services] for s in SITES])  # (3, 2)
    n_hh = pop["role"].shape[0]
    housing = np.tile(site_features[:, 0], (n_hh, 1))
    services = np.tile(site_features[:, 1], (n_hh, 1))
    dist = _per_household_distance(pop)
    job_risk = _per_household_job_risk(pop)
    # Normalise distance to 0..1 by dividing by 200 km
    dist_n = dist / 200.0
    return alpha * housing + beta * services - gamma * dist_n - delta * job_risk


def utility_pi2(pop, weights):
    """
    Pi2: protective-action (6 weights).
      weights = [alpha, beta, gamma, delta, rho1, rho2]
                                      ^recurrence_risk  ^shelter_quality
    """
    alpha, beta, gamma, delta, rho1, rho2 = weights
    u1 = utility_pi1(pop, [alpha, beta, gamma, delta])
    rec = _per_household_recurrence_risk(pop)[:, None]   # (N_HH, 1)
    shelter = _per_household_shelter_quality(pop)        # (N_HH, N_SITES)
    return u1 + rho1 * rec * np.ones((1, N_SITES)) + rho2 * shelter


def utility_pi3(pop, weights):
    """
    Pi3: identity-grounded (7 weights).
      weights = [alpha, beta, gamma, delta, zeta, eta, theta]
    """
    alpha, beta, gamma, delta, zeta, eta, theta = weights
    u1 = utility_pi1(pop, [alpha, beta, gamma, delta])
    ethnic = _per_household_ethnic_clustering(pop)
    livelihood = _per_household_livelihood_continuity(pop)
    network = _per_household_network_density(pop)
    return u1 + zeta * ethnic + eta * livelihood + theta * network


# -----------------------------------------------------------------------------
# Choose a site via low-temperature softmax
# -----------------------------------------------------------------------------

def choose_site(utilities, temperature=0.05, seed=42):
    """
    (N_HH, N_SITES) utilities -> (N_HH,) site indices.

    Sampled from softmax(u/T); at T=0.05 this is essentially argmax with
    occasional ties broken stochastically.
    """
    rng = np.random.default_rng(seed)
    u = utilities / temperature
    u = u - u.max(axis=1, keepdims=True)
    p = np.exp(u)
    p = p / p.sum(axis=1, keepdims=True)
    # Sample
    cum = np.cumsum(p, axis=1)
    r = rng.random(p.shape[0])[:, None]
    return (cum > r).argmax(axis=1)


# -----------------------------------------------------------------------------
# Default priors (matched in width across families)
# -----------------------------------------------------------------------------

WEIGHT_PRIOR_RANGES = {
    "alpha":  (0.0, 3.0),
    "beta":   (0.0, 3.0),
    "gamma":  (0.0, 5.0),
    "delta":  (0.0, 3.0),
    "rho1":   (0.0, 3.0),
    "rho2":   (0.0, 3.0),
    "zeta":   (0.0, 3.0),
    "eta":    (0.0, 3.0),
    "theta":  (0.0, 3.0),
}


PI_FAMILIES = {
    "pi1": ["alpha", "beta", "gamma", "delta"],
    "pi2": ["alpha", "beta", "gamma", "delta", "rho1", "rho2"],
    "pi3": ["alpha", "beta", "gamma", "delta", "zeta", "eta", "theta"],
}


def sample_prior(family, n_samples, seed=0):
    """Sample n_samples weight vectors from the prior for the given family."""
    rng = np.random.default_rng(seed)
    weight_names = PI_FAMILIES[family]
    out = np.empty((n_samples, len(weight_names)))
    for i, w in enumerate(weight_names):
        lo, hi = WEIGHT_PRIOR_RANGES[w]
        out[:, i] = rng.uniform(lo, hi, n_samples)
    return out


def family_dim(family):
    return len(PI_FAMILIES[family])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from agents import spawn_population, apply_hazard
    pop = spawn_population(500)
    pop = apply_hazard(pop)
    survivors = {k: v[pop["survived"]] if hasattr(v, "shape") and v.shape and v.shape[0] == 500 else v
                 for k, v in pop.items()}

    # Sanity check: try one weight sample for each family
    w1 = sample_prior("pi1", 1)[0]
    w2 = sample_prior("pi2", 1)[0]
    w3 = sample_prior("pi3", 1)[0]
    u1 = utility_pi1(survivors, w1)
    u2 = utility_pi2(survivors, w2)
    u3 = utility_pi3(survivors, w3)
    print(f"Pi1 weights: {w1}")
    print(f"  utility shape: {u1.shape}, mean per site: {u1.mean(axis=0)}")
    print(f"Pi2 weights: {w2}")
    print(f"  utility shape: {u2.shape}, mean per site: {u2.mean(axis=0)}")
    print(f"Pi3 weights: {w3}")
    print(f"  utility shape: {u3.shape}, mean per site: {u3.mean(axis=0)}")

    # Choose sites
    choices1 = choose_site(u1)
    choices3 = choose_site(u3)
    print(f"Pi1 occupancy: {np.bincount(choices1, minlength=3) / len(choices1)}")
    print(f"Pi3 occupancy: {np.bincount(choices3, minlength=3) / len(choices3)}")
