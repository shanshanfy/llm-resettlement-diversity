"""
500-household agent population for the calibration case.

Each household is a flat numpy record (no Python objects per agent —
keeps forward runs fast for ABC-SMC). All household state is in a single
dict-of-arrays so we can do per-household decisions in vectorised form.

Demographic mix sourced from:
- Role: paper §3.4 / design doc (farmer 38%, herder 20%, laborer 20%,
  infra 10%, gov 6%, rescuer 6%)
- Ethnicity: 2005 county census; affected area was ~95% Han, ~5% other
- Vulnerable: ~15% of non-institutional households
- Initial faith: per-household by ethnicity
"""

import numpy as np


# Role assignments
ROLES = ["farmer", "herder", "laborer", "infra", "gov", "rescuer"]
ROLE_FRACTIONS = np.array([0.38, 0.20, 0.20, 0.10, 0.06, 0.06])
N_ROLES = len(ROLES)

# Ethnicities; the affected area was ~95% Han per the field record
ETHNICITIES = ["Han", "Tibetan", "Hui", "Yi"]
ETHNIC_FRACTIONS = np.array([0.95, 0.03, 0.01, 0.01])
N_ETHNICITIES = len(ETHNICITIES)

# Faith assignments by ethnicity (initial strength of each faith category)
# Categories: [Buddhism, popo, Tibetan_Buddhism]
FAITH_BY_ETHNICITY = {
    "Han":      np.array([0.55, 0.40, 0.05]),
    "Tibetan":  np.array([0.05, 0.10, 0.85]),
    "Hui":      np.array([0.10, 0.05, 0.05]),
    "Yi":       np.array([0.20, 0.15, 0.10]),
}
FAITH_NAMES = ["buddhism", "popo", "tibetan_buddhism"]
N_FAITHS = len(FAITH_NAMES)

# Institutional roles do not lose homes; they do not face starvation
INSTITUTIONAL_ROLES = {"gov", "rescuer", "infra"}


def spawn_population(n_households=500, seed=42, village_centers=None):
    """
    Generate the initial agent population for the calibration case.

    Returns a dict-of-arrays:
      role           : (N,) int        - index into ROLES
      ethnicity      : (N,) int        - index into ETHNICITIES
      vulnerable     : (N,) bool
      household_size : (N,) int
      pre_disaster_xy: (N, 2) float    - synthetic location in 40x40 grid
      village_id     : (N,) int        - index 0..16 for the 17-village mortality target
      asset_loss     : (N,) float in [0,1]
      survived       : (N,) bool       - mortality applied here
      faith          : (N, 3) float    - per-faith strength
      pre_disaster_residence_years: (N,) int
    """
    rng = np.random.default_rng(seed)

    # Roles (rounded to integer counts)
    role_counts = (ROLE_FRACTIONS * n_households).astype(int)
    role_counts[-1] = n_households - role_counts[:-1].sum()
    role = np.repeat(np.arange(N_ROLES), role_counts)
    rng.shuffle(role)

    # Ethnicity
    ethnicity = rng.choice(N_ETHNICITIES, size=n_households, p=ETHNIC_FRACTIONS)

    # Vulnerable
    inst_mask = np.isin(role, [ROLES.index(r) for r in INSTITUTIONAL_ROLES])
    vulnerable = (rng.random(n_households) < 0.15) & ~inst_mask

    # Household size (poisson-ish around 4)
    household_size = np.clip(rng.poisson(4, n_households), 1, 12)

    # Pre-disaster location: place the heavily-affected villages near the
    # impact cone (centered at (10, 28) with radius ~8). Less-affected
    # villages further away.
    if village_centers is None:
        # First 7 villages in the impact zone, others scattered outside
        rng_v = np.random.default_rng(seed + 1)
        in_zone = np.column_stack([
            rng_v.uniform(5, 16, size=7),
            rng_v.uniform(23, 33, size=7),
        ])
        out_zone = np.column_stack([
            rng_v.uniform(20, 38, size=10),
            rng_v.uniform(2, 18, size=10),
        ])
        village_centers = np.vstack([in_zone, out_zone])

    # Distribute households across villages with the village-mortality
    # distribution as the location prior (most agents live in the principal affected village).
    from data.calibration_targets import OBSERVED_TARGETS  # local import
    village_dist = OBSERVED_TARGETS["village_mortality_dist"]
    village_assignment = rng.choice(17, size=n_households, p=village_dist)
    pre_disaster_xy = village_centers[village_assignment] + rng.normal(0, 1.5, (n_households, 2))

    # Faith strengths by ethnicity
    faith = np.zeros((n_households, N_FAITHS))
    for i, eth_name in enumerate(ETHNICITIES):
        mask = (ethnicity == i)
        if mask.sum() > 0:
            base = FAITH_BY_ETHNICITY[eth_name]
            # add small per-household noise
            noise = rng.normal(0, 0.05, size=(mask.sum(), N_FAITHS))
            faith[mask] = np.clip(base[None, :] + noise, 0.0, 1.0)

    # Asset loss: depends on village (mortality-distribution-weighted) and
    # whether household is institutional. We compute it post-hazard, but
    # initialise to zero here.
    asset_loss = np.zeros(n_households)

    # Pre-disaster residence (years): older for farmers/herders
    res_years = np.where(role < 2,
                         rng.integers(5, 40, n_households),
                         rng.integers(1, 25, n_households))

    return {
        "role": role,
        "ethnicity": ethnicity,
        "vulnerable": vulnerable,
        "household_size": household_size,
        "pre_disaster_xy": pre_disaster_xy,
        "village_id": village_assignment,
        "asset_loss": asset_loss,
        "survived": np.ones(n_households, dtype=bool),
        "faith": faith,
        "pre_disaster_residence_years": res_years,
        "_village_centers": village_centers,
    }


def apply_hazard(pop, hazard_intensity_at_xy=None, seed=42):
    """
    Apply the hazard to compute mortality and asset loss.

    `hazard_intensity_at_xy(x, y)` should return a scalar in [0,1].
    If None, we use a default cone centered around the impact zone
    (matches the RAMMS-trained CA in social-agent/dfmas/env/hazard.py).

    Mutates the population dict in place.
    """
    rng = np.random.default_rng(seed + 7)
    n = pop["role"].shape[0]

    if hazard_intensity_at_xy is None:
        # Default: cone of damage centered at (10, 28), radius ~8
        cx, cy = 10.0, 28.0
        d = np.linalg.norm(pop["pre_disaster_xy"] - np.array([cx, cy]), axis=1)
        intensity = np.clip(1.0 - d / 8.0, 0.0, 1.0)
    else:
        intensity = np.array([
            hazard_intensity_at_xy(x, y) for x, y in pop["pre_disaster_xy"]
        ])

    # Mortality probability scales with intensity (and with vulnerability).
    # Calibrated so that ~7.5% of the 20k affected population dies (1501).
    # Here each household represents ~40 people; per-household death rate
    # should average ~7.5% in expectation across the population.
    p_lethal = 0.25 * np.power(intensity, 1.2)
    p_lethal *= np.where(pop["vulnerable"], 1.3, 1.0)
    p_lethal[np.isin(pop["role"], [ROLES.index(r) for r in INSTITUTIONAL_ROLES])] *= 0.2

    survives = rng.random(n) > p_lethal
    pop["survived"] &= survives

    # Asset loss for survivors scales with hazard intensity.
    # Farmers/herders lose more (land/livestock); gov/rescuers lose less.
    loss_multiplier = np.array([0.95, 0.95, 0.85, 0.50, 0.20, 0.20])
    pop["asset_loss"] = np.clip(intensity * loss_multiplier[pop["role"]], 0.0, 1.0)

    return pop


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    pop = spawn_population(n_households=500, seed=42)
    pop = apply_hazard(pop, seed=42)
    print(f"Total households: {pop['role'].shape[0]}")
    print(f"Survived: {pop['survived'].sum()}")
    print(f"Mean asset loss (survivors): {pop['asset_loss'][pop['survived']].mean():.3f}")
    print(f"Role distribution: ")
    for i, r in enumerate(ROLES):
        print(f"  {r}: {(pop['role']==i).sum()}")
    print(f"Ethnicity distribution:")
    for i, e in enumerate(ETHNICITIES):
        print(f"  {e}: {(pop['ethnicity']==i).sum()}")
