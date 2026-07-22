import numpy as np
ROLES = ['farmer', 'herder', 'laborer', 'infra', 'gov', 'rescuer']
ROLE_FRACTIONS = np.array([0.38, 0.2, 0.2, 0.1, 0.06, 0.06])
N_ROLES = len(ROLES)
ETHNICITIES = ['Han', 'Tibetan', 'Hui', 'Yi']
ETHNIC_FRACTIONS = np.array([0.95, 0.03, 0.01, 0.01])
N_ETHNICITIES = len(ETHNICITIES)
FAITH_BY_ETHNICITY = {'Han': np.array([0.55, 0.4, 0.05]), 'Tibetan': np.array([0.05, 0.1, 0.85]), 'Hui': np.array([0.1, 0.05, 0.05]), 'Yi': np.array([0.2, 0.15, 0.1])}
FAITH_NAMES = ['buddhism', 'popo', 'tibetan_buddhism']
N_FAITHS = len(FAITH_NAMES)
INSTITUTIONAL_ROLES = {'gov', 'rescuer', 'infra'}

def spawn_population(n_households=500, seed=42, village_centers=None):
    rng = np.random.default_rng(seed)
    role_counts = (ROLE_FRACTIONS * n_households).astype(int)
    role_counts[-1] = n_households - role_counts[:-1].sum()
    role = np.repeat(np.arange(N_ROLES), role_counts)
    rng.shuffle(role)
    ethnicity = rng.choice(N_ETHNICITIES, size=n_households, p=ETHNIC_FRACTIONS)
    inst_mask = np.isin(role, [ROLES.index(r) for r in INSTITUTIONAL_ROLES])
    vulnerable = (rng.random(n_households) < 0.15) & ~inst_mask
    household_size = np.clip(rng.poisson(4, n_households), 1, 12)
    if village_centers is None:
        rng_v = np.random.default_rng(seed + 1)
        in_zone = np.column_stack([rng_v.uniform(5, 16, size=7), rng_v.uniform(23, 33, size=7)])
        out_zone = np.column_stack([rng_v.uniform(20, 38, size=10), rng_v.uniform(2, 18, size=10)])
        village_centers = np.vstack([in_zone, out_zone])
    from data.calibration_targets import OBSERVED_TARGETS
    village_dist = OBSERVED_TARGETS['village_mortality_dist']
    village_assignment = rng.choice(17, size=n_households, p=village_dist)
    pre_disaster_xy = village_centers[village_assignment] + rng.normal(0, 1.5, (n_households, 2))
    faith = np.zeros((n_households, N_FAITHS))
    for i, eth_name in enumerate(ETHNICITIES):
        mask = ethnicity == i
        if mask.sum() > 0:
            base = FAITH_BY_ETHNICITY[eth_name]
            noise = rng.normal(0, 0.05, size=(mask.sum(), N_FAITHS))
            faith[mask] = np.clip(base[None, :] + noise, 0.0, 1.0)
    asset_loss = np.zeros(n_households)
    res_years = np.where(role < 2, rng.integers(5, 40, n_households), rng.integers(1, 25, n_households))
    return {'role': role, 'ethnicity': ethnicity, 'vulnerable': vulnerable, 'household_size': household_size, 'pre_disaster_xy': pre_disaster_xy, 'village_id': village_assignment, 'asset_loss': asset_loss, 'survived': np.ones(n_households, dtype=bool), 'faith': faith, 'pre_disaster_residence_years': res_years, '_village_centers': village_centers}

def apply_hazard(pop, hazard_intensity_at_xy=None, seed=42):
    rng = np.random.default_rng(seed + 7)
    n = pop['role'].shape[0]
    if hazard_intensity_at_xy is None:
        cx, cy = (10.0, 28.0)
        d = np.linalg.norm(pop['pre_disaster_xy'] - np.array([cx, cy]), axis=1)
        intensity = np.clip(1.0 - d / 8.0, 0.0, 1.0)
    else:
        intensity = np.array([hazard_intensity_at_xy(x, y) for x, y in pop['pre_disaster_xy']])
    p_lethal = 0.25 * np.power(intensity, 1.2)
    p_lethal *= np.where(pop['vulnerable'], 1.3, 1.0)
    p_lethal[np.isin(pop['role'], [ROLES.index(r) for r in INSTITUTIONAL_ROLES])] *= 0.2
    survives = rng.random(n) > p_lethal
    pop['survived'] &= survives
    loss_multiplier = np.array([0.95, 0.95, 0.85, 0.5, 0.2, 0.2])
    pop['asset_loss'] = np.clip(intensity * loss_multiplier[pop['role']], 0.0, 1.0)
    return pop
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    pop = spawn_population(n_households=500, seed=42)
    pop = apply_hazard(pop, seed=42)
    print(f"Total households: {pop['role'].shape[0]}")
    print(f"Survived: {pop['survived'].sum()}")
    print(f"Mean asset loss (survivors): {pop['asset_loss'][pop['survived']].mean():.3f}")
    print(f'Role distribution: ')
    for i, r in enumerate(ROLES):
        print(f"  {r}: {(pop['role'] == i).sum()}")
    print(f'Ethnicity distribution:')
    for i, e in enumerate(ETHNICITIES):
        print(f"  {e}: {(pop['ethnicity'] == i).sum()}")
