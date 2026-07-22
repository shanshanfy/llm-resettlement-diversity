import numpy as np
OBSERVED_TARGETS = {'total_mortality': np.array([1501.0]), 'village_mortality_dist': (lambda v: v / v.sum())(np.array([0.55, 0.18, 0.07, 0.05, 0.04, 0.03, 0.02, 0.015, 0.012, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001])), 'three_site_occupancy': np.array([0.58, 0.22, 0.02]), 'ethnic_mix_per_site': np.array([[0.96, 0.03, 0.005, 0.005], [0.94, 0.04, 0.01, 0.01], [0.93, 0.05, 0.01, 0.01]]), 'unfamiliar_neighbour_rate': np.array([0.6]), 'buddhism_trajectory': np.array([0.55, 0.4, 0.32]), 'popo_trajectory': np.array([0.4, 0.55, 0.62]), 'tibetan_buddhism_trajectory': np.array([0.85, 0.83, 0.84]), 'livelihood_recovery_by_role': np.array([0.35, 0.4, 0.55, 0.75, 0.95, 0.95]), 'extended_family_share': np.array([0.6, 0.3]), 'three_orphans': np.array([53.0, 0.95])}
TARGET_SCALES = {'total_mortality': np.array([100.0]), 'village_mortality_dist': np.full(17, 0.05), 'three_site_occupancy': np.array([0.05, 0.05, 0.02]), 'ethnic_mix_per_site': np.full((3, 4), 0.03), 'unfamiliar_neighbour_rate': np.array([0.1]), 'buddhism_trajectory': np.array([0.08, 0.08, 0.08]), 'popo_trajectory': np.array([0.1, 0.1, 0.1]), 'tibetan_buddhism_trajectory': np.array([0.06, 0.06, 0.06]), 'livelihood_recovery_by_role': np.full(6, 0.1), 'extended_family_share': np.array([0.06, 0.06]), 'three_orphans': np.array([10.0, 0.05])}

def flatten_targets(target_dict):
    return np.concatenate([np.asarray(target_dict[k]).flatten() for k in OBSERVED_TARGETS.keys()])

def get_target_dim():
    return sum((np.asarray(v).size for v in OBSERVED_TARGETS.values()))
OBSERVED_FLAT = flatten_targets(OBSERVED_TARGETS)
SCALES_FLAT = flatten_targets(TARGET_SCALES)
TARGET_DIM = OBSERVED_FLAT.size
if __name__ == '__main__':
    print(f'Number of named targets: {len(OBSERVED_TARGETS)}')
    print(f'Total flattened target dim: {TARGET_DIM}')
    for k, v in OBSERVED_TARGETS.items():
        print(f'  {k}: shape={np.asarray(v).shape}, values={v}')
