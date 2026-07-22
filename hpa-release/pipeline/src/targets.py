import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.calibration_targets import OBSERVED_TARGETS
from data.sites import N_SITES
from src.agents import ROLES, ETHNICITIES, N_ROLES, N_ETHNICITIES, N_FAITHS

def t1_total_mortality(pop):
    deaths_in_simulation = (~pop['survived']).sum()
    return np.array([float(deaths_in_simulation * 40.0)])

def t2_village_mortality_dist(pop):
    deaths_per_village = np.zeros(17)
    for v in range(17):
        mask = (pop['village_id'] == v) & ~pop['survived']
        deaths_per_village[v] = mask.sum()
    if deaths_per_village.sum() > 0:
        deaths_per_village = deaths_per_village / deaths_per_village.sum()
    else:
        deaths_per_village = np.full(17, 1.0 / 17)
    return deaths_per_village

def t3_three_site_occupancy(pop):
    if 'site_choice' not in pop:
        return np.full(N_SITES, 1.0 / N_SITES)
    surv = pop['survived']
    if surv.sum() == 0:
        return np.full(N_SITES, 1.0 / N_SITES)
    choices = pop['site_choice'][surv]
    counts = np.bincount(choices, minlength=N_SITES).astype(float)
    return counts / counts.sum()

def t4_ethnic_mix_per_site(pop):
    out = np.zeros((N_SITES, N_ETHNICITIES))
    if 'site_choice' not in pop:
        out += 1.0 / N_ETHNICITIES
        return out
    surv = pop['survived']
    for p in range(N_SITES):
        for e in range(N_ETHNICITIES):
            mask = surv & (pop['site_choice'] == p) & (pop['ethnicity'] == e)
            site_count = (surv & (pop['site_choice'] == p)).sum()
            out[p, e] = mask.sum() / max(site_count, 1)
    return out

def t5_unfamiliar_neighbour_rate(pop):
    if 'neighbour_familiarity' not in pop:
        return np.array([0.5])
    surv = pop['survived']
    if surv.sum() == 0:
        return np.array([0.5])
    return np.array([1.0 - pop['neighbour_familiarity'][surv].mean()])

def t6_buddhism_trajectory(pop):
    if 'faith_trajectory' not in pop:
        return np.array([0.5, 0.4, 0.35])
    surv = pop['survived']
    return pop['faith_trajectory'][surv, 0, :].mean(axis=0)

def t7_popo_trajectory(pop):
    if 'faith_trajectory' not in pop:
        return np.array([0.4, 0.5, 0.55])
    surv = pop['survived']
    return pop['faith_trajectory'][surv, 1, :].mean(axis=0)

def t8_tibetan_buddhism_trajectory(pop):
    if 'faith_trajectory' not in pop:
        return np.array([0.85, 0.83, 0.84])
    surv = pop['survived']
    tib_mask = surv & (pop['ethnicity'] == 1)
    if tib_mask.sum() == 0:
        return np.array([0.85, 0.85, 0.85])
    return pop['faith_trajectory'][tib_mask, 2, :].mean(axis=0)

def t9_livelihood_recovery_by_role(pop):
    if 'livelihood_score' not in pop:
        return np.full(N_ROLES, 0.5)
    out = np.zeros(N_ROLES)
    surv = pop['survived']
    for r in range(N_ROLES):
        mask = surv & (pop['role'] == r)
        if mask.sum() > 0:
            out[r] = pop['livelihood_score'][mask].mean()
        else:
            out[r] = 0.5
    return out

def t10_extended_family_share(pop):
    if 'post_family_extended' not in pop:
        return np.array([0.6, 0.4])
    pre = pop['pre_family_extended'].mean()
    post = pop['post_family_extended'][pop['survived']].mean()
    return np.array([pre, post])

def t11_three_orphans(pop):
    if 'orphan_count' not in pop:
        return np.array([50.0, 0.9])
    return np.array([float(pop['orphan_count']), float(pop['orphan_kin_placement'])])
TARGET_FUNCTIONS = {'total_mortality': t1_total_mortality, 'village_mortality_dist': t2_village_mortality_dist, 'three_site_occupancy': t3_three_site_occupancy, 'ethnic_mix_per_site': t4_ethnic_mix_per_site, 'unfamiliar_neighbour_rate': t5_unfamiliar_neighbour_rate, 'buddhism_trajectory': t6_buddhism_trajectory, 'popo_trajectory': t7_popo_trajectory, 'tibetan_buddhism_trajectory': t8_tibetan_buddhism_trajectory, 'livelihood_recovery_by_role': t9_livelihood_recovery_by_role, 'extended_family_share': t10_extended_family_share, 'three_orphans': t11_three_orphans}

def compute_all_targets(pop):
    return {name: fn(pop) for name, fn in TARGET_FUNCTIONS.items()}

def flatten(target_dict):
    return np.concatenate([np.asarray(target_dict[k]).flatten() for k in OBSERVED_TARGETS.keys()])
