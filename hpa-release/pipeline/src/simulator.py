import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents import spawn_population, apply_hazard, ROLES, N_ROLES, ETHNICITIES, N_ETHNICITIES, N_FAITHS, INSTITUTIONAL_ROLES
from src.decision_rules import utility_pi1, utility_pi2, utility_pi3, choose_site
from src.targets import compute_all_targets
from data.sites import N_SITES, SITES

def apply_post_resettlement_dynamics(pop, seed=42):
    rng = np.random.default_rng(seed + 17)
    n = pop['role'].shape[0]
    surv = pop['survived']
    init_faith = pop['faith']
    faith_traj = np.zeros((n, N_FAITHS, 3))
    faith_traj[:, :, 0] = init_faith
    bud_decline = -0.15 * pop['asset_loss']
    faith_traj[:, 0, 1] = np.clip(init_faith[:, 0] + bud_decline, 0.0, 1.0)
    is_han = pop['ethnicity'] == 0
    popo_rise = 0.1 * pop['asset_loss'] * is_han.astype(float) + 0.05 * ~is_han
    faith_traj[:, 1, 1] = np.clip(init_faith[:, 1] + popo_rise, 0.0, 1.0)
    faith_traj[:, 2, 1] = init_faith[:, 2] + rng.normal(0, 0.02, n)
    if 'site_choice' in pop:
        site_eth_clust = np.array([0.6, 0.5, 0.4])
        for p in range(N_SITES):
            mask = surv & (pop['site_choice'] == p)
            faith_traj[mask, 0, 2] = np.clip(faith_traj[mask, 0, 1] - 0.08 * (1 - site_eth_clust[p]) * pop['asset_loss'][mask], 0.0, 1.0)
            faith_traj[mask, 1, 2] = np.clip(faith_traj[mask, 1, 1] + 0.07 * site_eth_clust[p], 0.0, 1.0)
            faith_traj[mask, 2, 2] = faith_traj[mask, 2, 1] + rng.normal(0, 0.02, mask.sum())
    pop['faith_trajectory'] = faith_traj
    if 'site_choice' in pop:
        familiarity = np.zeros(n)
        for p in range(N_SITES):
            site_mask = surv & (pop['site_choice'] == p)
            if site_mask.sum() == 0:
                continue
            for v in range(17):
                vil_at_site = site_mask & (pop['village_id'] == v)
                vil_total_alive = surv & (pop['village_id'] == v)
                if vil_total_alive.sum() == 0:
                    continue
                frac_co_villagers = vil_at_site.sum() / vil_total_alive.sum()
                familiarity[vil_at_site] = frac_co_villagers
        pop['neighbour_familiarity'] = familiarity
    else:
        pop['neighbour_familiarity'] = np.full(n, 0.5)
    if 'site_choice' in pop:
        livelihood_table = np.array([[0.45, 0.3, 0.05], [0.5, 0.35, 0.1], [0.65, 0.6, 0.5], [0.7, 0.65, 0.6], [0.95, 0.95, 0.9], [0.95, 0.95, 0.9]])
        livelihood = livelihood_table[pop['role'], pop['site_choice']]
        livelihood = livelihood * (1.0 - 0.3 * pop['asset_loss'])
        livelihood = np.clip(livelihood + rng.normal(0, 0.05, n), 0.0, 1.0)
        pop['livelihood_score'] = livelihood
    else:
        pop['livelihood_score'] = np.full(n, 0.5)
    pre_extended = rng.random(n) < 0.6
    if 'site_choice' in pop:
        extended_retention = np.array([0.55, 0.45, 0.2])
        post_extended = pre_extended.copy()
        for p in range(N_SITES):
            site_mask = surv & (pop['site_choice'] == p) & pre_extended
            n_kept = int(extended_retention[p] * site_mask.sum())
            site_indices = np.where(site_mask)[0]
            if len(site_indices) > 0:
                rng.shuffle(site_indices)
                still_extended = site_indices[:n_kept]
                post_extended_mask = np.zeros(n, dtype=bool)
                post_extended_mask[still_extended] = True
                for idx in site_indices:
                    post_extended[idx] = idx in still_extended
        pop['pre_family_extended'] = pre_extended
        pop['post_family_extended'] = post_extended
    else:
        pop['pre_family_extended'] = pre_extended
        pop['post_family_extended'] = pre_extended
    deaths_with_children = ~surv & (pop['household_size'] >= 2)
    orphan_count = int(round(deaths_with_children.sum() * 1.5))
    orphan_kin_placement = 0.95 + rng.normal(0, 0.02)
    pop['orphan_count'] = orphan_count
    pop['orphan_kin_placement'] = float(np.clip(orphan_kin_placement, 0.5, 1.0))
    return pop

def forward_run(weights, family='pi3', seed=42, n_households=500):
    pop = spawn_population(n_households, seed=seed)
    pop = apply_hazard(pop, seed=seed)
    if family == 'pi1':
        utilities = utility_pi1(pop, weights)
    elif family == 'pi2':
        utilities = utility_pi2(pop, weights)
    elif family == 'pi3':
        utilities = utility_pi3(pop, weights)
    else:
        raise ValueError(f'unknown family {family}')
    site_choice = choose_site(utilities, seed=seed + 1)
    site_choice[~pop['survived']] = -1
    pop['site_choice'] = site_choice
    pop = apply_post_resettlement_dynamics(pop, seed=seed)
    return compute_all_targets(pop)
if __name__ == '__main__':
    from src.decision_rules import sample_prior
    w = sample_prior('pi3', 1, seed=0)[0]
    print(f'Pi3 weights: {w}')
    targets = forward_run(w, family='pi3', seed=42)
    for name, val in targets.items():
        print(f'  {name}: {val}')
