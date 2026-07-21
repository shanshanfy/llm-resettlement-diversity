"""
Eleven observed calibration targets for the 2010 debris-flow case
(site identity anonymised for review; disclosed at final submission).

Each target maps to a fixed-shape numpy array; the simulator's `targets.py`
module computes simulator-implied values of the same shape. The discrepancy
in `abc_smc.py` is the diagonal Mahalanobis distance between observed and
simulated, scaled by the values in `TARGET_SCALES`.

Sources:
- Recovery plan (county disaster-relief headquarters, published)
- Field record (source five-year ethnographic study; citation withheld
  for review)
- Public-health survey (county social-bureau and CDC reports)

Some values are point estimates with uncertainty captured by `TARGET_SCALES`,
which is the per-target observation noise used in the ABC-SMC discrepancy.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Observed values
# ---------------------------------------------------------------------------

OBSERVED_TARGETS = {
    # T1: total mortality count
    "total_mortality": np.array([1501.0]),

    # T2: village-level mortality distribution
    # 17 villages in the affected area; dominated by Village-01
    # with 200 households -> ~57 survivors. Other villages partial losses.
    # Normalised to a probability vector over 17 villages.
    # Villages 01 and 02 took the brunt (names withheld for review).
    "village_mortality_dist": (lambda v: v / v.sum())(np.array([
        0.55,  # Village-01
        0.18,  # Village-02
        0.07,  # Village-03
        0.05,  # Village-04
        0.04,  # Village-05
        0.03,  # Village-06
        0.02,  # Village-07
        0.015, 0.012, 0.010, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,
    ])),

    # T3: three-site occupancy at T+3 (near / mid-valley / distant)
    "three_site_occupancy": np.array([0.58, 0.22, 0.02]),

    # T4: ethnic-mix balance at each of three sites
    # Each site: fraction Han / Tibetan / Hui / Yi
    # Field record indicates ~95% Han across sites (Tibetan villages were
    # less affected by the debris flow).
    "ethnic_mix_per_site": np.array([
        [0.96, 0.03, 0.005, 0.005],  # near
        [0.94, 0.04, 0.01,  0.01],   # mid-valley
        [0.93, 0.05, 0.01,  0.01],   # distant
    ]),

    # T5: unfamiliar-neighbour rate at T+1
    # From field interviews: ~60% of resettled households reported not
    # knowing most of their new neighbours one year after relocation.
    "unfamiliar_neighbour_rate": np.array([0.60]),

    # T6: Buddhism faith trajectory
    # [strength_T0, strength_T+1, strength_T+3], 0..1 scale
    # Field record observed declining ("它骗人"); decline ~0.55 -> ~0.40 -> ~0.32
    "buddhism_trajectory": np.array([0.55, 0.40, 0.32]),

    # T7: folk-faith (popo) trajectory
    # Locally rising — villagers rebuilt popo temples post-disaster
    "popo_trajectory": np.array([0.40, 0.55, 0.62]),

    # T8: Tibetan Buddhism trajectory
    # Stable for Tibetan minority subgroup
    "tibetan_buddhism_trajectory": np.array([0.85, 0.83, 0.84]),

    # T9: livelihood-recovery rate by role at T+3
    # [farmer, herder, laborer, infra, gov, rescuer]
    # Staff (gov, rescuer) recover near-fully; farmers recover poorly.
    "livelihood_recovery_by_role": np.array([
        0.35,  # farmer (low — lost land)
        0.40,  # herder
        0.55,  # laborer (mobile)
        0.75,  # infra
        0.95,  # gov staff
        0.95,  # rescuer
    ]),

    # T10: extended-vs-nuclear family share
    # Pre-disaster ~0.6 extended; post-disaster ~0.3
    # Reported as [pre_share_extended, post_share_extended at T+3]
    "extended_family_share": np.array([0.60, 0.30]),

    # T11: three-orphans kinship outcomes
    # Total 53 children; 95% placed under kin guardianship.
    "three_orphans": np.array([53.0, 0.95]),  # [count, kin_placement_fraction]
}


# ---------------------------------------------------------------------------
# Per-target observation noise (used as the scaling for ABC discrepancy)
# ---------------------------------------------------------------------------

TARGET_SCALES = {
    "total_mortality": np.array([100.0]),  # +/- 100 deaths uncertainty
    "village_mortality_dist": np.full(17, 0.05),  # 5pp on each share
    "three_site_occupancy": np.array([0.05, 0.05, 0.02]),
    "ethnic_mix_per_site": np.full((3, 4), 0.03),
    "unfamiliar_neighbour_rate": np.array([0.10]),
    "buddhism_trajectory": np.array([0.08, 0.08, 0.08]),
    "popo_trajectory": np.array([0.10, 0.10, 0.10]),
    "tibetan_buddhism_trajectory": np.array([0.06, 0.06, 0.06]),
    "livelihood_recovery_by_role": np.full(6, 0.10),
    "extended_family_share": np.array([0.06, 0.06]),
    "three_orphans": np.array([10.0, 0.05]),
}


# ---------------------------------------------------------------------------
# Convenience: flatten to a single vector for the ABC discrepancy
# ---------------------------------------------------------------------------

def flatten_targets(target_dict):
    """Concatenate target arrays into a single 1-D numpy vector."""
    return np.concatenate([
        np.asarray(target_dict[k]).flatten()
        for k in OBSERVED_TARGETS.keys()
    ])


def get_target_dim():
    """Total dimension of the flattened target vector."""
    return sum(np.asarray(v).size for v in OBSERVED_TARGETS.values())


OBSERVED_FLAT = flatten_targets(OBSERVED_TARGETS)
SCALES_FLAT = flatten_targets(TARGET_SCALES)
TARGET_DIM = OBSERVED_FLAT.size


if __name__ == "__main__":
    print(f"Number of named targets: {len(OBSERVED_TARGETS)}")
    print(f"Total flattened target dim: {TARGET_DIM}")
    for k, v in OBSERVED_TARGETS.items():
        print(f"  {k}: shape={np.asarray(v).shape}, values={v}")
