"""
Three resettlement-site features for the calibration case.
Site identities are anonymised for review (disclosed at final submission).

Sites:
  - near       : ~1 km, in-valley (adjacent to the county seat)
  - mid-valley : ~13 km, same valley
  - distant    : ~305 km, different climate basin (new-district area in the
    provincial-capital region). Distances are great-circle values computed
    from the actual geography; an earlier 185 km placeholder was corrected
    in v18.

Per-site feature vector (used by all decision rules):
  housing : 0..1, housing-quality score
  services: 0..1, supporting-services score
  amenity : 0..1, composite (housing + services / 2 in default; provided
                   here separately to give Pi1 access to both)
  climate_match : 0..1, similarity of climate to pre-disaster home
                  (1.0 = same valley, 0.0 = totally different basin)
  capacity_units : integer, planned housing units
  job_risk_baseline: 0..1, baseline job-loss risk for a new resident

`ethnic_concentration` is computed dynamically per household from the
fraction of co-ethnic resettlers at each site (see decision_rules.py).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Site:
    name: str
    distance_km: float          # from impact zone
    housing: float
    services: float
    climate_match: float
    capacity_units: int
    job_risk_baseline: float    # higher = riskier for incoming residents

    @property
    def amenity(self):
        return 0.5 * (self.housing + self.services)


SITES = [
    Site(
        name="near",
        distance_km=1.0,
        housing=0.55,
        services=0.50,
        climate_match=1.00,
        capacity_units=1313,
        job_risk_baseline=0.45,
    ),
    Site(
        name="mid_valley",
        distance_km=13.0,
        housing=0.72,
        services=0.62,
        climate_match=0.95,
        capacity_units=2240,
        job_risk_baseline=0.55,
    ),
    Site(
        name="distant",
        distance_km=305.0,
        housing=0.85,
        services=0.90,
        climate_match=0.20,
        capacity_units=1150,
        job_risk_baseline=0.70,
    ),
]


SITE_NAMES = [s.name for s in SITES]
N_SITES = len(SITES)


def site_features_array():
    """Return a (N_SITES, n_features) array of static site features."""
    return np.array([
        [s.housing, s.services, s.distance_km, s.climate_match,
         s.job_risk_baseline]
        for s in SITES
    ])


if __name__ == "__main__":
    print(f"Number of sites: {N_SITES}")
    for s in SITES:
        print(f"  {s.name}: distance={s.distance_km}km, "
              f"housing={s.housing}, services={s.services}, "
              f"climate_match={s.climate_match}")
