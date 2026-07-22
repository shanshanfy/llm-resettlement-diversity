from dataclasses import dataclass
import numpy as np

@dataclass
class Site:
    name: str
    distance_km: float
    housing: float
    services: float
    climate_match: float
    capacity_units: int
    job_risk_baseline: float

    @property
    def amenity(self):
        return 0.5 * (self.housing + self.services)
SITES = [Site(name='near', distance_km=1.0, housing=0.55, services=0.5, climate_match=1.0, capacity_units=1313, job_risk_baseline=0.45), Site(name='mid_valley', distance_km=13.0, housing=0.72, services=0.62, climate_match=0.95, capacity_units=2240, job_risk_baseline=0.55), Site(name='distant', distance_km=305.0, housing=0.85, services=0.9, climate_match=0.2, capacity_units=1150, job_risk_baseline=0.7)]
SITE_NAMES = [s.name for s in SITES]
N_SITES = len(SITES)

def site_features_array():
    return np.array([[s.housing, s.services, s.distance_km, s.climate_match, s.job_risk_baseline] for s in SITES])
if __name__ == '__main__':
    print(f'Number of sites: {N_SITES}')
    for s in SITES:
        print(f'  {s.name}: distance={s.distance_km}km, housing={s.housing}, services={s.services}, climate_match={s.climate_match}')
