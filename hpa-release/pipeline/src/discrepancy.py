import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.calibration_targets import OBSERVED_TARGETS, TARGET_SCALES, OBSERVED_FLAT, SCALES_FLAT, TARGET_DIM, flatten_targets

def discrepancy(sim_targets):
    sim_flat = flatten_targets(sim_targets)
    diff = (sim_flat - OBSERVED_FLAT) / SCALES_FLAT
    return float(np.sqrt((diff ** 2).sum()))

def per_target_log_likelihood(sim_targets):
    out = {}
    for k, obs in OBSERVED_TARGETS.items():
        sim = np.asarray(sim_targets[k])
        scale = np.asarray(TARGET_SCALES[k])
        ll = -0.5 * np.mean(((sim - obs) / scale) ** 2) - 0.5 * np.mean(np.log(2 * np.pi * scale ** 2))
        out[k] = ll
    return out
