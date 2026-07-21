"""
Discrepancy function for ABC-SMC.

Diagonal Mahalanobis distance between observed and simulated targets,
with per-target observation-noise scaling.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.calibration_targets import (
    OBSERVED_TARGETS, TARGET_SCALES, OBSERVED_FLAT, SCALES_FLAT, TARGET_DIM,
    flatten_targets,
)


def discrepancy(sim_targets):
    """
    sim_targets: dict of simulated target arrays (same keys as OBSERVED_TARGETS).
    Returns: scalar diagonal-Mahalanobis distance.
    """
    sim_flat = flatten_targets(sim_targets)
    diff = (sim_flat - OBSERVED_FLAT) / SCALES_FLAT
    # Some targets are larger arrays (e.g., 17-dim village dist). Average within
    # each named target so each named target contributes ~equally to the total.
    return float(np.sqrt((diff ** 2).sum()))


def per_target_log_likelihood(sim_targets):
    """Per-target log-likelihood (Gaussian) for PSIS-LOO. Returns dict."""
    out = {}
    for k, obs in OBSERVED_TARGETS.items():
        sim = np.asarray(sim_targets[k])
        scale = np.asarray(TARGET_SCALES[k])
        # Average across-elements within target so each named target has 1 LL.
        ll = -0.5 * np.mean(((sim - obs) / scale) ** 2) - 0.5 * np.mean(np.log(2 * np.pi * scale ** 2))
        out[k] = ll
    return out
