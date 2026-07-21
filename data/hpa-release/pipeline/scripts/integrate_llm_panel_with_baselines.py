"""
Panel-aware integration with three additional baselines:
  - random uniform baseline (1/3, 1/3, 1/3)
  - marginal-matched baseline (samples each household i.i.d. from the
    observed three-site occupancy (0.58, 0.22, 0.02))
  - multinomial-logit baseline fit on (role, ethnicity, asset_loss) ->
    site choice, with the target distribution being the observed marginal.

Same protocol as integrate_llm_panel.py: bootstrap each baseline B times,
re-aggregate to 11 targets, compute per-target and aggregate ELPD.
Output: outputs/analysis_panel_with_baselines.json

Usage:
    cd pipeline_v9
    python scripts/integrate_llm_panel_with_baselines.py \
        --llm outputs/llm_baseline_*.json \
        --out outputs/analysis_panel_with_baselines.json
"""

import sys
import json
import pickle
import argparse
import glob as _glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.agents import spawn_population, apply_hazard
from data.sites import N_SITES
from data.calibration_targets import OBSERVED_TARGETS
from src.simulator import apply_post_resettlement_dynamics
from src.targets import compute_all_targets
from src.discrepancy import per_target_log_likelihood
from src.elpd import psis_loo_elpd, compute_per_target_logliks


def _bootstrap_with_chooser(choose_fn, n_bootstrap, seed):
    """Generic bootstrap. choose_fn(i, rng, pop, asset_bin) -> int site."""
    rng = np.random.default_rng(seed)
    pop = spawn_population(500, seed=42)
    pop = apply_hazard(pop, seed=42)
    asset_bin = np.digitize(pop["asset_loss"], bins=[0.3, 0.6])

    target_names = list(OBSERVED_TARGETS.keys())
    n_targets = len(target_names)
    logliks = np.zeros((n_bootstrap, n_targets))

    for b in range(n_bootstrap):
        site_choice = -np.ones(pop["role"].shape[0], dtype=int)
        for i in np.where(pop["survived"])[0]:
            site_choice[i] = choose_fn(i, rng, pop, asset_bin)

        pop_b = {k: (v.copy() if hasattr(v, "copy") else v)
                 for k, v in pop.items()}
        pop_b["site_choice"] = site_choice
        pop_b = apply_post_resettlement_dynamics(pop_b, seed=42 + b)
        sim_targets = compute_all_targets(pop_b)
        ll = per_target_log_likelihood(sim_targets)
        for t, name in enumerate(target_names):
            logliks[b, t] = ll[name]

    return logliks


def bootstrap_random_uniform(n_bootstrap=100, seed=42):
    """Each household picks a site uniformly at random."""
    p = np.full(N_SITES, 1.0 / N_SITES)
    return _bootstrap_with_chooser(
        lambda i, rng, pop, ab: int(rng.choice(N_SITES, p=p)),
        n_bootstrap, seed)


def bootstrap_marginal_matched(n_bootstrap=100, seed=42):
    """Each household samples i.i.d. from the observed three-site
    marginal (0.58, 0.22, 0.02), with the missing 0.18 mass split
    proportionally so the distribution sums to 1."""
    obs = np.array(OBSERVED_TARGETS["three_site_occupancy"], dtype=float)
    p = obs / obs.sum()  # renormalise so sums to 1
    return _bootstrap_with_chooser(
        lambda i, rng, pop, ab: int(rng.choice(N_SITES, p=p)),
        n_bootstrap, seed)


def fit_multinomial_logit(n_iter=2000, seed=42):
    """Fit a small multinomial-logit on (role one-hot, ethnicity one-hot,
    asset_loss bin one-hot) -> 3-class site choice, training to match
    the observed marginal (0.58, 0.22, 0.02) on the surviving population.

    We fit by gradient descent on KL(observed_marginal || mean_predictions).
    Returns a callable that, given an agent's features, returns a (3,) probability vector.
    """
    rng = np.random.default_rng(seed)
    pop = spawn_population(500, seed=42)
    pop = apply_hazard(pop, seed=42)
    asset_bin = np.digitize(pop["asset_loss"], bins=[0.3, 0.6])
    surv = np.where(pop["survived"])[0]
    if len(surv) == 0:
        return lambda i, rng, pop, ab: 0

    # Build feature matrix (one-hot)
    n_roles = int(pop["role"].max()) + 1
    n_eth = int(pop["ethnicity"].max()) + 1
    n_ab = 3
    d_in = n_roles + n_eth + n_ab

    X = np.zeros((len(surv), d_in), dtype=float)
    for k, i in enumerate(surv):
        X[k, int(pop["role"][i])] = 1.0
        X[k, n_roles + int(pop["ethnicity"][i])] = 1.0
        X[k, n_roles + n_eth + int(asset_bin[i])] = 1.0

    # Parameters: (d_in, 3)
    W = rng.normal(0, 0.1, size=(d_in, 3))
    target_marg = np.array(OBSERVED_TARGETS["three_site_occupancy"], dtype=float)
    target_marg = target_marg / target_marg.sum()

    lr = 0.1
    for it in range(n_iter):
        logits = X @ W  # (n_surv, 3)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p = p / p.sum(axis=1, keepdims=True)
        mean_p = p.mean(axis=0)  # (3,)
        # Loss: KL(target || mean_p) (we want mean_p close to target)
        # gradient of KL(target||mean_p) wrt W: -(target - mean_p) ?
        # Simpler: minimize cross-entropy of target against mean_p
        # dL/dmean_p = -target/mean_p
        # dmean_p/dW = (1/n) sum_i p_i * (one_hot(c) - p_i) tensor X_i
        # but easier to push each agent's predicted distribution toward target_marg
        grad = np.zeros_like(W)
        diff = (mean_p - target_marg) / max(mean_p.shape[0], 1)
        for k in range(len(surv)):
            for c in range(3):
                grad[:, c] += diff[c] * p[k, c] * X[k]
        W -= lr * grad
        if it % 500 == 0 and False:  # disable verbose
            print(f"  fit iter {it}: mean_p = {mean_p}")
    return W, n_roles, n_eth


def bootstrap_logit(n_bootstrap=100, seed=42):
    """Fit logit, then bootstrap by sampling each household's choice
    from its predicted distribution."""
    W, n_roles, n_eth = fit_multinomial_logit(seed=seed)

    def choose(i, rng, pop, asset_bin):
        x = np.zeros(W.shape[0])
        x[int(pop["role"][i])] = 1.0
        x[n_roles + int(pop["ethnicity"][i])] = 1.0
        x[n_roles + n_eth + int(asset_bin[i])] = 1.0
        logits = x @ W
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        return int(rng.choice(N_SITES, p=p))

    return _bootstrap_with_chooser(choose, n_bootstrap, seed)


def bootstrap_llm_targets(decisions, n_bootstrap=100, seed=42):
    """Same as integrate_llm_panel.py — kept here for self-containment."""
    rng = np.random.default_rng(seed)
    pop = spawn_population(500, seed=42)
    pop = apply_hazard(pop, seed=42)
    asset_bin = np.digitize(pop["asset_loss"], bins=[0.3, 0.6])

    letter_to_site = {"A": 0, "B": 1, "C": 2}
    stratum_choices = {}
    for d in decisions:
        if d["parsed_choice"] is None:
            continue
        site = letter_to_site.get(d["parsed_choice"])
        if site is None:
            continue
        hh = d["hh_idx"]
        key = (int(pop["role"][hh]), int(pop["ethnicity"][hh]),
               int(asset_bin[hh]))
        stratum_choices.setdefault(key, []).append(site)

    global_choices = []
    for choices in stratum_choices.values():
        global_choices.extend(choices)
    global_dist = (np.bincount(global_choices, minlength=N_SITES) /
                   max(len(global_choices), 1))

    target_names = list(OBSERVED_TARGETS.keys())
    n_targets = len(target_names)
    logliks = np.zeros((n_bootstrap, n_targets))

    for b in range(n_bootstrap):
        site_choice = -np.ones(pop["role"].shape[0], dtype=int)
        for i in np.where(pop["survived"])[0]:
            key = (int(pop["role"][i]), int(pop["ethnicity"][i]),
                   int(asset_bin[i]))
            if key in stratum_choices and len(stratum_choices[key]) > 0:
                site_choice[i] = int(rng.choice(stratum_choices[key]))
            else:
                p = (global_dist if global_dist.sum() > 0
                     else np.full(N_SITES, 1 / N_SITES))
                site_choice[i] = int(rng.choice(N_SITES, p=p))
        pop_b = {k: (v.copy() if hasattr(v, "copy") else v)
                 for k, v in pop.items()}
        pop_b["site_choice"] = site_choice
        pop_b = apply_post_resettlement_dynamics(pop_b, seed=42 + b)
        sim_targets = compute_all_targets(pop_b)
        ll = per_target_log_likelihood(sim_targets)
        for t, name in enumerate(target_names):
            logliks[b, t] = ll[name]
    return logliks


def short_label(model_name):
    n = model_name.lower()
    if "gpt-4o-mini" in n:
        return "gpt4o-mini"
    if "claude-3-5-haiku" in n or "haiku" in n:
        return "haiku"
    if "claude-3-5-sonnet" in n or ("claude" in n and "sonnet" in n):
        return "sonnet-3.5"
    if "deepseek-v3" in n.replace("-", ""):
        return "DeepSeek-V3.1"
    if "deepseek-chat" in n:
        return "deepseek-chat"
    if "deepseek-reasoner" in n:
        return "deepseek-R"
    if "qwen" in n:
        return "qwen-flash"
    if "gemini" in n and "thinking" in n:
        return "gemini-2.5-T"
    if "gemini" in n:
        return "gemini"
    return model_name[:14]


def main(llm_paths, out_path, n_bootstrap=100):
    out_dir = PIPELINE_ROOT / "outputs"
    target_names = list(OBSERVED_TARGETS.keys())

    elpd, se, elpd_t = {}, {}, {}

    # ---- Naive baselines --------------------------------------------------
    print("Computing naive baselines...")
    print("  random-uniform  (1/3, 1/3, 1/3)")
    ll = bootstrap_random_uniform(n_bootstrap=n_bootstrap)
    e, s, et = psis_loo_elpd(ll)
    elpd["random"] = e; se["random"] = s; elpd_t["random"] = et
    print(f"  random-uniform    ELPD = {e:>8.2f} (SE {s:>5.2f})")

    print("  marginal-matched (0.58, 0.22, 0.02 renormalised)")
    ll = bootstrap_marginal_matched(n_bootstrap=n_bootstrap)
    e, s, et = psis_loo_elpd(ll)
    elpd["marginal"] = e; se["marginal"] = s; elpd_t["marginal"] = et
    print(f"  marginal-matched  ELPD = {e:>8.2f} (SE {s:>5.2f})")

    print("  logit (role+ethnicity+asset_bin)")
    ll = bootstrap_logit(n_bootstrap=n_bootstrap)
    e, s, et = psis_loo_elpd(ll)
    elpd["logit"] = e; se["logit"] = s; elpd_t["logit"] = et
    print(f"  logit             ELPD = {e:>8.2f} (SE {s:>5.2f})")

    # ---- Parametric families ---------------------------------------------
    posteriors = {}
    for fam in ["pi1", "pi2", "pi3"]:
        with open(out_dir / f"posterior_{fam}.pkl", "rb") as f:
            posteriors[fam] = pickle.load(f)
    print("\nComputing per-target ELPD for parametric families...")
    for fam in ["pi1", "pi2", "pi3"]:
        ll = compute_per_target_logliks(posteriors[fam]["particles"], fam,
                                        n_subsample=150, seed=42)
        e, s, et = psis_loo_elpd(ll)
        elpd[fam] = e; se[fam] = s; elpd_t[fam] = et
        print(f"  {fam:>14} ELPD = {e:>8.2f} (SE {s:>5.2f})")

    # ---- LLM panel -------------------------------------------------------
    panel = {}
    print(f"\nBootstrapping {len(llm_paths)} LLM baselines...")
    for path in llm_paths:
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        label = short_label(data["model"])
        if label in panel:
            label = label + "_" + path.stem
        ll = bootstrap_llm_targets(data["decisions"], n_bootstrap=n_bootstrap)
        e, s, et = psis_loo_elpd(ll)
        elpd[label] = e; se[label] = s; elpd_t[label] = et
        panel[label] = {
            "model": data["model"],
            "n_calls_made": data["n_calls_made"],
            "n_unparseable": data["n_unparseable"],
            "occupancy": data["aggregated_targets"]["three_site_occupancy"],
            "path": str(path),
        }
        print(f"  {label:>14} ELPD = {e:>8.2f} (SE {s:>5.2f})")

    # ---- Print summary ---------------------------------------------------
    families = (["random", "marginal", "logit", "pi1", "pi2", "pi3"]
                + list(panel.keys()))

    print(f"\n=== {len(families)}-way model comparison (sorted) ===")
    print(f"{'Family':<16} {'ELPD':>10} {'SE':>10}")
    for fam in sorted(families, key=lambda f: -elpd[f]):
        print(f"{fam:<16} {elpd[fam]:>10.2f} {se[fam]:>10.2f}")

    print("\n=== Per-target ELPD breakdown ===")
    header = f"{'target':<32}"
    for fam in families:
        header += f" {fam:>12}"
    print(header)
    for t, name in enumerate(target_names):
        row = f"{name:<32}"
        for fam in families:
            row += f" {elpd_t[fam][t]:>12.2f}"
        print(row)

    # ---- Save ------------------------------------------------------------
    results = {
        "elpd": {k: float(v) for k, v in elpd.items()},
        "se": {k: float(v) for k, v in se.items()},
        "elpd_t": {k: list(map(float, v)) for k, v in elpd_t.items()},
        "target_names": target_names,
        "panel": panel,
        "families": families,
        "observed_three_site_occupancy": list(OBSERVED_TARGETS["three_site_occupancy"]),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(families)}-way analysis (with baselines) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="append", required=True,
                        help="Path to llm_baseline_*.json. May repeat.")
    parser.add_argument("--out", default="outputs/analysis_panel_with_baselines.json")
    parser.add_argument("--n-bootstrap", type=int, default=100)
    args = parser.parse_args()
    main(args.llm, args.out, n_bootstrap=args.n_bootstrap)
