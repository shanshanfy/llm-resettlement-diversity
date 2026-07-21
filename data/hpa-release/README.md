# Heterogeneity Preservation Audit (HPA) — core code and results

Anonymized release accompanying the manuscript *"Heterogeneity
Preservation Audit"* (under review). This is a partial release: it
contains the core simulation/calibration/scoring code and the derived
results behind the paper's numbers. Raw per-model LLM responses and the
panel-runner harness are withheld and available on reasonable request.

The testbed is a post-disaster resettlement simulation calibrated to
published outcome statistics from a 2010 debris-flow event (secondary
analysis of a published 2015 doctoral dissertation; the authors had no
role in the original data collection). All agents are simulated; no
personal data are included. Site and village identities are anonymized
for review.

## Layout

```
pipeline/
  src/                    Simulator, agent population, decision rules
                          (Pi1-Pi3), ABC-SMC calibration, ELPD scoring,
                          discrepancy and sensitivity utilities
  data/                   Site features and the 11 calibration targets
  scripts/
    run_calibration.py                   ABC-SMC calibration entry point
    integrate_llm_panel_with_baselines.py  ELPD scoring of the model
                                           panel against parametric
                                           baselines and oracle controls
  outputs/                Scored analysis JSONs (per-target ELPD for the
                          full panel: analysis_panel*.json)
  outputs_5000x6/         The paper's 5,000-particle x 6-generation
                          ABC-SMC run (seed 42): posteriors for Pi1-Pi3
                          and calibration.log (acceptance 6.9% / 10.6% /
                          37.7%)
results/                  Aggregated panel results: leaderboard.md,
                          panel_summary.csv, panel_long.csv, and the
                          frontier-model scores (analysis_frontier.json,
                          leaderboard_frontier.md)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing

Calibration (writes Pi1-Pi3 posteriors; the committed run used
`n_particles=5000, n_generations=6, seed=42`):

```bash
cd pipeline
python scripts/run_calibration.py
```

Scoring logic for every ELPD number in the paper is in
`pipeline/src/elpd.py` and
`pipeline/scripts/integrate_llm_panel_with_baselines.py`; its committed
output is `pipeline/outputs/analysis_panel_with_baselines.json`
(11 targets, 14 models plus random / marginal / logit baselines and
oracle controls). ELPD uses log-mean-exp over posterior draws; TVD
conventions are stated in the paper's Methods.

## License

Code: MIT (see LICENSE). Derived result files: CC BY 4.0. Attribution
withheld during anonymous review.
