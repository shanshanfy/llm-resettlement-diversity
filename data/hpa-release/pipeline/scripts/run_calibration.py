"""
Run ABC-SMC calibration for all three decision-rule families and save
posteriors. Designed to run in background; writes progress to a log file.

Usage:
    python scripts/run_calibration.py [n_particles] [n_generations]
"""

import sys, os, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.abc_smc import run_abc_smc, posterior_summary


def main(n_particles=500, n_generations=5, seed=42, output_dir="outputs"):
    out_dir = Path(__file__).resolve().parents[1] / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "calibration.log"
    log = open(log_path, "w", buffering=1)
    def lprint(msg):
        print(msg)
        log.write(msg + "\n")

    t0 = time.time()
    lprint(f"=== ABC-SMC calibration ===")
    lprint(f"n_particles={n_particles}, n_generations={n_generations}, seed={seed}")
    lprint(f"output_dir={out_dir}")
    lprint("")

    results = {}
    for fam in ['pi1', 'pi2', 'pi3']:
        lprint(f"--- {fam.upper()} ---")
        t1 = time.time()
        results[fam] = run_abc_smc(family=fam, n_particles=n_particles,
                                    n_generations=n_generations, seed=seed,
                                    verbose=False)
        elapsed = time.time() - t1
        lprint(f"  time: {elapsed:.1f}s, evals={results[fam]['n_evals']}")
        lprint(f"  tolerances: {[f'{t:.2f}' for t in results[fam]['tolerances']]}")
        lprint(f"  final accept rate: {results[fam]['gen_acceptance'][-1]:.3f}")
        lprint("  posterior summary:")
        for k, (m, s) in posterior_summary(results[fam], fam).items():
            lprint(f"    {k}: {m:.3f} +/- {s:.3f}")
        lprint("")

        # Save particles
        pickle_path = out_dir / f"posterior_{fam}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(results[fam], f)
        lprint(f"  saved {pickle_path}")
        lprint("")

    lprint(f"=== Total time: {time.time()-t0:.1f}s ===")
    log.close()
    return results


if __name__ == "__main__":
    n_particles = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_generations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(n_particles=n_particles, n_generations=n_generations)
