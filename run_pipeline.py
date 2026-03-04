"""
Run the full H2 research pipeline.

Usage:
    python run_pipeline.py              # Run everything
    python run_pipeline.py --from 3     # Resume from step 3 (build_networks)
    python run_pipeline.py --only 6 7   # Run only event_study and regression
"""

import argparse
import subprocess
import sys
import time


STEPS = {
    1: ("src.fetch_eth_data",   "Fetch ETH transaction data from BigQuery"),
    2: ("src.fetch_spy_data",   "Fetch SPY hourly price data"),
    3: ("src.build_networks",   "Build hourly networks & compute modularity"),
    4: ("src.compute_spikes",   "Compute SPY returns & identify volatility spikes"),
    5: ("src.merge_dataset",    "Merge ETH metrics + SPY spikes into panel"),
    6: ("src.event_study",      "Event study analysis"),
    7: ("src.regression",       "Predictive logistic regression"),
    8: ("src.robustness",       "Robustness checks"),
}


def run_step(step_num, module, description):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {description}")
    print(f"{'='*60}\n")
    start = time.time()
    result = subprocess.run([sys.executable, "-m", module], cwd=".")
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  Step {step_num} FAILED (exit code {result.returncode})")
        sys.exit(1)
    print(f"\n  Step {step_num} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run H2 research pipeline")
    parser.add_argument("--from", type=int, dest="from_step", default=1,
                        help="Start from this step number (default: 1)")
    parser.add_argument("--only", type=int, nargs="+",
                        help="Run only these step numbers")
    args = parser.parse_args()

    print("=" * 60)
    print("  Ethereum Network Modularity & Volatility Spikes")
    print("  Full Research Pipeline")
    print("=" * 60)

    if args.only:
        steps_to_run = args.only
    else:
        steps_to_run = [s for s in STEPS if s >= args.from_step]

    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"  Unknown step: {step_num}")
            continue
        module, description = STEPS[step_num]
        run_step(step_num, module, description)

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"  Figures: output/figures/")
    print(f"  Tables:  output/tables/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
