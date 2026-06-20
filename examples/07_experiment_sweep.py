"""Example 7: Experiment and Parameter Sweep

Run batch experiments with parameter sweeps to find optimal configurations.

Usage:
    python examples/07_experiment_sweep.py
"""

from brain_agent import Experiment, ExperimentConfig, ParameterSweep


def main():
    print("=" * 60)
    print("Experiment & Parameter Sweep Example")
    print("=" * 60)

    # --- Example 1: Single experiment ---
    print("\n1. Single Experiment")
    print("-" * 50)

    exp = Experiment(ExperimentConfig(
        name="baseline_run",
        steps=20,
        seed=42,
        dt_fast=0.005,
        dt_medium=0.02,
        dt_slow=0.1,
    ))
    results = exp.run()
    print(exp.summary())

    # --- Example 2: Parameter sweep ---
    print("\n2. Parameter Sweep: Finding optimal dt_fast")
    print("-" * 50)

    sweep = ParameterSweep(
        name="dt_fast_sweep",
        steps=10,
        seed=42,
    )
    sweep.add_param("dt_fast", [0.002, 0.005, 0.01])
    sweep.run()

    print(sweep.summary())

    # --- Example 3: Two-parameter sweep ---
    print("\n3. Two-Parameter Sweep: dt_fast x dt_medium")
    print("-" * 50)

    sweep2 = ParameterSweep(
        name="dt_grid",
        steps=5,
        seed=42,
    )
    sweep2.add_param("dt_fast", [0.005, 0.01])
    sweep2.add_param("dt_medium", [0.02, 0.04])
    results = sweep2.run()

    print(f"\n  Grid results ({len(results)} combinations):")
    for r in results:
        print(f"    dt_fast={r['params']['dt_fast']}, "
              f"dt_medium={r['params']['dt_medium']} -> "
              f"avg_latency={r['avg_latency_ms']:.1f}ms, "
              f"errors={r['total_errors']}")

    # Find best configuration
    best = sweep2.best(metric="avg_latency_ms", minimize=True)
    print(f"\n  Best: dt_fast={best['params']['dt_fast']}, "
          f"dt_medium={best['params']['dt_medium']} "
          f"({best['avg_latency_ms']:.1f}ms)")

    print("\n" + "=" * 60)
    print("Key takeaways:")
    print("  - Experiment: single run with reproducibility (seed)")
    print("  - ParameterSweep: grid search over config parameters")
    print("  - best(): find optimal configuration by metric")
    print("  - export(): save all results to JSON")
    print("=" * 60)


if __name__ == "__main__":
    main()
