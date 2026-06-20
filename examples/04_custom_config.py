"""Example 4: Custom Configuration

Use a custom YAML configuration file to control simulation parameters.

Usage:
    python examples/04_custom_config.py
"""

import tempfile
import os
from brain_agent import Simulation, BrainConfig, SchedulerConfig


def main():
    # Method 1: Use BrainConfig to create SchedulerConfig
    print("Method 1: Programmatic configuration")
    config = BrainConfig({
        "simulation": {
            "dt_fast": 0.002,    # 2ms = 500Hz fast loop
            "dt_medium": 0.01,   # 10ms = 100Hz medium loop
            "dt_slow": 0.05,     # 50ms = 20Hz slow loop
            "max_latency_ms": 5.0,
        },
    })
    errors = config.validate()
    assert not errors, f"Config errors: {errors}"

    sched_config = config.to_scheduler_config()
    print(f"  dt_fast={sched_config.dt_fast}s, dt_medium={sched_config.dt_medium}s, "
          f"dt_slow={sched_config.dt_slow}s")

    sim = Simulation.from_json(config=sched_config)
    print(f"  Medium ratio: {sim.scheduler._medium_ratio} "
          f"(medium every {sim.scheduler._medium_ratio} fast steps)")
    print(f"  Slow ratio: {sim.scheduler._slow_ratio} "
          f"(slow every {sim.scheduler._slow_ratio} fast steps)")

    results = sim.run(steps=10)
    print(f"  Completed {len(results)} steps")

    # Method 2: Direct SchedulerConfig
    print("\nMethod 2: Direct SchedulerConfig")
    sim2 = Simulation.from_json(config=SchedulerConfig(
        dt_fast=0.001,
        dt_medium=0.005,
        dt_slow=0.05,
    ))
    print(f"  dt_fast={sim2.scheduler._config.dt_fast}s")
    results = sim2.run(steps=5)
    print(f"  Completed {len(results)} steps")

    # Method 3: YAML config file
    print("\nMethod 3: YAML config file")
    yaml_content = """
simulation:
  dt_fast: 0.003
  dt_medium: 0.015
  dt_slow: 0.075
  max_latency_ms: 8.0
monitoring:
  metrics_port: 9090
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        f.write(yaml_content)
        yaml_path = f.name

    config3 = BrainConfig.from_yaml(yaml_path)
    sched_config3 = config3.to_scheduler_config()
    sim3 = Simulation.from_json(config=sched_config3)
    print(f"  dt_fast={sched_config3.dt_fast}s, "
          f"dt_medium={sched_config3.dt_medium}s, "
          f"dt_slow={sched_config3.dt_slow}s")
    results = sim3.run(steps=5)
    print(f"  Completed {len(results)} steps")

    os.unlink(yaml_path)
    print("\nDone!")


if __name__ == "__main__":
    main()
