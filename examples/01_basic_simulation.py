"""Example 1: Basic Simulation

Run a brain simulation with just 5 lines of code.

Usage:
    python examples/01_basic_simulation.py
"""

from brain_agent import Simulation


def main():
    # Create simulation from the default brain tree JSON
    sim = Simulation.from_json()

    # Run 100 simulation steps
    results = sim.run(steps=100)

    # Print summary
    print(sim.summary())
    print(f"\nCompleted {len(results)} steps")


if __name__ == "__main__":
    main()
