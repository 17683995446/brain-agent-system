"""Example 2: Custom Input Data

Feed custom input data into the simulation using different input adapters.

Usage:
    python examples/02_custom_input.py
"""

import numpy as np
from brain_agent import Simulation, Signal, create_input


def main():
    sim = Simulation.from_json()

    # Method 1: List of BrainData objects
    print("Method 1: List input")
    data = [Signal(value=float(i) * 0.01, timestamp=i * 0.005) for i in range(10)]
    inp = create_input(data)
    results = sim.run(steps=10, input_callback=inp)
    print(f"  Ran {len(results)} steps with list input")

    # Method 2: Callback function (procedural input)
    print("\nMethod 2: Callback input (sine wave)")
    sim.reset()
    inp = create_input(lambda step: Signal(
        value=float(np.sin(step * 0.1)),
        timestamp=step * 0.005,
    ))
    results = sim.run(steps=20, input_callback=inp)
    print(f"  Ran {len(results)} steps with callback input")

    # Method 3: Numpy arrays
    print("\nMethod 3: Array input")
    sim.reset()
    arrays = [np.random.randn(1, 8, 8).astype(np.float32) for _ in range(5)]
    inp = create_input(arrays, dt=0.005)
    results = sim.run(steps=5, input_callback=inp)
    print(f"  Ran {len(results)} steps with array input")

    print("\nDone!")


if __name__ == "__main__":
    main()
