"""Example 5: File-Based I/O

Read input data from files and write results to files.

Usage:
    python examples/05_file_io.py
"""

import json
import csv
import tempfile
import os
import numpy as np
from brain_agent import Simulation, Signal
from brain_agent.io import FileInput, create_input, create_exporter


def main():
    # Create input JSON file
    print("Creating input file...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json_path = f.name
        data = [
            {"type": "signal", "value": float(i) * 0.1, "timestamp": i * 0.005}
            for i in range(20)
        ]
        json.dump(data, f)
    print(f"  Input file: {json_path}")

    # Create simulation and run with file input
    sim = Simulation.from_json()
    inp = FileInput(json_path)
    print(f"  Input adapter loaded {inp.length} samples")

    results = sim.run(steps=20, input_callback=inp)
    print(f"  Completed {len(results)} steps")

    # Export results to JSON
    json_out = json_path.replace(".json", "_results.json")
    exporter = create_exporter(json_out)
    exporter.export_results(results)
    exporter.export_metrics(sim.metrics)
    exporter.close()
    print(f"\nResults exported to: {json_out}")

    # Export to CSV
    csv_out = json_path.replace(".json", "_results.csv")
    exporter = create_exporter(csv_out)
    exporter.export_results(results)
    exporter.close()
    print(f"CSV exported to: {csv_out}")

    # Export to NPZ
    npz_out = json_path.replace(".json", "_results.npz")
    exporter = create_exporter(npz_out)
    exporter.export_results(results)
    exporter.close()
    print(f"NPZ exported to: {npz_out}")

    # Verify CSV
    print("\nCSV preview (first 5 rows):")
    with open(csv_out) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 5:
                break
            print(f"  Step {row['step']}: "
                  f"nodes={row['nodes_processed']}, "
                  f"latency={row['total_latency_ms']}ms")

    # Cleanup
    for path in [json_path, json_out, csv_out, npz_out]:
        if os.path.exists(path):
            os.unlink(path)
    print("\nDone!")


if __name__ == "__main__":
    main()
