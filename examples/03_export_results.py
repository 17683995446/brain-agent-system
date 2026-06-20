"""Example 3: Result Export and Analysis

Export simulation results to files and analyze them.

Usage:
    python examples/03_export_results.py
"""

import json
import tempfile
import os
from brain_agent import Simulation


def main():
    sim = Simulation.from_json()

    # Run simulation
    results = sim.run(steps=50)
    print(f"Completed {len(results)} steps")

    # Export to JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json_path = f.name
    sim.export(json_path)
    print(f"Results exported to: {json_path}")

    # Read back and analyze
    with open(json_path) as f:
        data = json.load(f)

    print(f"\nAnalysis of {len(data)} steps:")
    avg_nodes = sum(d["nodes_processed"] for d in data) / len(data)
    avg_latency = sum(d["total_latency_ms"] for d in data) / len(data)
    total_errors = sum(len(d.get("errors", [])) for d in data)

    print(f"  Avg nodes processed per step: {avg_nodes:.0f}")
    print(f"  Avg step latency: {avg_latency:.2f}ms")
    print(f"  Total errors: {total_errors}")

    # Export to CSV
    csv_path = json_path.replace(".json", ".csv")
    sim.export(csv_path)
    print(f"\nCSV exported to: {csv_path}")

    # Export node states
    exporter = sim.export(include_node_states=True)
    states = exporter.export_node_states(sim.tree)
    print(f"\nNode states: {len(states)} nodes")
    # Show a few active nodes
    active = [(nid, s) for nid, s in states.items() if s["process_count"] > 0]
    active.sort(key=lambda x: x[1]["process_count"], reverse=True)
    print("Top 5 most active nodes:")
    for nid, s in active[:5]:
        print(f"  Node {nid} ({s['short_name']}): "
              f"processed {s['process_count']}x, "
              f"avg {s['avg_process_time']:.6f}s")

    # Cleanup
    os.unlink(json_path)
    os.unlink(csv_path)


if __name__ == "__main__":
    main()
