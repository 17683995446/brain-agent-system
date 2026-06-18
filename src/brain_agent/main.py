"""Main entry point for the brain simulation agent system.

Usage:
    python -m brain_agent.main --steps 100 --json /path/to/brain_tree_v5.json
    python -m brain_agent.main --help
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from .tree import BrainTreeBuilder
from .scheduler import FrequencyScheduler, SchedulerConfig
from .monitoring import MetricsCollector, Tracer


def _find_json_path():
    """Find brain_tree_v5.json in standard locations."""
    search_paths = [
        Path(__file__).parent.parent / "data" / "brain_tree_v5.json",
        Path("/workspace/BrainAgentSystem/data/brain_tree_v5.json"),
        Path("/data/user/work/brain_tree_v5.json"),
    ]
    for p in search_paths:
        if p.exists():
            return str(p)
    raise FileNotFoundError("brain_tree_v5.json not found in any search path")


def main():
    parser = argparse.ArgumentParser(description="大脑仿真Agent系统")
    parser.add_argument("--json", type=str, default=None, help="Path to brain tree JSON file")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--output", type=str, default=None, help="Output metrics file")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("大脑仿真Agent系统 v0.1.0")
    logger.info("=" * 60)

    # Build tree
    json_path = args.json or _find_json_path()
    logger.info(f"Loading brain tree from: {json_path}")

    start = time.perf_counter()
    builder = BrainTreeBuilder()
    tree = builder.build_full_tree(json_path)
    build_time = time.perf_counter() - start
    logger.info(f"Tree built in {build_time:.2f}s: {tree.total_nodes} nodes, {tree.max_layer + 1} layers")

    # Validate
    valid, errors = tree.validate()
    if not valid:
        logger.error(f"Tree validation failed: {errors}")
        sys.exit(1)
    logger.info("Tree validation passed")

    # Create scheduler
    config = SchedulerConfig()
    scheduler = FrequencyScheduler(tree, config)
    logger.info(f"Scheduler initialized: {scheduler.get_stats()}")

    # Create monitoring
    metrics = MetricsCollector(tree)
    Tracer()

    # Run simulation
    logger.info(f"Running {args.steps} simulation steps...")
    sim_start = time.perf_counter()
    results = scheduler.run(steps=args.steps)
    sim_time = time.perf_counter() - sim_start

    # Collect final metrics
    metrics.collect_node_metrics()

    # Report
    logger.info(f"Simulation completed in {sim_time:.2f}s")
    logger.info(f"Total steps: {len(results)}")
    if results:
        avg_latency = sum(r.total_latency_ms for r in results) / len(results)
        max_latency = max(r.max_node_latency_ms for r in results)
        total_errors = sum(len(r.errors) for r in results)
        logger.info(f"Avg step latency: {avg_latency:.2f}ms")
        logger.info(f"Max node latency: {max_latency:.2f}ms")
        logger.info(f"Total errors: {total_errors}")

    # Export metrics
    if args.output:
        with open(args.output, 'w') as f:
            f.write(metrics.export_prometheus())
        logger.info(f"Metrics exported to: {args.output}")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
