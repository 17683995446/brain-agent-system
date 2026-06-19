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
from typing import Optional, List

from .tree import BrainTreeBuilder
from .scheduler import FrequencyScheduler, SchedulerConfig, StepResult
from .monitoring import MetricsCollector, Tracer
from .logging_config import setup_logging
from .server import HealthServer
from .lifecycle import GracefulShutdown
from .config import BrainConfig


def _find_json_path() -> str:
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


def main() -> None:
    """Main entry point for the brain simulation system."""
    parser = argparse.ArgumentParser(description="大脑仿真Agent系统")
    parser.add_argument("--json", type=str, default=None, help="Path to brain tree JSON file")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps (0=infinite)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--output", type=str, default=None, help="Output metrics file")
    parser.add_argument("--json-logs", action="store_true", help="Use JSON log format")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")
    parser.add_argument("--port", type=int, default=9090, help="Health/metrics server port")
    parser.add_argument("--no-server", action="store_true", help="Disable HTTP health server")
    args = parser.parse_args()

    # Setup structured logging
    setup_logging(
        level=args.log_level,
        json_format=args.json_logs,
        log_file=args.log_file,
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("大脑仿真Agent系统 v0.1.0")
    logger.info("=" * 60)

    # Initialize graceful shutdown
    shutdown = GracefulShutdown()
    shutdown.install_signal_handlers()

    # Load and validate configuration
    try:
        brain_config = BrainConfig.from_defaults()
        config_errors = brain_config.validate()
        if config_errors:
            logger.error(f"Configuration validation failed: {config_errors}")
            sys.exit(1)
        logger.info("Configuration loaded and validated")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)

    # Determine JSON path: CLI arg > env var > config > search paths
    json_path = args.json
    if not json_path:
        env_path = brain_config.get("brain.json_path")
        if env_path and Path(env_path).exists():
            json_path = env_path
    if not json_path:
        try:
            json_path = _find_json_path()
        except FileNotFoundError as e:
            logger.error(f"JSON file not found: {e}")
            sys.exit(1)

    # Build tree
    try:
        logger.info(f"Loading brain tree from: {json_path}")
        start = time.perf_counter()
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(json_path)
        build_time = time.perf_counter() - start
        logger.info(
            f"Tree built in {build_time:.2f}s: "
            f"{tree.total_nodes} nodes, {tree.max_layer + 1} layers"
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to find JSON file: {e}")
        sys.exit(1)
    except MemoryError:
        logger.critical("Out of memory while building tree")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to build tree: {e}", exc_info=True)
        sys.exit(1)

    # Validate tree
    try:
        valid, errors = tree.validate()
        if not valid:
            logger.error(f"Tree validation failed: {errors}")
            sys.exit(1)
        logger.info("Tree validation passed")
    except Exception as e:
        logger.error(f"Tree validation error: {e}", exc_info=True)
        sys.exit(1)

    # Create scheduler
    config = SchedulerConfig(max_steps=args.steps if args.steps > 0 else 0)
    scheduler = FrequencyScheduler(tree, config)
    logger.info(f"Scheduler initialized: {scheduler.get_stats()}")

    # Create monitoring
    metrics = MetricsCollector(tree)
    Tracer()

    # Start health server (non-fatal if it fails)
    server: Optional[HealthServer] = None
    if not args.no_server:
        server_port = args.port or brain_config.get_int("monitoring.metrics_port", 9090)
        try:
            server = HealthServer(
                port=server_port,
                scheduler=scheduler,
                tree=tree,
                metrics=metrics,
            )
            server.start()
            server.set_ready(True)
            logger.info(f"Health server ready on port {server_port}")
        except OSError as e:
            logger.warning(
                f"Failed to start health server on port {server_port}: {e}. "
                f"Continuing without health server."
            )
            server = None
        except Exception as e:
            logger.warning(
                f"Unexpected error starting health server: {e}. "
                f"Continuing without health server."
            )
            server = None

    # Register cleanup callbacks (LIFO order)
    if server is not None:
        shutdown.register_callback(server.stop)
    shutdown.register_callback(lambda: logger.info("Shutdown complete"))

    # Run simulation
    steps_to_run = args.steps if args.steps > 0 else 10_000_000
    logger.info(f"Running up to {steps_to_run} simulation steps...")
    sim_start = time.perf_counter()

    results: List[StepResult] = []
    try:
        results = scheduler.run(
            steps=steps_to_run,
            shutdown_checker=lambda: shutdown.is_shutting_down,
        )
    except MemoryError:
        logger.critical("Out of memory during simulation - initiating graceful shutdown")
        shutdown.run_callbacks()
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received - initiating graceful shutdown")
        shutdown.run_callbacks()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        shutdown.run_callbacks()
        sys.exit(1)

    sim_time = time.perf_counter() - sim_start

    # Collect final metrics
    try:
        metrics.collect_node_metrics()
    except Exception as e:
        logger.warning(f"Failed to collect final metrics: {e}")

    # Report
    logger.info(f"Simulation completed in {sim_time:.2f}s")
    logger.info(f"Total steps: {len(results)}")
    if results:
        try:
            avg_latency = sum(r.total_latency_ms for r in results) / len(results)
            max_latency = max(r.max_node_latency_ms for r in results)
            total_errors = sum(len(r.errors) for r in results)
            logger.info(f"Avg step latency: {avg_latency:.2f}ms")
            logger.info(f"Max node latency: {max_latency:.2f}ms")
            logger.info(f"Total errors: {total_errors}")
        except Exception as e:
            logger.warning(f"Failed to compute summary statistics: {e}")

    # Export metrics
    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(metrics.export_prometheus())
            logger.info(f"Metrics exported to: {args.output}")
        except OSError as e:
            logger.error(f"Failed to write metrics file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing metrics: {e}", exc_info=True)

    # Graceful shutdown
    shutdown.run_callbacks()

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
