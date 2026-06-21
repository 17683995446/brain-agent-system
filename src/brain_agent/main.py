"""Main entry point for the brain simulation agent system.

Usage:
    brain-agent run --steps 100
    brain-agent run --steps 100 --export results.json
    brain-agent info
    brain-agent validate --json /path/to/brain_tree_v5.json
    brain-agent --version
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Any

from .tree import BrainTreeBuilder
from .scheduler import FrequencyScheduler, StepResult
from .monitoring import MetricsCollector
from .logging_config import setup_logging
from .server import HealthServer
from .lifecycle import GracefulShutdown
from .config import BrainConfig
from .paths import find_brain_tree_json
from .exceptions import ConfigValidationError

# Version: use importlib.metadata directly to avoid circular import
__version__ = "0.5.0"
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("brain-agent-system")
except Exception:
    pass


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by subcommands."""
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to brain tree JSON file (default: auto-discover)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: built-in defaults)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument("--json-logs", action="store_true", help="Use JSON log format")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path")


def _load_config(args: argparse.Namespace) -> BrainConfig:
    """Load and validate configuration."""
    if args.config:
        config = BrainConfig.from_yaml(args.config)
    else:
        config = BrainConfig.from_defaults()
    errors = config.validate()
    if errors:
        raise ConfigValidationError(f"Configuration validation failed: {errors}")
    return config


def _build_tree(args: argparse.Namespace, config: BrainConfig, logger: logging.Logger) -> Any:
    """Build the brain tree from JSON."""
    json_path = args.json
    if not json_path:
        env_path = config.get("brain.json_path")
        if env_path and Path(env_path).exists():
            json_path = env_path
    if not json_path:
        json_path = find_brain_tree_json()

    logger.info(f"Loading brain tree from: {json_path}")
    start = time.perf_counter()
    builder = BrainTreeBuilder()
    tree = builder.build_full_tree(json_path)
    build_time = time.perf_counter() - start
    logger.info(
        f"Tree built in {build_time:.2f}s: "
        f"{tree.total_nodes} nodes, {tree.max_layer + 1} layers"
    )
    return tree


def cmd_run(args: argparse.Namespace) -> int:
    """Run a brain simulation."""
    setup_logging(level=args.log_level, json_format=args.json_logs, log_file=args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Brain Agent System v%s - Run", __version__)
    logger.info("=" * 60)

    shutdown = GracefulShutdown()
    shutdown.install_signal_handlers()

    try:
        config = _load_config(args)
        logger.info("Configuration loaded and validated")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    try:
        tree = _build_tree(args, config, logger)
    except FileNotFoundError as e:
        logger.error(f"JSON file not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to build tree: {e}", exc_info=True)
        return 1

    try:
        valid, errors = tree.validate()
        if not valid:
            logger.error(f"Tree validation failed: {errors}")
            return 1
        logger.info("Tree validation passed")
    except Exception as e:
        logger.error(f"Tree validation error: {e}", exc_info=True)
        return 1

    scheduler_config = config.to_scheduler_config()
    if args.steps > 0:
        scheduler_config.max_steps = args.steps
    scheduler = FrequencyScheduler(tree, scheduler_config)
    logger.info(
        f"Scheduler: dt_fast={scheduler_config.dt_fast}s, "
        f"dt_medium={scheduler_config.dt_medium}s, "
        f"dt_slow={scheduler_config.dt_slow}s, "
        f"medium_ratio={scheduler.medium_ratio}, "
        f"slow_ratio={scheduler.slow_ratio}"
    )

    metrics = MetricsCollector(tree)

    server: Optional[HealthServer] = None
    if not args.no_server:
        server_port = args.port or config.get_int("monitoring.metrics_port", 9090)
        try:
            server = HealthServer(
                port=server_port, scheduler=scheduler, tree=tree, metrics=metrics,
            )
            server.start()
            server.set_ready(True)
            logger.info(f"Health server ready on port {server_port}")
        except Exception as e:
            logger.warning(f"Health server failed: {e}. Continuing without it.")
            server = None

    if server is not None:
        shutdown.register_callback(server.stop)
    shutdown.register_callback(lambda: logger.info("Shutdown complete"))

    steps_to_run = args.steps if args.steps > 0 else 10_000_000
    logger.info(f"Running {steps_to_run} simulation steps...")
    sim_start = time.perf_counter()

    results: List[StepResult] = []
    try:
        results = scheduler.run(
            steps=steps_to_run,
            shutdown_checker=lambda: shutdown.is_shutting_down,
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt - graceful shutdown")
        shutdown.run_callbacks()
        return 0
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        shutdown.run_callbacks()
        return 1

    sim_time = time.perf_counter() - sim_start

    try:
        metrics.collect_node_metrics()
    except Exception as e:
        logger.warning(f"Failed to collect final metrics: {e}")

    logger.info(f"Simulation completed in {sim_time:.2f}s")
    logger.info(f"Total steps: {len(results)}")
    if results:
        avg_latency = sum(r.total_latency_ms for r in results) / len(results)
        max_latency = max(r.max_node_latency_ms for r in results)
        total_errors = sum(len(r.errors) for r in results)
        logger.info(f"Avg step latency: {avg_latency:.2f}ms")
        logger.info(f"Max node latency: {max_latency:.2f}ms")
        logger.info(f"Total errors: {total_errors}")

    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(metrics.export_prometheus())
            logger.info(f"Metrics exported to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to write metrics: {e}")

    if args.export:
        try:
            from .io.output import create_exporter
            exporter = create_exporter(args.export)
            exporter.export_results(results)
            exporter.export_metrics(metrics)
            exporter.close()
            logger.info(f"Results exported to: {args.export}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}", exc_info=True)

    shutdown.run_callbacks()
    logger.info("Done!")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Display system information."""
    setup_logging(level="WARNING")
    logger = logging.getLogger(__name__)

    print(f"Brain Agent System v{__version__}")
    print(f"Python: {sys.version.split()[0]}")

    try:
        config = _load_config(args)
        print("\nConfiguration:")
        print(f"  dt_fast:   {config.get_float('simulation.dt_fast')}s")
        print(f"  dt_medium: {config.get_float('simulation.dt_medium')}s")
        print(f"  dt_slow:   {config.get_float('simulation.dt_slow')}s")
    except Exception:
        print("\nConfiguration: (using defaults)")

    try:
        tree = _build_tree(args, config, logger)
        print("\nBrain Tree:")
        print(f"  Total nodes: {tree.total_nodes}")
        print(f"  Layers:      {tree.max_layer + 1}")
        print(f"  Root node:   {tree.root.short_name if tree.root else 'N/A'}")

        # Count by region type
        from collections import Counter
        region_counts = Counter(n.region_type.value for n in tree.bfs_traverse())
        print("\n  Nodes by region:")
        for region, count in sorted(region_counts.items()):
            print(f"    {region}: {count}")
    except Exception as e:
        print(f"\nBrain Tree: Unable to load ({e})")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate the brain tree JSON file."""
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    try:
        config = _load_config(args)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return 1

    try:
        tree = _build_tree(args, config, logger)
    except Exception as e:
        logger.error(f"Failed to build tree: {e}")
        return 1

    valid, errors = tree.validate()
    if valid:
        print(f"VALID: {tree.total_nodes} nodes, {tree.max_layer + 1} layers")
        return 0
    else:
        print(f"INVALID: {len(errors)} errors found")
        for err in errors[:10]:
            print(f"  - {err}")
        return 1


def main() -> None:
    """Main entry point for the brain simulation system."""
    parser = argparse.ArgumentParser(
        prog="brain-agent",
        description="Brain simulation agent system with 2047 nodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  brain-agent run --steps 100                    Run 100 simulation steps
  brain-agent run --steps 100 --export out.json  Run and export results
  brain-agent info                                Show system information
  brain-agent validate                            Validate brain tree JSON
  brain-agent --version                           Show version
""",
    )
    parser.add_argument("--version", action="version", version=f"brain-agent {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run a brain simulation",
        description="Run a brain simulation for a specified number of steps.",
    )
    _add_common_args(run_parser)
    run_parser.add_argument("--steps", type=int, default=100, help="Number of steps (0=infinite, default: 100)")
    run_parser.add_argument("--output", type=str, default=None, help="Output Prometheus metrics file")
    run_parser.add_argument("--export", type=str, default=None, help="Export results to file (JSON/CSV/NPZ)")
    run_parser.add_argument("--port", type=int, default=9090, help="Health server port (default: 9090)")
    run_parser.add_argument("--no-server", action="store_true", help="Disable HTTP health server")
    run_parser.set_defaults(func=cmd_run)

    # info subcommand
    info_parser = subparsers.add_parser(
        "info", help="Display system information",
        description="Display system information including version, config, and tree stats.",
    )
    _add_common_args(info_parser)
    info_parser.set_defaults(func=cmd_info)

    # validate subcommand
    validate_parser = subparsers.add_parser(
        "validate", help="Validate brain tree JSON",
        description="Validate the brain tree JSON file and report any errors.",
    )
    _add_common_args(validate_parser)
    validate_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
