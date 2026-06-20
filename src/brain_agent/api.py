"""High-level API for brain simulation.

Provides a simplified interface for creating and running brain simulations
without needing to understand the internal architecture.

Usage:
    from brain_agent import Simulation

    # Create simulation from JSON
    sim = Simulation.from_json("data/brain_tree_v5.json")

    # Run 100 steps
    results = sim.run(steps=100)

    # Get statistics
    stats = sim.stats
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

from .tree import BrainTree, BrainTreeBuilder
from .node import BrainNode
from .scheduler import FrequencyScheduler, SchedulerConfig, StepResult
from .monitoring import MetricsCollector, Tracer
from .types import BrainData

logger = logging.getLogger(__name__)


def _find_json_path() -> str:
    """Find brain_tree_v5.json in standard locations."""
    search_paths = [
        Path(__file__).parent.parent / "data" / "brain_tree_v5.json",
        Path("/workspace/BrainAgentSystem/data/brain_tree_v5.json"),
    ]
    for p in search_paths:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "brain_tree_v5.json not found. Specify json_path explicitly."
    )


def create_simulation(
    json_path: Optional[str] = None,
    config: Optional[SchedulerConfig] = None,
) -> "Simulation":
    """Create a Simulation instance from a brain tree JSON file.

    Args:
        json_path: Path to brain tree JSON. If None, searches standard locations.
        config: Optional scheduler configuration. Uses defaults if None.

    Returns:
        A Simulation instance ready to run.

    Example:
        from brain_agent import create_simulation
        sim = create_simulation()
        results = sim.run(steps=100)
    """
    return Simulation.from_json(json_path, config)


def run_simulation(
    json_path: Optional[str] = None,
    steps: int = 100,
    input_data: Optional[BrainData] = None,
    config: Optional[SchedulerConfig] = None,
) -> List[StepResult]:
    """One-shot simulation run.

    Creates a simulation, runs it for the given steps, and returns results.

    Args:
        json_path: Path to brain tree JSON. If None, searches standard locations.
        steps: Number of simulation steps to run.
        input_data: Optional input data fed to the root node each step.
        config: Optional scheduler configuration.

    Returns:
        List of StepResult for each completed step.

    Example:
        from brain_agent import run_simulation
        results = run_simulation(steps=50)
    """
    sim = create_simulation(json_path, config)
    return sim.run(steps=steps, input_data=input_data)


class Simulation:
    """High-level brain simulation interface.

    Wraps BrainTree, FrequencyScheduler, and MetricsCollector into a
    single object with a simple API.

    Attributes:
        tree: The brain node tree.
        scheduler: The frequency scheduler.
        metrics: The metrics collector.
        tracer: The execution tracer.
    """

    def __init__(
        self,
        tree: BrainTree,
        scheduler: Optional[FrequencyScheduler] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        self.tree = tree
        self.scheduler = scheduler or FrequencyScheduler(tree)
        self.metrics = metrics or MetricsCollector(tree)
        self.tracer = Tracer()
        self._results: List[StepResult] = []

    @classmethod
    def from_json(
        cls,
        json_path: Optional[str] = None,
        config: Optional[SchedulerConfig] = None,
    ) -> "Simulation":
        """Create a Simulation from a brain tree JSON file.

        Args:
            json_path: Path to brain tree JSON. If None, searches standard locations.
            config: Optional scheduler configuration.

        Returns:
            A Simulation instance.
        """
        if json_path is None:
            json_path = _find_json_path()

        logger.info(f"Loading brain tree from: {json_path}")
        start = time.perf_counter()
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(json_path)
        build_time = time.perf_counter() - start
        logger.info(
            f"Tree built in {build_time:.2f}s: "
            f"{tree.total_nodes} nodes, {tree.max_layer + 1} layers"
        )

        scheduler = FrequencyScheduler(tree, config) if config else FrequencyScheduler(tree)
        metrics = MetricsCollector(tree)

        return cls(tree, scheduler, metrics)

    def run(
        self,
        steps: int = 100,
        input_data: Optional[BrainData] = None,
        input_callback: Optional[Callable[[int], BrainData]] = None,
        shutdown_checker: Optional[Callable[[], bool]] = None,
    ) -> List[StepResult]:
        """Run the simulation for a number of steps.

        Args:
            steps: Number of simulation steps to run.
            input_data: Static input data fed to root node each step.
            input_callback: Function called each step with step number,
                returns input data for that step. Overrides input_data.
            shutdown_checker: Optional callable returning True to stop gracefully.

        Returns:
            List of StepResult for each completed step.

        Example:
            # Static input
            results = sim.run(steps=100, input_data=Signal(value=1.0))

            # Dynamic input per step
            def my_input(step):
                return Signal(value=float(step) * 0.01)

            results = sim.run(steps=100, input_callback=my_input)
        """
        if input_callback is not None:
            results: List[StepResult] = []
            for i in range(steps):
                if shutdown_checker is not None and shutdown_checker():
                    break
                data = input_callback(i)
                result = self.scheduler.step(input_data=data)
                self.metrics.collect_step_result(result)
                results.append(result)
            self._results.extend(results)
            return results

        results = self.scheduler.run(
            steps=steps,
            input_data=input_data,
            shutdown_checker=shutdown_checker,
        )
        for result in results:
            self.metrics.collect_step_result(result)
        self._results.extend(results)
        return results

    def step(self, input_data: Optional[BrainData] = None) -> StepResult:
        """Execute a single simulation step.

        Args:
            input_data: Optional input data for the root node.

        Returns:
            StepResult for this step.
        """
        result = self.scheduler.step(input_data=input_data)
        self.metrics.collect_step_result(result)
        self._results.append(result)
        return result

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self.scheduler.reset()
        self.metrics.clear()
        self.tracer.clear()
        self._results.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get current simulation statistics."""
        return self.scheduler.get_stats()

    @property
    def results(self) -> List[StepResult]:
        """Get all step results from this simulation."""
        return self._results

    @property
    def step_count(self) -> int:
        """Get the current step number."""
        return self.scheduler._step

    @property
    def sim_time(self) -> float:
        """Get the current simulation time in seconds."""
        return self.scheduler._sim_time

    def export_metrics(self, format: str = "prometheus") -> str:
        """Export collected metrics.

        Args:
            format: Output format. Currently supports "prometheus".

        Returns:
            Metrics string in the requested format.
        """
        self.metrics.collect_node_metrics()
        if format == "prometheus":
            return self.metrics.export_prometheus()
        raise ValueError(f"Unsupported metrics format: {format}")

    def get_node(self, node_id: int) -> Optional[BrainNode]:
        """Get a node by ID.

        Args:
            node_id: The node ID to look up.

        Returns:
            The BrainNode with the given ID.
        """
        return self.tree.get_node(node_id)

    def summary(self) -> str:
        """Get a human-readable summary of the simulation.

        Returns:
            A formatted string with simulation statistics.
        """
        stats = self.stats
        lines = [
            "Brain Simulation Summary",
            "=" * 40,
            f"  Total nodes:    {stats['total_nodes']}",
            f"  Steps run:      {stats['step']}",
            f"  Sim time:       {stats['sim_time']:.3f}s",
            f"  Fast band:      {stats['band_counts']['fast']} nodes",
            f"  Medium band:    {stats['band_counts']['medium']} nodes",
            f"  Slow band:      {stats['band_counts']['slow']} nodes",
        ]
        if self._results:
            avg_latency = sum(r.total_latency_ms for r in self._results) / len(self._results)
            max_latency = max(r.max_node_latency_ms for r in self._results)
            total_errors = sum(len(r.errors) for r in self._results)
            lines.extend([
                f"  Avg latency:    {avg_latency:.2f}ms",
                f"  Max latency:    {max_latency:.2f}ms",
                f"  Total errors:   {total_errors}",
            ])
        return "\n".join(lines)

    def export(
        self,
        path: Optional[str] = None,
        format: Optional[str] = None,
        include_metrics: bool = True,
        include_node_states: bool = False,
    ) -> Any:
        """Export simulation results to a file or memory.

        Args:
            path: Output file path. If None, exports to memory.
            format: Output format ("json", "csv", "npz"). Auto-detected from path.
            include_metrics: Whether to include Prometheus metrics.
            include_node_states: Whether to include per-node state snapshots.

        Returns:
            The ResultExporter instance (call .close() to finalize file output).

        Example:
            sim.run(steps=100)
            sim.export("results.json")
            # or to memory:
            exporter = sim.export()
            data = exporter.data
        """
        from .io.output import create_exporter

        exporter = create_exporter(path, format)
        if self._results:
            exporter.export_results(self._results)
        if include_metrics:
            exporter.export_metrics(self.metrics)
        if include_node_states:
            exporter.export_node_states(self.tree)
        exporter.close()
        return exporter
