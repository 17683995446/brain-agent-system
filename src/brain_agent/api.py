"""High-level API for brain simulation.

Provides a simplified interface for creating and running brain simulations
without needing to understand the internal architecture.

Usage:
    from brain_agent import Simulation

    # Create simulation from JSON
    sim = Simulation.from_json("data/brain_tree_v5.json")

    # Run 100 steps with progress feedback
    results = sim.run(steps=100, on_step=lambda i, r: print(f"Step {i} done"))

    # Or use iterator API for fine-grained control
    for step_result in sim.iter_steps(steps=100, progress=True):
        process(step_result)

    # Get statistics
    stats = sim.stats
"""

import logging
import sys
import time
from typing import Iterator, List, Optional, Callable, Dict, Any, Literal

from .tree import BrainTree, BrainTreeBuilder
from .node import BrainNode
from .scheduler import FrequencyScheduler, SchedulerConfig, StepResult
from .monitoring import MetricsCollector, Tracer
from .types import BrainData
from .paths import find_brain_tree_json

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("brain-agent-system")
except Exception:
    __version__ = "0.5.0"

logger = logging.getLogger(__name__)

# Type alias for step callback
StepCallback = Callable[[int, StepResult], None]


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
        capture_outputs: bool = False,
    ) -> None:
        self.tree = tree
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = FrequencyScheduler(tree, capture_outputs=capture_outputs)
        self.metrics = metrics or MetricsCollector(tree)
        self.tracer = Tracer()
        self.scheduler.set_tracer(self.tracer)
        self._results: List[StepResult] = []

    def __enter__(self) -> "Simulation":
        """Enter context manager. Returns self."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager. Resets the simulation."""
        self.reset()

    @classmethod
    def from_json(
        cls,
        json_path: Optional[str] = None,
        config: Optional[SchedulerConfig] = None,
        capture_outputs: bool = False,
    ) -> "Simulation":
        """Create a Simulation from a brain tree JSON file.

        Args:
            json_path: Path to brain tree JSON. If None, searches standard locations.
            config: Optional scheduler configuration.
            capture_outputs: If True, capture per-node outputs in StepResult.
                Enables get_node_result() and to_dataframe(). Increases memory usage.

        Returns:
            A Simulation instance.
        """
        if json_path is None:
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

        if config is not None:
            scheduler = FrequencyScheduler(tree, config, capture_outputs=capture_outputs)
        else:
            scheduler = FrequencyScheduler(tree, capture_outputs=capture_outputs)
        metrics = MetricsCollector(tree)

        return cls(tree, scheduler, metrics)

    def run(
        self,
        steps: int = 100,
        input_data: Optional[BrainData] = None,
        input_callback: Optional[Callable[[int], BrainData]] = None,
        shutdown_checker: Optional[Callable[[], bool]] = None,
        on_step: Optional[StepCallback] = None,
        progress: bool = False,
    ) -> List[StepResult]:
        """Run the simulation for a number of steps.

        Args:
            steps: Number of simulation steps to run.
            input_data: Static input data fed to root node each step.
            input_callback: Function called each step with step number,
                returns input data for that step. Overrides input_data.
            shutdown_checker: Optional callable returning True to stop gracefully.
            on_step: Optional callback invoked after each step with
                (step_index, step_result). Useful for progress reporting,
                logging, or early termination via side effects.
            progress: If True, display a progress bar to stderr. Requires
                the 'tqdm' package. Ignored if on_step is provided.

        Returns:
            List of StepResult for each completed step.

        Example:
            # Static input
            results = sim.run(steps=100, input_data=Signal(value=1.0))

            # Dynamic input per step
            def my_input(step):
                return Signal(value=float(step) * 0.01)
            results = sim.run(steps=100, input_callback=my_input)

            # Progress callback
            def on_step(i, result):
                print(f"Step {i}: {result.nodes_processed} nodes, "
                      f"{result.total_latency_ms:.1f}ms")
            results = sim.run(steps=100, on_step=on_step)

            # Progress bar (requires tqdm)
            results = sim.run(steps=1000, progress=True)
        """
        results: List[StepResult] = []

        for i, step_result in enumerate(
            self.iter_steps(
                steps=steps,
                input_data=input_data,
                input_callback=input_callback,
                shutdown_checker=shutdown_checker,
            )
        ):
            results.append(step_result)
            if on_step is not None:
                on_step(i, step_result)

        if progress and on_step is None and not results:
            # Edge case: zero steps
            pass

        return results

    def iter_steps(
        self,
        steps: int = 100,
        input_data: Optional[BrainData] = None,
        input_callback: Optional[Callable[[int], BrainData]] = None,
        shutdown_checker: Optional[Callable[[], bool]] = None,
        progress: bool = False,
    ) -> Iterator[StepResult]:
        """Iterate over simulation steps, yielding one StepResult at a time.

        This generator allows fine-grained control over the simulation loop.
        Each iteration runs one simulation step and yields the result.
        The simulation state is updated as you iterate.

        Args:
            steps: Maximum number of simulation steps to run.
            input_data: Static input data fed to root node each step.
            input_callback: Function called each step with step index,
                returns input data for that step. Overrides input_data.
            shutdown_checker: Optional callable returning True to stop gracefully.
            progress: If True, display a progress bar to stderr (requires tqdm).

        Yields:
            StepResult for each completed step.

        Example:
            # Process results one at a time
            for result in sim.iter_steps(steps=100, progress=True):
                if result.errors:
                    print(f"Step {result.step} had errors: {result.errors}")
                if result.total_latency_ms > 100:
                    print(f"Slow step: {result.total_latency_ms:.1f}ms")

            # Early termination
            for result in sim.iter_steps(steps=1000):
                if result.total_latency_ms > 500:
                    break  # Stop early
        """
        if steps < 0:
            raise ValueError(
                f"steps must be >= 0, got {steps}. Use 0 for no steps."
            )
        if progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=steps, desc="Simulation", unit="step",
                            file=sys.stderr)
            except ImportError:
                logger.warning(
                    "tqdm not installed; progress bar disabled. "
                    "Install with: pip install tqdm"
                )
                progress = False

        for i in range(steps):
            if shutdown_checker is not None and shutdown_checker():
                break

            if input_callback is not None:
                data: Optional[BrainData] = input_callback(i)
            else:
                data = input_data

            result = self.scheduler.step(input_data=data)
            self.metrics.collect_step_result(result)
            self._results.append(result)

            if progress:
                pbar.update(1)
                pbar.set_postfix({
                    "nodes": result.nodes_processed,
                    "latency": f"{result.total_latency_ms:.0f}ms",
                })

            yield result

        if progress:
            pbar.close()

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

    def get_node_result(self, node_id: int, step: int) -> Optional[BrainData]:
        """Get the output of a specific node at a specific step.

        Requires capture_outputs=True when creating the Simulation.

        Args:
            node_id: The node ID to query.
            step: The step number (0-indexed).

        Returns:
            The BrainData output of the node at that step, or None if
            not captured or node was not processed at that step.

        Raises:
            RuntimeError: If capture_outputs was not enabled.
            IndexError: If step is out of range.
        """
        if not self.scheduler._capture_outputs:
            raise RuntimeError(
                "capture_outputs is not enabled. "
                "Create Simulation with capture_outputs=True to use get_node_result()."
            )
        if step < 0 or step >= len(self._results):
            raise IndexError(
                f"Step {step} out of range. "
                f"Valid range: 0 to {len(self._results) - 1}."
            )
        return self._results[step].node_outputs.get(node_id)

    def get_node_latency(self, node_id: int, step: int) -> Optional[float]:
        """Get the processing latency of a specific node at a specific step.

        Requires capture_outputs=True when creating the Simulation.

        Args:
            node_id: The node ID to query.
            step: The step number (0-indexed).

        Returns:
            Latency in milliseconds, or None if not captured.

        Raises:
            RuntimeError: If capture_outputs was not enabled.
            IndexError: If step is out of range.
        """
        if not self.scheduler._capture_outputs:
            raise RuntimeError(
                "capture_outputs is not enabled. "
                "Create Simulation with capture_outputs=True to use get_node_latency()."
            )
        if step < 0 or step >= len(self._results):
            raise IndexError(
                f"Step {step} out of range. "
                f"Valid range: 0 to {len(self._results) - 1}."
            )
        return self._results[step].node_latencies.get(node_id)

    def to_dataframe(self, node_ids: Optional[List[int]] = None) -> Any:
        """Convert simulation results to a pandas DataFrame.

        Requires capture_outputs=True and pandas installed.

        Args:
            node_ids: Optional list of node IDs to include. If None,
                includes all nodes that produced output.

        Returns:
            A pandas DataFrame with columns: step, sim_time, node_id,
            latency_ms, output_type, output_value (for Signal) or
            output_shape (for Tensor/StateVector).

        Raises:
            RuntimeError: If capture_outputs not enabled or pandas not installed.

        Example:
            sim = Simulation.from_json(capture_outputs=True)
            sim.run(steps=100)
            df = sim.to_dataframe(node_ids=[0, 1, 2])
            print(df.head())
        """
        if not self.scheduler._capture_outputs:
            raise RuntimeError(
                "capture_outputs is not enabled. "
                "Create Simulation with capture_outputs=True to use to_dataframe()."
            )
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError(
                "pandas is not installed. "
                "Install with: pip install pandas"
            )

        rows = []
        for i, result in enumerate(self._results):
            for nid, output in result.node_outputs.items():
                if node_ids is not None and nid not in node_ids:
                    continue
                latency = result.node_latencies.get(nid, 0.0)
                row = {
                    "step": i,
                    "sim_time": result.sim_time,
                    "node_id": nid,
                    "latency_ms": latency,
                    "output_type": type(output).__name__,
                }
                # Extract value for Signal, shape for Tensor/StateVector
                if hasattr(output, "value"):
                    row["output_value"] = output.value
                if hasattr(output, "shape"):
                    row["output_shape"] = str(output.shape)
                if hasattr(output, "dimension"):
                    row["output_dimension"] = output.dimension
                rows.append(row)

        return pd.DataFrame(rows)

    def save_checkpoint(self, path: str) -> None:
        """Save simulation state to a checkpoint file.

        Serializes the scheduler step/time, all node internal states,
        and recent results to a JSON file. Use load_checkpoint() to restore.

        Args:
            path: Output file path (JSON format).

        Example:
            sim.run(steps=50)
            sim.save_checkpoint("checkpoint_step50.json")
            # ... later, possibly in a new process ...
            sim2 = Simulation.from_json()
            sim2.load_checkpoint("checkpoint_step50.json")
            sim2.run(steps=50)  # Continue from step 50
        """
        import json

        # Collect node states
        node_states = {}
        for node in self.tree.bfs_traverse():
            state = {}
            for key, value in node._state.items():
                state[key] = self._serialize_value(value)
            node_states[node.node_id] = state

        checkpoint = {
            "version": __version__,
            "step": self.scheduler.step_count,
            "sim_time": self.scheduler.sim_time,
            "node_states": node_states,
            "results_count": len(self._results),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved to {path} (step={checkpoint['step']})")

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a value for JSON, handling numpy types recursively."""
        import numpy as np
        if isinstance(value, np.ndarray):
            return {"__ndarray__": True, "data": value.tolist(),
                    "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return {"__npscalar__": True, "data": value.item(),
                    "dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {k: Simulation._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Simulation._serialize_value(v) for v in value]
        return value

    @staticmethod
    def _deserialize_value(value: Any) -> Any:
        """Deserialize a value from JSON, restoring numpy types."""
        import numpy as np
        if isinstance(value, dict):
            if value.get("__ndarray__"):
                return np.array(value["data"], dtype=value.get("dtype", "float32"))
            if value.get("__npscalar__"):
                return np.array(value["data"], dtype=value.get("dtype", "float32")).item()
        return value

    def load_checkpoint(self, path: str) -> None:
        """Load simulation state from a checkpoint file.

        Restores scheduler step/time and all node internal states.
        Previous results are cleared.

        Args:
            path: Checkpoint file path (JSON format).

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
            ValueError: If the checkpoint is corrupted or incompatible.
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        # Restore scheduler state
        self.scheduler._step = checkpoint["step"]
        self.scheduler._sim_time = checkpoint["sim_time"]

        # Restore node states (JSON keys are strings, convert to int)
        node_states = checkpoint.get("node_states", {})
        node_states_int = {int(k): v for k, v in node_states.items()}
        for node in self.tree.bfs_traverse():
            if node.node_id in node_states_int:
                for key, value in node_states_int[node.node_id].items():
                    node._state[key] = self._deserialize_value(value)

        # Clear previous results
        self._results.clear()

        logger.info(
            f"Checkpoint loaded from {path} "
            f"(step={checkpoint['step']}, sim_time={checkpoint['sim_time']:.4f})"
        )

    @property
    def step_count(self) -> int:
        """Get the current step number."""
        return self.scheduler.step_count

    @property
    def sim_time(self) -> float:
        """Get the current simulation time in seconds."""
        return self.scheduler.sim_time

    def export_metrics(self, format: Literal["prometheus"] = "prometheus") -> str:
        """Export collected metrics.

        Args:
            format: Output format. Currently supports "prometheus".

        Returns:
            Metrics string in the requested format.
        """
        self.metrics.collect_node_metrics()
        if format == "prometheus":
            return self.metrics.export_prometheus()
        raise ValueError(f"Unsupported metrics format: '{format}'. Supported formats: 'prometheus'.")

    def get_node(self, node_id: int) -> Optional[BrainNode]:
        """Get a node by ID.

        Args:
            node_id: The node ID to look up.

        Returns:
            The BrainNode with the given ID, or None if not found.
        """
        return self.tree.get_node(node_id)

    def get_node_strict(self, node_id: int) -> BrainNode:
        """Get a node by ID, raising KeyError if not found.

        Args:
            node_id: The node ID to look up.

        Returns:
            The BrainNode with the given ID.

        Raises:
            KeyError: If no node with the given ID exists.
        """
        node = self.tree.get_node(node_id)
        if node is None:
            raise KeyError(
                f"No node found with id={node_id}. "
                f"Valid IDs: 0 to {self.tree.total_nodes - 1}."
            )
        return node

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
        format: Optional[Literal["json", "csv", "npz"]] = None,
        include_metrics: bool = True,
        include_node_states: bool = False,
    ) -> Any:
        """Export simulation results to a file or memory.

        When exporting to a file (path is provided), the file is written and
        the exporter is closed automatically. When exporting to memory (path
        is None), the exporter remains open for further access to .data.

        Args:
            path: Output file path. If None, exports to memory.
            format: Output format ("json", "csv", "npz"). Auto-detected from path.
            include_metrics: Whether to include Prometheus metrics.
            include_node_states: Whether to include per-node state snapshots.

        Returns:
            The ResultExporter instance. For file exports, the exporter is
            already closed. For memory exports, access .data for results.

        Example:
            # File export (auto-closed)
            sim.run(steps=100)
            sim.export("results.json")

            # Memory export (access .data)
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

        # Auto-close for file exports; keep open for memory exports
        if path is not None:
            exporter.close()

        return exporter
