"""Output data adapters for brain simulation.

Provides exporters for writing simulation results to various formats:
JSON, CSV, NPZ (NumPy native), and in-memory storage.

Usage:
    from brain_agent.io import create_exporter
    from brain_agent import Simulation

    sim = Simulation.from_json("brain_tree_v5.json")
    results = sim.run(steps=100)

    exporter = create_exporter("results.json")
    exporter.export_results(results)
    exporter.export_metrics(sim.metrics)
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import numpy as np

from ..scheduler import StepResult
from ..monitoring import MetricsCollector, Tracer

logger = logging.getLogger(__name__)


class OutputAdapter:
    """Base class for output data adapters."""

    def write(self, data: Any) -> None:
        """Write data to the output."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources."""
        pass


class MemoryOutput(OutputAdapter):
    """In-memory output adapter for testing and programmatic access.

    Stores all written data in a list accessible via the `data` attribute.
    """

    def __init__(self) -> None:
        self.data: List[Any] = []

    def write(self, data: Any) -> None:
        self.data.append(data)


class FileOutput(OutputAdapter):
    """File-based output adapter.

    Writes data to a file in the specified format.
    """

    def __init__(
        self,
        path: Union[str, Path],
        format: Optional[Literal["json", "csv", "npz", "memory"]] = None,
    ) -> None:
        self._path = Path(path)
        self._format = format or self._path.suffix.lstrip(".")
        self._buffer: List[Any] = []

    def write(self, data: Any) -> None:
        self._buffer.append(data)

    def flush(self) -> None:
        """Write all buffered data to the file."""
        if not self._buffer:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._format == "json":
            self._flush_json()
        elif self._format == "csv":
            self._flush_csv()
        elif self._format == "npz":
            self._flush_npz()
        else:
            raise ValueError(
                f"Unsupported output format: '{self._format}'. "
                f"Supported formats: 'json', 'csv', 'npz', 'memory'."
            )

    def _flush_json(self) -> None:
        """Write buffer as JSON."""
        serializable: List[Any] = []
        for item in self._buffer:
            if isinstance(item, dict):
                serializable.append(item)
            elif isinstance(item, StepResult):
                serializable.append(self._step_result_to_dict(item))
            else:
                serializable.append(str(item))

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

    def _flush_csv(self) -> None:
        """Write buffer as CSV."""
        if not self._buffer:
            return

        first = self._buffer[0]
        if isinstance(first, StepResult):
            fieldnames = [
                "step", "sim_time", "band", "nodes_processed",
                "total_latency_ms", "max_node_latency_ms", "error_count",
            ]
            with open(self._path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in self._buffer:
                    writer.writerow({
                        "step": item.step,
                        "sim_time": item.sim_time,
                        "band": item.band.value,
                        "nodes_processed": item.nodes_processed,
                        "total_latency_ms": f"{item.total_latency_ms:.4f}",
                        "max_node_latency_ms": f"{item.max_node_latency_ms:.4f}",
                        "error_count": len(item.errors),
                    })
        elif isinstance(first, dict):
            fieldnames = list(first.keys())
            with open(self._path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in self._buffer:
                    writer.writerow(item)

    def _flush_npz(self) -> None:
        """Write buffer as NPZ (NumPy compressed)."""
        arrays: Dict[str, np.ndarray] = {}
        for i, item in enumerate(self._buffer):
            if isinstance(item, StepResult):
                arrays[f"step_{i}_step"] = np.array(item.step)
                arrays[f"step_{i}_sim_time"] = np.array(item.sim_time)
                arrays[f"step_{i}_nodes_processed"] = np.array(item.nodes_processed)
                arrays[f"step_{i}_total_latency_ms"] = np.array(item.total_latency_ms)
                arrays[f"step_{i}_max_latency_ms"] = np.array(item.max_node_latency_ms)
                arrays[f"step_{i}_error_count"] = np.array(len(item.errors))
            elif isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, (int, float)):
                        arrays[f"step_{i}_{k}"] = np.array(v)
            elif isinstance(item, np.ndarray):
                arrays[f"step_{i}"] = item

        np.savez_compressed(str(self._path), **arrays)  # type: ignore[arg-type]

    def close(self) -> None:
        """Flush and close."""
        self.flush()

    @staticmethod
    def _step_result_to_dict(result: StepResult) -> Dict[str, Any]:
        """Convert StepResult to a serializable dict."""
        return {
            "step": result.step,
            "sim_time": result.sim_time,
            "band": result.band.value,
            "nodes_processed": result.nodes_processed,
            "total_latency_ms": result.total_latency_ms,
            "max_node_latency_ms": result.max_node_latency_ms,
            "errors": result.errors,
        }


class ResultExporter:
    """High-level result exporter.

    Provides methods to export simulation results, metrics, and traces
    in various formats.

    Args:
        path: Output file path.
        format: Output format ("json", "csv", "npz"). Auto-detected from path if None.

    Example:
        from brain_agent.io import ResultExporter

        exporter = ResultExporter("results.json")
        exporter.export_results(results)
        exporter.export_metrics(metrics)
        exporter.close()
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        format: Optional[Literal["json", "csv", "npz", "memory"]] = None,
    ) -> None:
        self._path = Path(path) if path else None
        self._format = format
        self._output = FileOutput(path, format) if path else MemoryOutput()
        self._results: List[StepResult] = []
        self._metrics_str: Optional[str] = None
        self._trace_str: Optional[str] = None

    def __enter__(self) -> "ResultExporter":
        """Enter context manager. Returns self."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager. Closes the exporter."""
        self.close()

    def export_results(self, results: List[StepResult]) -> None:
        """Export simulation step results.

        Args:
            results: List of StepResult from a simulation run.
        """
        self._results = results
        for result in results:
            self._output.write(result)

    def export_metrics(self, metrics: MetricsCollector) -> None:
        """Export collected metrics.

        Args:
            metrics: MetricsCollector with collected data.
        """
        metrics.collect_node_metrics()
        self._metrics_str = metrics.export_prometheus()

    def export_trace(self, tracer: Tracer) -> None:
        """Export execution trace.

        Args:
            tracer: Tracer with recorded spans.
        """
        self._trace_str = tracer.format_trace()

    def export_node_states(self, tree: Any) -> Dict[int, Dict[str, Any]]:
        """Export current state of all nodes.

        Args:
            tree: BrainTree instance.

        Returns:
            Dict mapping node_id to state dict.
        """
        states: Dict[int, Dict[str, Any]] = {}
        for node in tree.bfs_traverse():
            metrics = node.get_metrics()
            metrics["last_error"] = node.last_error
            states[node.node_id] = metrics
        return states

    def save(self) -> None:
        """Save all exported data to the output file."""
        if isinstance(self._output, FileOutput):
            self._output.flush()
        elif isinstance(self._output, MemoryOutput):
            pass  # Data already in memory

    def close(self) -> None:
        """Flush and close the output."""
        self._output.close()

    @property
    def data(self) -> List[Any]:
        """Get in-memory data (only for MemoryOutput)."""
        if isinstance(self._output, MemoryOutput):
            return self._output.data
        return []

    def summary(self) -> Dict[str, Any]:
        """Get a summary of exported data.

        Returns:
            Dict with counts and statistics.
        """
        return {
            "result_count": len(self._results),
            "has_metrics": self._metrics_str is not None,
            "has_trace": self._trace_str is not None,
            "output_path": str(self._path) if self._path else None,
        }


def create_exporter(
    path: Optional[Union[str, Path]] = None,
    format: Optional[Literal["json", "csv", "npz", "memory"]] = None,
) -> ResultExporter:
    """Create a result exporter.

    Args:
        path: Output file path. If None, uses in-memory storage.
        format: Output format ("json", "csv", "npz"). Auto-detected from path.

    Returns:
        A ResultExporter instance.

    Example:
        # Export to JSON file
        exporter = create_exporter("results.json")
        exporter.export_results(results)
        exporter.close()

        # Export to memory (for testing)
        exporter = create_exporter()
        exporter.export_results(results)
        data = exporter.data
    """
    return ResultExporter(path, format)
