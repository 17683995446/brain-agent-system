"""Metrics collection and export for brain simulation.

Provides Prometheus-compatible metrics for monitoring node performance.
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..tree import BrainTree
from ..scheduler import StepResult


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics from all nodes and the scheduler.

    Metrics tracked:
    - brain_node_processing_latency_ms: per-node processing time
    - brain_node_process_count: per-node process call count
    - brain_node_queue_depth: per-node input queue depth
    - brain_system_step_latency_ms: per-step total latency
    - brain_system_nodes_processed: per-step node count
    - brain_system_errors_total: cumulative error count
    """

    def __init__(self, tree: Optional[BrainTree] = None):
        self._tree = tree
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._lock = threading.Lock()
        self._collection_interval = 1.0  # seconds
        self._running = False

    def collect_node_metrics(self):
        """Collect metrics from all nodes in the tree."""
        if not self._tree:
            return
        timestamp = time.time()
        for node in self._tree.bfs_traverse():
            m = node.get_metrics()
            self._add_metric(
                "brain_node_processing_latency_ms",
                m["avg_process_time_ms"],
                timestamp,
                {"node_id": str(m["node_id"]), "layer": str(m["layer"]), "type": m["region_type"]}
            )
            self._add_metric(
                "brain_node_process_count",
                m["process_count"],
                timestamp,
                {"node_id": str(m["node_id"]), "layer": str(m["layer"]), "type": m["region_type"]}
            )
            self._add_metric(
                "brain_node_queue_depth",
                m["queue_depth"],
                timestamp,
                {"node_id": str(m["node_id"]), "layer": str(m["layer"])}
            )

    def collect_step_result(self, result: StepResult):
        """Collect metrics from a scheduler step result."""
        timestamp = time.time()
        self._add_metric("brain_system_step_latency_ms", result.total_latency_ms, timestamp)
        self._add_metric("brain_system_nodes_processed", result.nodes_processed, timestamp)
        if result.errors:
            self._add_metric("brain_system_errors_total", len(result.errors), timestamp)

    def _add_metric(self, name: str, value: float, timestamp: float, labels: Optional[Dict[str, str]] = None):
        with self._lock:
            self._metrics[name].append(MetricPoint(
                name=name, value=value, timestamp=timestamp, labels=labels or {}
            ))

    def get_metric(self, name: str) -> List[MetricPoint]:
        with self._lock:
            return list(self._metrics.get(name, []))

    def get_latest(self, name: str) -> Optional[MetricPoint]:
        points = self.get_metric(name)
        return points[-1] if points else None

    def get_all_latest(self) -> Dict[str, float]:
        """Get latest value for all metrics."""
        result = {}
        with self._lock:
            for name, points in self._metrics.items():
                if points:
                    result[name] = points[-1].value
        return result

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        with self._lock:
            for name, points in self._metrics.items():
                latest = points[-1] if points else None
                if latest:
                    labels_str = ", ".join(f'{k}="{v}"' for k, v in latest.labels.items())
                    if labels_str:
                        labels_str = "{" + labels_str + "}"
                    lines.append(f"{name}{labels_str} {latest.value}")
        return "\n".join(lines)

    def clear(self):
        with self._lock:
            self._metrics.clear()
