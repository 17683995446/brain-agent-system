"""Multi-rate frequency scheduler for brain simulation.

Implements three frequency bands with exact specifications:
- Fast loop (60-200Hz, nominal=100Hz, dt=5ms): sensory/relay pathways (ASYNC_PUSH)
- Medium loop (5-50Hz, nominal=20Hz, dt=20ms): motor/memory pathways (SYNC_CALL)
- Slow loop (0.1-10Hz, nominal=1Hz, dt=100ms): modulation systems (PUB_SUB)

The scheduler drives the simulation by using each node's protocol method
(push_async / call_sync / publish) to propagate data through the tree.
"""

import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from .tree import BrainTree
from .node import BrainNode, Protocol
from .types import BrainData, Signal, Tensor, StateVector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrequencySpec:
    """Exact frequency band specification.

    Attributes:
        min_hz: Minimum frequency in Hz
        max_hz: Maximum frequency in Hz
        nominal_hz: Nominal (target) frequency in Hz
        dt_ms: Time step in milliseconds
        dt_s: Time step in seconds
    """
    min_hz: float
    max_hz: float
    nominal_hz: float
    dt_ms: float
    dt_s: float


class FrequencyBand(Enum):
    """Simulation frequency bands with exact specifications.

    Each band has a defined frequency range and fixed time step.
    """
    FAST = "fast"       # 60-200Hz, nominal=100Hz, dt=5ms
    MEDIUM = "medium"   # 5-50Hz, nominal=20Hz, dt=20ms
    SLOW = "slow"       # 0.1-10Hz, nominal=1Hz, dt=100ms

    @property
    def spec(self) -> FrequencySpec:
        """Return the exact FrequencySpec for this band."""
        _SPECS = {
            FrequencyBand.FAST: FrequencySpec(
                min_hz=60.0, max_hz=200.0, nominal_hz=100.0, dt_ms=5.0, dt_s=0.005
            ),
            FrequencyBand.MEDIUM: FrequencySpec(
                min_hz=5.0, max_hz=50.0, nominal_hz=20.0, dt_ms=20.0, dt_s=0.02
            ),
            FrequencyBand.SLOW: FrequencySpec(
                min_hz=0.1, max_hz=10.0, nominal_hz=1.0, dt_ms=100.0, dt_s=0.1
            ),
        }
        return _SPECS[self]


@dataclass
class SchedulerConfig:
    """Scheduler configuration.

    dt_fast, dt_medium, dt_slow must match the exact FrequencySpec values.
    """
    dt_fast: float = 0.005      # 5ms for fast loop (exact: 100Hz nominal)
    dt_medium: float = 0.02     # 20ms for medium loop (exact: 20Hz nominal)
    dt_slow: float = 0.1        # 100ms for slow loop (exact: 1Hz nominal)
    max_latency_ms: float = 10.0
    max_steps: int = 0          # 0 = infinite
    enable_metrics: bool = True


@dataclass
class StepResult:
    """Result of a single simulation step."""
    step: int
    sim_time: float
    band: FrequencyBand
    nodes_processed: int
    total_latency_ms: float
    max_node_latency_ms: float
    errors: List[str] = field(default_factory=list)


class FrequencyScheduler:
    """Multi-rate event loop for brain simulation.

    Drives the simulation by processing nodes at their designated frequencies.
    Fast nodes execute every step, medium nodes every 4 steps (20ms/5ms),
    slow nodes every 20 steps (100ms/5ms).
    """

    def __init__(self, tree: BrainTree, config: Optional[SchedulerConfig] = None):
        self._tree = tree
        self._config = config or SchedulerConfig()
        self._step = 0
        self._sim_time = 0.0
        self._running = False
        self._results: List[StepResult] = []
        self._band_counters: Dict[FrequencyBand, int] = {
            FrequencyBand.FAST: 0,
            FrequencyBand.MEDIUM: 0,
            FrequencyBand.SLOW: 0,
        }
        self._band_nodes: Dict[FrequencyBand, List[BrainNode]] = {}
        self._classify_nodes()

    def _classify_nodes(self) -> None:
        """Classify nodes into frequency bands based on their protocol."""
        self._band_nodes[FrequencyBand.FAST] = []
        self._band_nodes[FrequencyBand.MEDIUM] = []
        self._band_nodes[FrequencyBand.SLOW] = []

        for node in self._tree.bfs_traverse():
            band = self._get_band(node)
            self._band_nodes[band].append(node)

        logger.info(
            f"Node classification: "
            f"fast={len(self._band_nodes[FrequencyBand.FAST])}, "
            f"medium={len(self._band_nodes[FrequencyBand.MEDIUM])}, "
            f"slow={len(self._band_nodes[FrequencyBand.SLOW])}"
        )

    def _get_band(self, node: BrainNode) -> FrequencyBand:
        """Determine frequency band for a node based on its protocol."""
        if node.protocol == Protocol.ASYNC_PUSH:
            return FrequencyBand.FAST
        elif node.protocol == Protocol.SYNC_CALL:
            return FrequencyBand.MEDIUM
        else:  # PUB_SUB
            return FrequencyBand.SLOW

    def _get_default_input(self, node: BrainNode) -> BrainData:
        """Generate appropriate default input based on node's expected data type.

        Returns a Signal for modulation nodes, Tensor for sensory/motor/relay,
        and StateVector for memory nodes. Shape/dimension is read from the
        node's input port specification.
        """
        in_port = node._input_ports.get("IN")
        if in_port is not None and in_port.data_type is not None:
            expected_type = in_port.data_type
            if expected_type is Tensor:
                # Use the port's expected shape, or a small default
                shape = in_port.shape if in_port.shape else (1, 1, 8, 8)
                return Tensor.zeros(shape)
            elif expected_type is StateVector:
                # Use the port's expected dimension
                dim = in_port.shape[0] if in_port.shape else 128
                return StateVector.zeros(dim)
        # Default: Signal for modulation/unknown nodes
        return Signal(value=1.0, timestamp=self._sim_time)

    def _process_node_with_protocol(self, node: BrainNode, data: BrainData) -> Optional[str]:
        """Process a node and propagate results to children via its protocol.

        The scheduler calls safe_process(), then uses the protocol
        method to push results to children's input queues (without
        processing them). This preserves frequency band isolation.

        Note: Metrics are updated inside each node's process() method.
        The scheduler does NOT call _update_metrics to avoid double counting.

        - ASYNC_PUSH: process, then push_async to children (queue only)
        - SYNC_CALL: process, then queue results to children
        - PUB_SUB: process, then publish to children (queue only)

        Returns error string if process() failed, None on success.
        """
        # Process the node directly (metrics updated inside process())
        t_start = time.perf_counter()
        result = node.safe_process(data)
        elapsed = time.perf_counter() - t_start
        # Record latency for generated nodes that don't self-report
        node._record_scheduler_latency(elapsed)

        # Propagate to children based on protocol (queue only, don't process)
        if node._children:
            if node.protocol == Protocol.ASYNC_PUSH:
                for child in node._children.values():
                    child.receive(result)
            elif node.protocol == Protocol.SYNC_CALL:
                for child in node._children.values():
                    child.receive(result)
            else:  # PUB_SUB
                for child in node._children.values():
                    child.receive(result)

        return node._last_process_error

    def step(self, input_data: Optional[BrainData] = None) -> StepResult:
        """Execute one simulation step.

        Processes all nodes in the fast band, and nodes in medium/slow bands
        based on their step counters. After processing each node, results
        are propagated to children's input queues via the node's protocol.
        """
        errors: List[str] = []
        nodes_processed = 0
        max_latency = 0.0
        total_latency = 0.0

        # Determine the input data for this step
        # (external input for root, per-node defaults for others)
        external_input: Optional[BrainData] = input_data

        # Always process fast band
        for node in self._band_nodes[FrequencyBand.FAST]:
            try:
                t0 = time.perf_counter()
                if node.node_id == 0 and external_input is not None:
                    err = self._process_node_with_protocol(node, external_input)
                else:
                    if node._input_queue:
                        queued = node._input_queue.popleft()
                        err = self._process_node_with_protocol(node, queued)
                    else:
                        err = self._process_node_with_protocol(node, self._get_default_input(node))
                elapsed = (time.perf_counter() - t0) * 1000
                max_latency = max(max_latency, elapsed)
                total_latency += elapsed
                nodes_processed += 1
                if err is not None:
                    errors.append(err)
            except Exception as e:
                errors.append(f"Node {node.node_id}: {str(e)}")

        # Process medium band every 4 steps (20ms / 5ms = 4)
        self._band_counters[FrequencyBand.MEDIUM] += 1
        if self._band_counters[FrequencyBand.MEDIUM] >= 4:
            self._band_counters[FrequencyBand.MEDIUM] = 0
            for node in self._band_nodes[FrequencyBand.MEDIUM]:
                try:
                    t0 = time.perf_counter()
                    if node._input_queue:
                        queued = node._input_queue.popleft()
                        err = self._process_node_with_protocol(node, queued)
                    else:
                        err = self._process_node_with_protocol(node, self._get_default_input(node))
                    elapsed = (time.perf_counter() - t0) * 1000
                    max_latency = max(max_latency, elapsed)
                    total_latency += elapsed
                    nodes_processed += 1
                    if err is not None:
                        errors.append(err)
                except Exception as e:
                    errors.append(f"Node {node.node_id}: {str(e)}")

        # Process slow band every 20 steps (100ms / 5ms = 20)
        self._band_counters[FrequencyBand.SLOW] += 1
        if self._band_counters[FrequencyBand.SLOW] >= 20:
            self._band_counters[FrequencyBand.SLOW] = 0
            for node in self._band_nodes[FrequencyBand.SLOW]:
                try:
                    t0 = time.perf_counter()
                    if node._input_queue:
                        queued = node._input_queue.popleft()
                        err = self._process_node_with_protocol(node, queued)
                    else:
                        err = self._process_node_with_protocol(node, self._get_default_input(node))
                    elapsed = (time.perf_counter() - t0) * 1000
                    max_latency = max(max_latency, elapsed)
                    total_latency += elapsed
                    nodes_processed += 1
                    if err is not None:
                        errors.append(err)
                except Exception as e:
                    errors.append(f"Node {node.node_id}: {str(e)}")

        self._step += 1
        self._sim_time += self._config.dt_fast

        result = StepResult(
            step=self._step,
            sim_time=self._sim_time,
            band=FrequencyBand.FAST,
            nodes_processed=nodes_processed,
            total_latency_ms=total_latency,
            max_node_latency_ms=max_latency,
            errors=errors,
        )
        self._results.append(result)
        return result

    def run(
        self,
        steps: int = 100,
        input_data: Optional[BrainData] = None,
        shutdown_checker: Optional[Callable[[], bool]] = None,
    ) -> List[StepResult]:
        """Run simulation for N steps.

        Args:
            steps: Maximum number of steps to run.
            input_data: Optional input data for each step.
            shutdown_checker: Optional callable that returns True
                when the simulation should stop gracefully.

        Returns:
            List of StepResult for each completed step.
        """
        results = []
        for i in range(steps):
            if shutdown_checker is not None and shutdown_checker():
                logger.info(f"Graceful shutdown triggered at step {self._step}")
                break
            result = self.step(input_data)
            results.append(result)
            if result.errors:
                logger.warning(f"Step {result.step}: {len(result.errors)} errors")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "step": self._step,
            "sim_time": self._sim_time,
            "total_results": len(self._results),
            "band_counts": {
                band.value: len(nodes)
                for band, nodes in self._band_nodes.items()
            },
            "total_nodes": self._tree.total_nodes,
        }

    def reset(self) -> None:
        """Reset scheduler state."""
        self._step = 0
        self._sim_time = 0.0
        self._results.clear()
        for band in self._band_counters:
            self._band_counters[band] = 0
        for node in self._tree.bfs_traverse():
            node.reset()
