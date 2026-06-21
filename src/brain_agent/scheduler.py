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
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

from .tree import BrainTree
from .node import BrainNode, Protocol
from .types import BrainData, Signal, Tensor, StateVector
from .exceptions import SchedulerConfigError
from .constants import DEFAULT_DT_FAST, DEFAULT_DT_MEDIUM, DEFAULT_DT_SLOW, DEFAULT_MAX_LATENCY_MS

if TYPE_CHECKING:
    from .monitoring.tracer import Tracer

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
    Validation is performed in __post_init__ to catch errors early.
    """
    dt_fast: float = DEFAULT_DT_FAST
    dt_medium: float = DEFAULT_DT_MEDIUM
    dt_slow: float = DEFAULT_DT_SLOW
    max_latency_ms: float = DEFAULT_MAX_LATENCY_MS
    max_steps: int = 0          # 0 = infinite
    enable_metrics: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values at construction time."""
        if self.dt_fast <= 0:
            raise SchedulerConfigError(
                f"dt_fast must be positive, got {self.dt_fast}."
            )
        if self.dt_medium <= 0:
            raise SchedulerConfigError(
                f"dt_medium must be positive, got {self.dt_medium}."
            )
        if self.dt_slow <= 0:
            raise SchedulerConfigError(
                f"dt_slow must be positive, got {self.dt_slow}."
            )
        if self.dt_medium < self.dt_fast:
            raise SchedulerConfigError(
                f"dt_medium ({self.dt_medium}) must be >= dt_fast ({self.dt_fast})."
            )
        if self.dt_slow < self.dt_medium:
            raise SchedulerConfigError(
                f"dt_slow ({self.dt_slow}) must be >= dt_medium ({self.dt_medium})."
            )
        if self.max_latency_ms <= 0:
            raise SchedulerConfigError(
                f"max_latency_ms must be positive, got {self.max_latency_ms}."
            )
        if self.max_steps < 0:
            raise SchedulerConfigError(
                f"max_steps must be >= 0 (0=infinite), got {self.max_steps}."
            )


@dataclass
class StepResult:
    """Result of a single simulation step.

    Attributes:
        step: Step number (1-based).
        sim_time: Simulation time in seconds.
        band: Frequency band that was active.
        nodes_processed: Number of nodes processed this step.
        total_latency_ms: Total processing latency in milliseconds.
        max_node_latency_ms: Maximum single-node latency in milliseconds.
        errors: List of error messages from failed nodes.
        node_outputs: Per-node output data for this step.
            Maps node_id to the BrainData output. Only populated when
            capture_outputs=True is passed to the scheduler.
        node_latencies: Per-node latency in milliseconds.
            Maps node_id to processing latency. Only populated when
            capture_outputs=True is passed to the scheduler.
    """
    step: int
    sim_time: float
    band: FrequencyBand
    nodes_processed: int
    total_latency_ms: float
    max_node_latency_ms: float
    errors: List[str] = field(default_factory=list)
    node_outputs: Dict[int, Any] = field(default_factory=dict)
    node_latencies: Dict[int, float] = field(default_factory=dict)


class FrequencyScheduler:
    """Multi-rate event loop for brain simulation.

    Drives the simulation by processing nodes at their designated frequencies.
    Fast nodes execute every step, medium nodes every 4 steps (20ms/5ms),
    slow nodes every 20 steps (100ms/5ms).
    """

    def __init__(
        self,
        tree: BrainTree,
        config: Optional[SchedulerConfig] = None,
        capture_outputs: bool = False,
    ):
        self._tree = tree
        self._config = config or SchedulerConfig()
        self._step = 0
        self._sim_time = 0.0
        self._running = False
        self._results: List[StepResult] = []
        self._capture_outputs = capture_outputs
        self._tracer: Optional["Tracer"] = None
        self._band_counters: Dict[FrequencyBand, int] = {
            FrequencyBand.FAST: 0,
            FrequencyBand.MEDIUM: 0,
            FrequencyBand.SLOW: 0,
        }
        self._band_nodes: Dict[FrequencyBand, List[BrainNode]] = {}
        # Track nodes already processed in current step (SYNC_CALL dedup)
        self._processed_this_step: set = set()
        # Cache for default inputs (avoids re-allocating zeros every step)
        self._default_input_cache: Dict[int, BrainData] = {}
        # Compute step ratios from dt values (configurable)
        self._medium_ratio = max(1, int(round(self._config.dt_medium / self._config.dt_fast)))
        self._slow_ratio = max(1, int(round(self._config.dt_slow / self._config.dt_fast)))
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

    def set_tracer(self, tracer: Optional["Tracer"]) -> None:
        """Enable or disable execution tracing.

        When a tracer is set, each node processing is recorded as a span,
        allowing users to visualize the data flow and identify bottlenecks.

        Args:
            tracer: A Tracer instance, or None to disable tracing.
        """
        self._tracer = tracer

    @property
    def tracer(self) -> Optional["Tracer"]:
        """The current tracer, or None if tracing is disabled."""
        return self._tracer

    @property
    def capture_outputs(self) -> bool:
        """Whether per-node outputs are being captured in StepResult."""
        return self._capture_outputs

    @property
    def step_count(self) -> int:
        """Current step number (0-based)."""
        return self._step

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds."""
        return self._sim_time

    def set_state(self, step: int, sim_time: float) -> None:
        """Restore scheduler state (used by checkpoint/resume).

        Args:
            step: Step number to restore.
            sim_time: Simulation time to restore.
        """
        self._step = step
        self._sim_time = sim_time

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

        Uses a cache to avoid re-allocating zero tensors every step.
        Returns a Signal for modulation nodes, Tensor for sensory/motor/relay,
        and StateVector for memory nodes. Shape/dimension is read from the
        node's input port specification.
        """
        # Check cache first
        if node.node_id in self._default_input_cache:
            return self._default_input_cache[node.node_id]

        in_port = node.input_ports.get("IN")
        result: BrainData
        if in_port is not None and in_port.data_type is not None:
            expected_type = in_port.data_type
            if expected_type is Tensor:
                # Use the port's expected shape, or a small default
                shape = in_port.shape if in_port.shape else (1, 1, 8, 8)
                result = Tensor.zeros(shape)
            elif expected_type is StateVector:
                # Use the port's expected dimension
                dim = in_port.shape[0] if in_port.shape else 128
                result = StateVector.zeros(dim)
            else:
                result = Signal(value=1.0, timestamp=self._sim_time)
        else:
            # Default: Signal for modulation/unknown nodes
            result = Signal(value=1.0, timestamp=self._sim_time)

        # Cache for reuse (Tensor/StateVector are immutable, safe to share)
        self._default_input_cache[node.node_id] = result
        return result

    def _process_node_with_protocol(self, node: BrainNode, data: BrainData) -> Optional[str]:
        """Process a node and propagate results to children via its protocol.

        Each protocol has distinct semantics:

        - ASYNC_PUSH: Fire-and-forget. Process parent, push result to
          children's input queues. Children process at their own frequency.
        - SYNC_CALL: Synchronous call. Process parent, then immediately
          process each child with the parent's result. Children are
          processed inline within this step (recursive).
        - PUB_SUB: Broadcast. Process parent, push result to all
          children's queues. Semantically a broadcast to subscribers.

        Returns error string if process() failed, None on success.
        """
        # Skip if already processed via SYNC_CALL recursion this step
        if node.node_id in self._processed_this_step:
            return None
        self._processed_this_step.add(node.node_id)

        # Start trace span if tracer is enabled
        span = None
        if self._tracer is not None:
            span = self._tracer.start_span(
                node_id=node.node_id,
                node_name=node.short_name,
                layer=node.layer,
                data_type=type(data).__name__,
            )

        t_start = time.perf_counter()
        result = node.safe_process(data)
        elapsed = time.perf_counter() - t_start
        # Record latency for generated nodes that don't self-report
        node.record_latency(elapsed)

        # Capture per-node output if enabled
        if self._capture_outputs:
            self._step_node_outputs[node.node_id] = result
            self._step_node_latencies[node.node_id] = elapsed * 1000.0

        # Finish trace span
        if span is not None and self._tracer is not None:
            self._tracer.finish_span(span, error=node.last_error)

        # Propagate to children based on protocol semantics
        if node.children:
            if node.protocol == Protocol.ASYNC_PUSH:
                # Fire-and-forget: queue result for children
                for child in node.children.values():
                    child.receive(result)

            elif node.protocol == Protocol.SYNC_CALL:
                # Synchronous: process each child immediately with parent's result
                for child in node.children.values():
                    self._process_node_with_protocol(child, result)
                    # Child errors are recorded but don't stop siblings

            else:  # PUB_SUB
                # Broadcast: queue result for all subscriber children
                for child in node.children.values():
                    child.receive(result)

        return node.last_error

    def step(self, input_data: Optional[BrainData] = None) -> StepResult:
        """Execute one simulation step.

        Processes all nodes in the fast band, and nodes in medium/slow bands
        based on their step counters. After processing each node, results
        are propagated to children's input queues via the node's protocol.

        Nodes already processed via SYNC_CALL recursive propagation are
        skipped in the band iteration to avoid double processing.
        """
        errors: List[str] = []
        nodes_processed = 0
        max_latency = 0.0
        total_latency = 0.0
        # Reset per-step processing tracker (for SYNC_CALL dedup)
        self._processed_this_step.clear()
        # Reset per-step capture dicts
        self._step_node_outputs: Dict[int, Any] = {}
        self._step_node_latencies: Dict[int, float] = {}

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
                    queued = node.pop_input()
                    if queued is not None:
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

        # Process medium band based on configurable ratio
        self._band_counters[FrequencyBand.MEDIUM] += 1
        if self._band_counters[FrequencyBand.MEDIUM] >= self._medium_ratio:
            self._band_counters[FrequencyBand.MEDIUM] = 0
            for node in self._band_nodes[FrequencyBand.MEDIUM]:
                try:
                    t0 = time.perf_counter()
                    queued = node.pop_input()
                    if queued is not None:
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

        # Process slow band based on configurable ratio
        self._band_counters[FrequencyBand.SLOW] += 1
        if self._band_counters[FrequencyBand.SLOW] >= self._slow_ratio:
            self._band_counters[FrequencyBand.SLOW] = 0
            for node in self._band_nodes[FrequencyBand.SLOW]:
                try:
                    t0 = time.perf_counter()
                    queued = node.pop_input()
                    if queued is not None:
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
            node_outputs=dict(self._step_node_outputs) if self._capture_outputs else {},
            node_latencies=dict(self._step_node_latencies) if self._capture_outputs else {},
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

    @property
    def medium_ratio(self) -> int:
        """Medium band step ratio (public read-only access)."""
        return self._medium_ratio

    @property
    def slow_ratio(self) -> int:
        """Slow band step ratio (public read-only access)."""
        return self._slow_ratio

    def reset(self) -> None:
        """Reset scheduler state."""
        self._step = 0
        self._sim_time = 0.0
        self._results.clear()
        self._default_input_cache.clear()
        for band in self._band_counters:
            self._band_counters[band] = 0
        for node in self._tree.bfs_traverse():
            node.reset()
