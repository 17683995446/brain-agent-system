"""Brain node base class for the brain simulation agent system.

Each node in the 2047-node binary tree is an instance of BrainNode (or subclass).
Nodes communicate via three protocols: ASYNC_PUSH, SYNC_CALL, PUB_SUB.
"""

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Type
from collections import deque
import time
import logging
import ast

from .types import BrainData, Tensor, StateVector, Signal, Message

logger = logging.getLogger(__name__)

# Maximum input queue depth. When full, oldest items are dropped.
DEFAULT_MAX_QUEUE_SIZE = 1000


class RegionType(Enum):
    """Brain region classification."""
    SENSORY = "sensory"
    MOTOR = "motor"
    MEMORY = "memory"
    MODULATION = "modulation"
    RELAY = "relay"
    UNKNOWN = "unknown"


class Protocol(Enum):
    """Communication protocol types."""
    ASYNC_PUSH = "ASYNC_PUSH"
    SYNC_CALL = "SYNC_CALL"
    PUB_SUB = "PUB_SUB"


# RegionType -> data type mapping per INTERFACE_SPEC.md
REGION_TYPE_DATA_MAP: Dict[RegionType, Any] = {
    RegionType.SENSORY: Tensor,
    RegionType.MOTOR: Tensor,
    RegionType.MEMORY: StateVector,
    RegionType.MODULATION: Signal,
    RegionType.RELAY: Tensor,
    RegionType.UNKNOWN: BrainData,
}

# RegionType -> standard shape mapping per INTERFACE_SPEC.md
REGION_TYPE_SHAPE_MAP: Dict[RegionType, Optional[Tuple[int, ...]]] = {
    RegionType.SENSORY: (1, 8, 32, 32),
    RegionType.MOTOR: (1, 1, 32, 32),
    RegionType.MEMORY: (128,),
    RegionType.MODULATION: None,  # scalar
    RegionType.RELAY: (1, 1, 32, 32),
    RegionType.UNKNOWN: None,
}


@dataclass
class PortSpec:
    """Input/output port specification."""
    port_id: str
    name: str
    data_type: Type  # Type[BrainData] - Tensor, StateVector, Signal, etc.
    shape: Optional[tuple] = None
    frequency: str = ""
    protocol: Protocol = Protocol.ASYNC_PUSH
    required: bool = True


@dataclass
class NodeConfig:
    """Configuration for a brain node."""
    node_id: int
    name_cn: str
    short_name: str
    folder_name: str
    layer: int
    parent_id: Optional[int]
    responsibility: str
    region_type: RegionType
    tensor_shape: str
    frequency: str
    protocol: Protocol
    algorithm: str
    children_ids: List[int] = field(default_factory=list)


class BrainNode(ABC):
    """Base class for all brain nodes.

    Each of the 2047 nodes in the binary tree inherits from this class.
    Subclasses must implement the process() method.

    Lifecycle:
    1. Create node with NodeConfig
    2. Connect to parent/children
    3. Call process() to handle input data
    4. Results propagate to children via protocol
    """

    def __init__(self, config: NodeConfig):
        # BrainNode is an abstract base class. ``process`` is provided as a
        # concrete template method (subclasses override the ``_process_*``
        # hooks instead), so we guard direct instantiation here to preserve
        # the "cannot instantiate BrainNode directly" contract.
        if type(self) is BrainNode:
            raise TypeError(
                "Cannot instantiate abstract class BrainNode directly; "
                "subclass it and override the _process_* hooks."
            )
        self._config = config
        self._state: Dict[str, Any] = {}
        self._children: Dict[int, BrainNode] = {}
        self._parent: Optional[BrainNode] = None
        self._input_queue: deque = deque(maxlen=DEFAULT_MAX_QUEUE_SIZE)
        self._output_history: List[BrainData] = []
        self._process_count: int = 0
        self._last_process_time: float = 0.0
        self._total_process_time: float = 0.0
        self._last_process_error: Optional[str] = None
        self._input_ports: Dict[str, PortSpec] = {}
        self._output_ports: Dict[str, PortSpec] = {}
        self._setup_ports()
        logger.debug(f"Node {config.node_id} ({config.short_name}) initialized at layer {config.layer}")

    def _setup_ports(self) -> None:
        """Set up input/output ports based on region_type mapping.

        Uses INTERFACE_SPEC.md RegionType -> data_type mapping to configure
        the correct data type and shape for each port.
        """
        expected_type = REGION_TYPE_DATA_MAP.get(
            self._config.region_type, BrainData
        )
        expected_shape = REGION_TYPE_SHAPE_MAP.get(
            self._config.region_type, None
        )
        # Parse tensor_shape string if available and region uses Tensor
        parsed_shape = self._parse_tensor_shape(self._config.tensor_shape)
        if expected_type is Tensor and parsed_shape is not None:
            expected_shape = parsed_shape

        self._input_ports["IN"] = PortSpec(
            port_id="IN", name="Input Signal",
            data_type=expected_type, shape=expected_shape,
            frequency=self._config.frequency,
            protocol=self._config.protocol,
            required=True,
        )
        self._output_ports["OUT"] = PortSpec(
            port_id="OUT", name="Output Signal",
            data_type=expected_type, shape=expected_shape,
            frequency=self._config.frequency,
            protocol=self._config.protocol,
            required=True,
        )

    @staticmethod
    def _parse_tensor_shape(shape_str: str) -> Optional[Tuple[int, ...]]:
        """Parse a tensor shape string into a tuple of ints.

        Supports formats like:
        - "[1,8,32,32]"
        - "(1, 8, 32, 32)"
        - "1x8x32x32"
        - "32x32x8"

        Returns None if parsing fails or shape_str is empty.
        """
        if not shape_str or not shape_str.strip():
            return None
        s = shape_str.strip()
        try:
            # Try bracket/parenthesis formats
            if (s.startswith("[") and s.endswith("]")) or \
               (s.startswith("(") and s.endswith(")")):
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return tuple(int(x) for x in parsed)
            # Try "x" separated format
            if "x" in s:
                parts = s.split("x")
                return tuple(int(p.strip()) for p in parts)
            # Try comma separated without brackets
            if "," in s:
                parts = s.split(",")
                return tuple(int(p.strip()) for p in parts)
        except Exception:
            return None
        return None

    def _validate_input(self, data: BrainData) -> None:
        """Validate input data against the IN port specification.

        Enforces:
        1. Data type matches expected type (or is a subtype)
        2. For Tensor: shape matches expected shape if defined

        Raises:
            ValueError: If validation fails.
        """
        in_port = self._input_ports.get("IN")
        if in_port is None:
            return

        expected_type = in_port.data_type
        if expected_type != BrainData and not isinstance(data, expected_type):
            actual_type = type(data).__name__
            expected_name = getattr(expected_type, '__name__', str(expected_type))
            raise ValueError(
                f"Node {self.node_id} ({self.short_name}): "
                f"Input data type mismatch: expected {expected_name}, actual {actual_type}"
            )

        # Shape validation for Tensor
        if isinstance(data, Tensor) and in_port.shape is not None:
            if data.shape != in_port.shape:
                raise ValueError(
                    f"Node {self.node_id} ({self.short_name}): "
                    f"Input Tensor shape mismatch: expected {in_port.shape}, actual {data.shape}"
                )

    def _validate_output(self, data: BrainData) -> None:
        """Validate output data against the OUT port specification.

        Enforces:
        1. Data type matches expected type (or is a subtype)
        2. For Tensor: shape matches expected shape if defined

        Raises:
            ValueError: If validation fails.
        """
        out_port = self._output_ports.get("OUT")
        if out_port is None:
            return

        expected_type = out_port.data_type
        if expected_type != BrainData and not isinstance(data, expected_type):
            actual_type = type(data).__name__
            expected_name = getattr(expected_type, '__name__', str(expected_type))
            raise ValueError(
                f"Node {self.node_id} ({self.short_name}): "
                f"Output data type mismatch: expected {expected_name}, actual {actual_type}"
            )

        # Shape validation for Tensor
        if isinstance(data, Tensor) and out_port.shape is not None:
            if data.shape != out_port.shape:
                raise ValueError(
                    f"Node {self.node_id} ({self.short_name}): "
                    f"Output Tensor shape mismatch: expected {out_port.shape}, actual {data.shape}"
                )

    @property
    def node_id(self) -> int:
        return self._config.node_id

    @property
    def name_cn(self) -> str:
        return self._config.name_cn

    @property
    def short_name(self) -> str:
        return self._config.short_name

    @property
    def folder_name(self) -> str:
        return self._config.folder_name

    @property
    def layer(self) -> int:
        return self._config.layer

    @property
    def region_type(self) -> RegionType:
        return self._config.region_type

    @property
    def protocol(self) -> Protocol:
        return self._config.protocol

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def children(self) -> Dict[int, BrainNode]:
        return self._children

    @property
    def parent(self) -> Optional[BrainNode]:
        return self._parent

    @property
    def avg_process_time_ms(self) -> float:
        if self._process_count == 0:
            return 0.0
        return (self._total_process_time / self._process_count) * 1000

    @property
    def last_error(self) -> Optional[str]:
        """Last process error message, or None if last process succeeded."""
        return self._last_process_error

    def connect_parent(self, parent: "BrainNode") -> None:
        """Connect this node to a parent node."""
        self._parent = parent

    def connect_child(self, child: "BrainNode") -> None:
        """Connect a child node."""
        self._children[child.node_id] = child
        child.connect_parent(self)

    def disconnect_child(self, child_id: int) -> None:
        """Disconnect a child node."""
        if child_id in self._children:
            self._children[child_id]._parent = None
            del self._children[child_id]

    def process(self, data: BrainData) -> BrainData:
        """Process input data and return output.

        This is the core computation method, implemented as a template
        method: it records timing, dispatches to a type-specific
        ``_process_*`` hook based on the runtime type of ``data``,
        updates processing metrics, and returns the result.

        Subclasses customize behavior by overriding the ``_process_*``
        hooks (``_process_tensor``, ``_process_state_vector``,
        ``_process_signal``, ``_process_message``) rather than overriding
        this method. ``Message`` inputs pass through unchanged by default.

        Args:
            data: Input data from parent or external source

        Returns:
            Processed output data

        Raises:
            ValueError: If input data type is not supported
        """
        start = time.perf_counter()
        if isinstance(data, Tensor):
            result: BrainData = self._process_tensor(data)
        elif isinstance(data, StateVector):
            result = self._process_state_vector(data)
        elif isinstance(data, Signal):
            result = self._process_signal(data)
        elif isinstance(data, Message):
            result = self._process_message(data)
        else:
            raise ValueError(
                f"Unsupported data type: {type(data).__name__}. "
                f"Supported types: Tensor, StateVector, Signal, Message, SpikeTrain."
            )

        self._update_metrics(time.perf_counter() - start)
        return result

    def _process_tensor(self, data: Tensor) -> BrainData:
        """Process a Tensor input.

        Override in subclasses to implement tensor-specific logic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _process_tensor(). "
            f"Override this method to handle Tensor input, or override process() for custom dispatch."
        )

    def _process_state_vector(self, data: StateVector) -> BrainData:
        """Process a StateVector input.

        Override in subclasses to implement state-vector-specific logic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _process_state_vector(). "
            f"Override this method to handle StateVector input, or override process() for custom dispatch."
        )

    def _process_signal(self, data: Signal) -> BrainData:
        """Process a Signal input.

        Override in subclasses to implement signal-specific logic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _process_signal(). "
            f"Override this method to handle Signal input, or override process() for custom dispatch."
        )

    def _process_message(self, data: Message) -> BrainData:
        """Process a Message input.

        Default behavior is pass-through: the message is returned
        unchanged. Override in subclasses to customize.
        """
        return data

    def _get_default_output(self) -> BrainData:
        """Create a zero/default output based on the node's output port spec.

        Used as fallback when process() raises an exception.
        """
        out_port = self._output_ports.get("OUT")
        if out_port is not None and out_port.data_type is not None:
            expected_type = out_port.data_type
            shape = out_port.shape if out_port.shape else (1, 1, 8, 8)
            if expected_type is Tensor:
                return Tensor.zeros(shape)
            elif expected_type is StateVector:
                dim = shape[0] if shape else 128
                return StateVector.zeros(dim)
        return Signal(value=0.0, timestamp=time.time())

    def safe_process(self, data: BrainData) -> BrainData:
        """Process with validation and graceful degradation.

        Wraps process() with _validate_input and _validate_output.
        If validation fails, logs a warning and continues (graceful degradation).
        If process() itself raises, logs the error and returns a default output.

        Use this in production code paths instead of calling process() directly.
        """
        try:
            self._validate_input(data)
        except ValueError as e:
            logger.warning(f"Input validation failed (degrading): {e}")

        try:
            result = self.process(data)
        except Exception as e:
            error_msg = f"Node {self.node_id}: {str(e)}"
            logger.warning(
                f"Node {self.node_id} ({self.short_name}) process() failed "
                f"(returning default): {e}"
            )
            self._last_process_error = error_msg
            return self._get_default_output()

        self._last_process_error = None

        try:
            self._validate_output(result)
        except ValueError as e:
            logger.warning(f"Output validation failed (degrading): {e}")

        return result

    def push_async(self, data: BrainData) -> None:
        """ASYNC_PUSH protocol: fire-and-forget to children.

        Process data, then push results to all children without waiting.
        Used by sensory pathways (60-200Hz).
        """
        start = time.perf_counter()
        result = self.process(data)
        elapsed = time.perf_counter() - start
        self._update_metrics(elapsed)

        for child in self._children.values():
            child.receive(result)

        logger.debug(f"[ASYNC_PUSH] {self.short_name} -> {len(self._children)} children")

    def call_sync(self, data: BrainData) -> List[BrainData]:
        """SYNC_CALL protocol: process and wait for all children.

        Process data, call all children synchronously, collect results.
        Used by memory and motor pathways (5-50Hz).

        Returns:
            List of results from all children
        """
        start = time.perf_counter()
        result = self.process(data)
        elapsed = time.perf_counter() - start
        self._update_metrics(elapsed)

        child_results = []
        for child in self._children.values():
            child_result = child.process(result)
            child_results.append(child_result)

        logger.debug(f"[SYNC_CALL] {self.short_name} -> {len(child_results)} results")
        return child_results

    def publish(self, data: BrainData) -> None:
        """PUB_SUB protocol: broadcast to all subscribers.

        Process data, then publish to subscriber callback.
        Used by modulation systems (1-10Hz).
        """
        start = time.perf_counter()
        result = self.process(data)
        elapsed = time.perf_counter() - start
        self._update_metrics(elapsed)

        for child in self._children.values():
            child.receive(result)

        logger.debug(f"[PUB_SUB] {self.short_name} published to {len(self._children)} subscribers")

    def receive(self, data: BrainData) -> None:
        """Receive data from parent (called by parent's push/publish).

        If the input queue is full, the oldest item is dropped
        (backpressure via drop-oldest strategy).
        """
        maxlen = self._input_queue.maxlen
        if maxlen is not None and len(self._input_queue) >= maxlen:
            logger.warning(
                f"Node {self.node_id} ({self.short_name}) "
                f"input queue full, dropping oldest item"
            )
        self._input_queue.append(data)

    def _update_metrics(self, elapsed: float) -> None:
        """Update processing metrics.

        Called internally by node's process() method. Increments
        process_count and records latency.
        """
        self._process_count += 1
        self._last_process_time = elapsed
        self._total_process_time += elapsed

    def _record_scheduler_latency(self, elapsed: float) -> None:
        """Record latency measured by the scheduler.

        Unlike _update_metrics, this does NOT increment process_count
        (avoiding double counting). It only updates latency stats if
        the node did not already record them in process().
        """
        # Only update if the node hasn't recorded its own latency
        # (i.e., _last_process_time is still 0 or this is a generated node)
        if self._last_process_time == 0.0:
            self._last_process_time = elapsed
            self._total_process_time += elapsed

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get internal state value."""
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set internal state value."""
        self._state[key] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get node processing metrics."""
        return {
            "node_id": self.node_id,
            "short_name": self.short_name,
            "layer": self.layer,
            "region_type": self.region_type.value,
            "process_count": self._process_count,
            "last_process_time_ms": self._last_process_time * 1000,
            "avg_process_time_ms": self.avg_process_time_ms,
            "queue_depth": len(self._input_queue),
            "children_count": len(self._children),
            "is_leaf": self.is_leaf,
        }

    def reset(self) -> None:
        """Reset node state and metrics."""
        self._state.clear()
        self._input_queue.clear()
        self._output_history.clear()
        self._process_count = 0
        self._last_process_time = 0.0
        self._total_process_time = 0.0

    def __repr__(self) -> str:
        return (f"BrainNode(id={self.node_id}, name={self.short_name}, "
                f"layer=L{self.layer:02d}, type={self.region_type.value})")
