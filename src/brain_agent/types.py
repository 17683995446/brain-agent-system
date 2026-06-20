"""Core data types for brain simulation.

All types are immutable dataclasses backed by numpy arrays.
Shape validation is performed at construction time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
import base64
import json
import numpy as np


class BrainDataType:
    """Base class for all brain data types.

    All brain data types must implement the nbytes property
    to report their memory footprint in bytes.
    """

    @property
    def nbytes(self) -> int:
        """Size in bytes of the underlying data buffer.

        Subclasses must override this property.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement nbytes property"
        )


@dataclass(frozen=True)
class Tensor(BrainDataType):
    """Multi-dimensional tensor data.

    Used for: visual, auditory, somatosensory processing
    Example shapes: [1, 100, 100], [32, 32, 8], [1, 64, 64]

    Args:
        data: numpy array with dtype float32
        shape: expected shape tuple (validated on construction)
    """
    data: np.ndarray
    shape: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Convert to float32 if needed
        object.__setattr__(self, 'data', np.asarray(self.data, dtype=np.float32))

        # If no expected shape was provided, infer it from the data
        expected_shape = self.shape if self.shape else self.data.shape
        object.__setattr__(self, 'shape', expected_shape)

        # Validate shape matches expected
        if self.data.shape != expected_shape:
            raise ValueError(
                f"张量形状不匹配: 期望 {expected_shape}, 实际 {self.data.shape}"
            )

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> "Tensor":
        """Create a zero tensor with given shape."""
        return cls(data=np.zeros(shape, dtype=np.float32), shape=shape)

    @classmethod
    def random(cls, shape: Tuple[int, ...]) -> "Tensor":
        """Create a random tensor with given shape."""
        return cls(data=np.random.randn(*shape).astype(np.float32), shape=shape)

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        return self.data.nbytes


@dataclass(frozen=True)
class StateVector(BrainDataType):
    """Fixed-length state vector.

    Used for: memory systems (hippocampus, amygdala, PFC)
    Example shapes: [128], [256], [64]

    Args:
        data: 1D numpy array with dtype float32
        dimension: expected dimension (validated)
    """
    data: np.ndarray
    dimension: int = 0

    def __post_init__(self) -> None:
        # Convert to float32 if needed
        object.__setattr__(self, 'data', np.asarray(self.data, dtype=np.float32))

        # If no expected dimension was provided, infer it from the data
        expected_dim = self.dimension if self.dimension > 0 else self.data.shape[0]
        object.__setattr__(self, 'dimension', expected_dim)

        # Validate that data is 1D
        if self.data.ndim != 1:
            raise ValueError(
                f"状态向量必须是一维数组, 实际维度为 {self.data.ndim}D, 形状为 {self.data.shape}"
            )

        # Validate dimension matches expected
        if self.data.shape[0] != expected_dim:
            raise ValueError(
                f"状态向量维度不匹配: 期望 {expected_dim}, 实际 {self.data.shape[0]}"
            )

    @classmethod
    def zeros(cls, dimension: int) -> "StateVector":
        """Create a zero state vector with given dimension."""
        return cls(data=np.zeros(dimension, dtype=np.float32), dimension=dimension)

    @classmethod
    def random(cls, dimension: int) -> "StateVector":
        """Create a random state vector with given dimension."""
        return cls(data=np.random.randn(dimension).astype(np.float32), dimension=dimension)

    def normalize(self) -> "StateVector":
        """L2 normalize the vector.

        Returns a new StateVector with unit length.
        If the vector has zero magnitude, returns a zero vector unchanged.
        """
        norm = np.linalg.norm(self.data)
        if norm == 0.0:
            return StateVector(data=np.zeros_like(self.data), dimension=self.dimension)
        return StateVector(data=(self.data / norm).astype(np.float32), dimension=self.dimension)

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.data))

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        return self.data.nbytes


@dataclass(frozen=True)
class SpikeTrain(BrainDataType):
    """Sparse spike event train.

    Used for: neural firing patterns
    Stored as structured array with (timestamp, neuron_id)

    Args:
        timestamps: 1D array of spike times in seconds
        neuron_ids: 1D array of neuron IDs that fired
        duration: total time window in seconds
    """
    timestamps: np.ndarray  # float64
    neuron_ids: np.ndarray   # int64
    duration: float = 1.0

    def __post_init__(self) -> None:
        # Ensure correct dtypes
        object.__setattr__(
            self, 'timestamps',
            np.asarray(self.timestamps, dtype=np.float64)
        )
        object.__setattr__(
            self, 'neuron_ids',
            np.asarray(self.neuron_ids, dtype=np.int64)
        )

        # Validate 1D arrays
        if self.timestamps.ndim != 1:
            raise ValueError(
                f"时间戳必须是一维数组, 实际维度为 {self.timestamps.ndim}D, "
                f"形状为 {self.timestamps.shape}"
            )
        if self.neuron_ids.ndim != 1:
            raise ValueError(
                f"神经元ID必须是一维数组, 实际维度为 {self.neuron_ids.ndim}D, "
                f"形状为 {self.neuron_ids.shape}"
            )

        # Validate same length
        if len(self.timestamps) != len(self.neuron_ids):
            raise ValueError(
                f"时间戳和神经元ID长度不匹配: "
                f"时间戳长度={len(self.timestamps)}, 神经元ID长度={len(self.neuron_ids)}"
            )

        # Validate duration is non-negative
        if self.duration < 0:
            raise ValueError(
                f"持续时间不能为负数, 实际值: {self.duration}"
            )

        # Validate timestamps are within duration
        if len(self.timestamps) > 0:
            if np.any(self.timestamps < 0) or np.any(self.timestamps > self.duration):
                raise ValueError(
                    f"时间戳必须在 [0, {self.duration}] 范围内, "
                    f"实际范围: [{self.timestamps.min()}, {self.timestamps.max()}]"
                )

    @classmethod
    def empty(cls, duration: float = 1.0) -> "SpikeTrain":
        """Create an empty spike train with no spikes."""
        return cls(
            timestamps=np.array([], dtype=np.float64),
            neuron_ids=np.array([], dtype=np.int64),
            duration=duration,
        )

    @property
    def spike_count(self) -> int:
        return len(self.timestamps)

    @property
    def firing_rate(self) -> float:
        """Spikes per second."""
        return self.spike_count / self.duration if self.duration > 0 else 0.0

    @property
    def nbytes(self) -> int:
        """Size in bytes (timestamps + neuron_ids arrays)."""
        return self.timestamps.nbytes + self.neuron_ids.nbytes


@dataclass(frozen=True)
class Signal(BrainDataType):
    """Scalar signal value.

    Used for: neuromodulators (dopamine, serotonin, etc.)
    Simple float32 scalar with timestamp.

    Args:
        value: signal amplitude
        timestamp: time of signal in seconds
    """
    value: float
    timestamp: float = 0.0

    @property
    def magnitude(self) -> float:
        return abs(self.value)

    @property
    def nbytes(self) -> int:
        """Size in bytes (scalar float64 = 8 bytes)."""
        return 8


@dataclass(frozen=True)
class Message(BrainDataType):
    """Generic message with string payload.

    Used for: control signals, configuration updates

    Args:
        topic: message topic/subject
        payload: dict payload
        timestamp: time of message
    """
    topic: str
    payload: dict = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def nbytes(self) -> int:
        """Size in bytes (topic + payload JSON + timestamp)."""
        payload_bytes = len(json.dumps(self.payload).encode('utf-8'))
        return len(self.topic.encode('utf-8')) + payload_bytes + 8


# Type alias for any brain data
BrainData = Tensor | StateVector | SpikeTrain | Signal | Message


def get_type_name(data: BrainData) -> str:
    """Get the type name string for a brain data instance."""
    return type(data).__name__


def validate_tensor_shape(data: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
    """Validate that a numpy array matches expected shape."""
    return bool(data.shape == expected_shape)


# ============================================================
# Serialization / Deserialization
# ============================================================

def _ndarray_to_base64(arr: np.ndarray) -> str:
    """Serialize a numpy array to base64-encoded bytes string."""
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _base64_to_ndarray(b64: str, dtype: str, shape: Tuple[int, ...]) -> np.ndarray:
    """Deserialize a base64-encoded string back to a numpy array."""
    raw = base64.b64decode(b64.encode("ascii"))
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def serialize(data: BrainData) -> Dict[str, Any]:
    """Serialize any BrainData instance to a plain dict.

    The resulting dict can be converted to JSON for network transfer
    or persisted to disk. Numpy arrays are base64-encoded.

    Args:
        data: A BrainData instance (Tensor, StateVector, SpikeTrain, Signal, Message)

    Returns:
        A dict with keys ``type`` and ``data``.
    """
    type_name = get_type_name(data)
    if isinstance(data, Tensor):
        return {
            "type": type_name,
            "data": {
                "shape": list(data.shape),
                "dtype": str(data.data.dtype),
                "buffer": _ndarray_to_base64(data.data),
            },
        }
    elif isinstance(data, StateVector):
        return {
            "type": type_name,
            "data": {
                "dimension": data.dimension,
                "dtype": str(data.data.dtype),
                "buffer": _ndarray_to_base64(data.data),
            },
        }
    elif isinstance(data, SpikeTrain):
        return {
            "type": type_name,
            "data": {
                "duration": data.duration,
                "timestamps_dtype": str(data.timestamps.dtype),
                "timestamps_buffer": _ndarray_to_base64(data.timestamps),
                "neuron_ids_dtype": str(data.neuron_ids.dtype),
                "neuron_ids_buffer": _ndarray_to_base64(data.neuron_ids),
            },
        }
    elif isinstance(data, Signal):
        return {
            "type": type_name,
            "data": {
                "value": data.value,
                "timestamp": data.timestamp,
            },
        }
    elif isinstance(data, Message):
        return {
            "type": type_name,
            "data": {
                "topic": data.topic,
                "payload": data.payload,
                "timestamp": data.timestamp,
            },
        }
    else:
        raise TypeError(f"Unsupported BrainData type: {type_name}")


def deserialize(serialized: Dict[str, Any]) -> BrainData:
    """Deserialize a dict back to a BrainData instance.

    This is the inverse of :func:`serialize`.

    Args:
        serialized: A dict produced by :func:`serialize`.

    Returns:
        The reconstructed BrainData instance.
    """
    type_name = serialized["type"]
    payload = serialized["data"]

    if type_name == "Tensor":
        shape = tuple(payload["shape"])
        dtype = payload["dtype"]
        arr = _base64_to_ndarray(payload["buffer"], dtype, shape)
        return Tensor(data=arr, shape=shape)
    elif type_name == "StateVector":
        dimension = payload["dimension"]
        dtype = payload["dtype"]
        arr = _base64_to_ndarray(payload["buffer"], dtype, (dimension,))
        return StateVector(data=arr, dimension=dimension)
    elif type_name == "SpikeTrain":
        duration = payload["duration"]
        ts_dtype = payload["timestamps_dtype"]
        nid_dtype = payload["neuron_ids_dtype"]
        timestamps = _base64_to_ndarray(
            payload["timestamps_buffer"], ts_dtype, (-1,)
        )
        neuron_ids = _base64_to_ndarray(
            payload["neuron_ids_buffer"], nid_dtype, (-1,)
        )
        return SpikeTrain(
            timestamps=timestamps,
            neuron_ids=neuron_ids,
            duration=duration,
        )
    elif type_name == "Signal":
        return Signal(value=payload["value"], timestamp=payload["timestamp"])
    elif type_name == "Message":
        return Message(
            topic=payload["topic"],
            payload=payload["payload"],
            timestamp=payload["timestamp"],
        )
    else:
        raise TypeError(f"Unknown BrainData type: {type_name}")


def to_json(data: BrainData) -> str:
    """Serialize a BrainData instance to a JSON string."""
    return json.dumps(serialize(data))


def from_json(json_str: str) -> BrainData:
    """Deserialize a JSON string back to a BrainData instance."""
    return deserialize(json.loads(json_str))
