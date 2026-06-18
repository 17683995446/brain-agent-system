"""Zero-copy transport layer for brain data types.

Provides shared-memory-style buffer management for efficient
inter-node communication without data copying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

from brain_agent.types import (
    BrainData,
    Tensor,
    StateVector,
    SpikeTrain,
    Signal,
    Message,
)


@dataclass(frozen=True)
class BufferView:
    """A read-only view into a shared memory buffer.

    Attributes:
        buffer_id: Unique identifier for the buffer.
        offset: Byte offset into the underlying buffer.
        size: Size in bytes of this view.
        dtype: Numpy dtype string (e.g. 'float32').
        shape: Shape tuple for multi-dimensional arrays.
    """
    buffer_id: str
    offset: int
    size: int
    dtype: str
    shape: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict for serialization."""
        return {
            "buffer_id": self.buffer_id,
            "offset": self.offset,
            "size": self.size,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BufferView":
        """Reconstruct from a plain dict."""
        return cls(
            buffer_id=d["buffer_id"],
            offset=d["offset"],
            size=d["size"],
            dtype=d["dtype"],
            shape=tuple(d["shape"]),
        )


class ZeroCopyBuffer:
    """A zero-copy buffer manager backed by numpy arrays.

    This class simulates shared-memory semantics by managing
    numpy arrays that can be viewed without copying data.
    In a distributed deployment, this would be backed by
    Apache Arrow Plasma or similar shared-memory store.
    """

    def __init__(self, buffer_id: str, capacity_bytes: int = 1024 * 1024):
        self._buffer_id = buffer_id
        self._capacity = capacity_bytes
        self._data = np.zeros(capacity_bytes, dtype=np.uint8)
        self._offset = 0

    @property
    def buffer_id(self) -> str:
        return self._buffer_id

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def free_space(self) -> int:
        return self._capacity - self._offset

    def allocate(self, size_bytes: int, dtype: str, shape: Tuple[int, ...]) -> BufferView:
        """Allocate a region in the buffer and return a view.

        Args:
            size_bytes: Number of bytes to allocate.
            dtype: Numpy dtype string.
            shape: Expected shape of the data.

        Returns:
            A BufferView describing the allocated region.

        Raises:
            MemoryError: If the buffer does not have enough free space.
        """
        if size_bytes > self.free_space:
            raise MemoryError(
                f"Buffer {self._buffer_id} out of space: "
                f"need {size_bytes}, have {self.free_space}"
            )
        view = BufferView(
            buffer_id=self._buffer_id,
            offset=self._offset,
            size=size_bytes,
            dtype=dtype,
            shape=shape,
        )
        self._offset += size_bytes
        return view

    def write(self, view: BufferView, arr: np.ndarray) -> None:
        """Write a numpy array into the buffer at the given view's offset.

        Args:
            view: BufferView returned by allocate().
            arr: Numpy array to write. Must match view's size and dtype.
        """
        if view.buffer_id != self._buffer_id:
            raise ValueError(
                f"View buffer_id {view.buffer_id} does not match "
                f"this buffer {self._buffer_id}"
            )
        if arr.nbytes != view.size:
            raise ValueError(
                f"Array size {arr.nbytes} does not match view size {view.size}"
            )
        self._data[view.offset : view.offset + view.size] = np.frombuffer(
            arr.tobytes(), dtype=np.uint8
        )

    def read(self, view: BufferView) -> np.ndarray:
        """Read a numpy array from the buffer at the given view's offset.

        This operation returns a view into the underlying buffer
        (zero-copy) whenever possible.

        Args:
            view: BufferView describing the region to read.

        Returns:
            A numpy array (possibly a view) of the data.
        """
        if view.buffer_id != self._buffer_id:
            raise ValueError(
                f"View buffer_id {view.buffer_id} does not match "
                f"this buffer {self._buffer_id}"
            )
        raw = self._data[view.offset : view.offset + view.size]
        return np.frombuffer(raw, dtype=view.dtype).reshape(view.shape)

    def reset(self) -> None:
        """Reset the buffer, clearing all allocated regions."""
        self._offset = 0


class SharedMemoryTransport:
    """Transport layer that uses ZeroCopyBuffer for inter-node data transfer.

    This class provides a high-level API for sending and receiving
    BrainData instances via shared-memory buffers.
    """

    def __init__(self, buffer_capacity: int = 1024 * 1024):
        self._buffers: Dict[str, ZeroCopyBuffer] = {}
        self._buffer_capacity = buffer_capacity

    def _get_or_create_buffer(self, buffer_id: str) -> ZeroCopyBuffer:
        if buffer_id not in self._buffers:
            self._buffers[buffer_id] = ZeroCopyBuffer(
                buffer_id=buffer_id, capacity_bytes=self._buffer_capacity
            )
        return self._buffers[buffer_id]

    def send(self, data: BrainData, buffer_id: str = "default") -> BufferView:
        """Send a BrainData instance into a shared buffer.

        Args:
            data: The data to send.
            buffer_id: Identifier for the target buffer.

        Returns:
            A BufferView describing where the data was written.
        """
        buf = self._get_or_create_buffer(buffer_id)

        if isinstance(data, Tensor):
            arr = data.data
            view = buf.allocate(arr.nbytes, str(arr.dtype), data.shape)
            buf.write(view, arr)
            return view

        elif isinstance(data, StateVector):
            arr = data.data
            view = buf.allocate(arr.nbytes, str(arr.dtype), (data.dimension,))
            buf.write(view, arr)
            return view

        elif isinstance(data, SpikeTrain):
            ts_arr = data.timestamps
            nid_arr = data.neuron_ids
            total_bytes = ts_arr.nbytes + nid_arr.nbytes
            # Store concatenated: timestamps followed by neuron_ids
            view = buf.allocate(total_bytes, str(ts_arr.dtype), (data.spike_count,))
            buf.write(
                BufferView(
                    buffer_id=view.buffer_id,
                    offset=view.offset,
                    size=ts_arr.nbytes,
                    dtype=str(ts_arr.dtype),
                    shape=(data.spike_count,),
                ),
                ts_arr,
            )
            buf.write(
                BufferView(
                    buffer_id=view.buffer_id,
                    offset=view.offset + ts_arr.nbytes,
                    size=nid_arr.nbytes,
                    dtype=str(nid_arr.dtype),
                    shape=(data.spike_count,),
                ),
                nid_arr,
            )
            return view

        elif isinstance(data, Signal):
            arr = np.array([data.value, data.timestamp], dtype=np.float64)
            view = buf.allocate(arr.nbytes, "float64", (2,))
            buf.write(view, arr)
            return view

        elif isinstance(data, Message):
            import json
            json_bytes = json.dumps({
                "topic": data.topic,
                "payload": data.payload,
                "timestamp": data.timestamp,
            }).encode("utf-8")
            arr = np.frombuffer(json_bytes, dtype=np.uint8)
            view = buf.allocate(arr.nbytes, "uint8", (len(arr),))
            buf.write(view, arr)
            return view

        else:
            raise TypeError(f"Unsupported BrainData type: {type(data).__name__}")

    def receive(self, view: BufferView, data_type: str) -> BrainData:
        """Receive a BrainData instance from a shared buffer.

        Args:
            view: BufferView describing where to read from.
            data_type: The type of data to reconstruct
                      ('Tensor', 'StateVector', 'SpikeTrain', 'Signal', 'Message').

        Returns:
            The reconstructed BrainData instance.
        """
        buf = self._get_or_create_buffer(view.buffer_id)

        if data_type == "Tensor":
            arr = buf.read(view)
            return Tensor(data=arr, shape=view.shape)

        elif data_type == "StateVector":
            arr = buf.read(view)
            return StateVector(data=arr, dimension=view.shape[0])

        elif data_type == "SpikeTrain":
            # For SpikeTrain, we need to split timestamps and neuron_ids
            # The view shape contains spike_count
            spike_count = view.shape[0]
            ts_size = spike_count * 8  # float64 = 8 bytes
            nid_size = spike_count * 8  # int64 = 8 bytes
            ts_view = BufferView(
                buffer_id=view.buffer_id,
                offset=view.offset,
                size=ts_size,
                dtype="float64",
                shape=(spike_count,),
            )
            nid_view = BufferView(
                buffer_id=view.buffer_id,
                offset=view.offset + ts_size,
                size=nid_size,
                dtype="int64",
                shape=(spike_count,),
            )
            timestamps = buf.read(ts_view)
            neuron_ids = buf.read(nid_view)
            return SpikeTrain(
                timestamps=timestamps,
                neuron_ids=neuron_ids,
                duration=1.0,  # duration is not stored in buffer, caller should set
            )

        elif data_type == "Signal":
            arr = buf.read(view)
            return Signal(value=float(arr[0]), timestamp=float(arr[1]))

        elif data_type == "Message":
            arr = buf.read(view)
            import json
            json_str = arr.tobytes().decode("utf-8")
            d = json.loads(json_str)
            return Message(
                topic=d["topic"],
                payload=d["payload"],
                timestamp=d["timestamp"],
            )

        else:
            raise TypeError(f"Unknown BrainData type: {data_type}")

    def reset_buffer(self, buffer_id: str) -> None:
        """Reset a specific buffer."""
        if buffer_id in self._buffers:
            self._buffers[buffer_id].reset()

    def reset_all(self) -> None:
        """Reset all buffers."""
        for buf in self._buffers.values():
            buf.reset()
