"""Tests for zero-copy transport layer."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from brain_agent.transport import BufferView, ZeroCopyBuffer, SharedMemoryTransport
from brain_agent.types import Tensor, StateVector, SpikeTrain, Signal, Message


class TestBufferView:
    """Tests for BufferView dataclass."""

    def test_creation(self):
        """BufferView should store all attributes correctly."""
        view = BufferView(buffer_id="buf1", offset=100, size=400, dtype="float32", shape=(10, 10))
        assert view.buffer_id == "buf1"
        assert view.offset == 100
        assert view.size == 400
        assert view.dtype == "float32"
        assert view.shape == (10, 10)

    def test_to_dict(self):
        """to_dict should produce a serializable dict."""
        view = BufferView(buffer_id="buf1", offset=0, size=16, dtype="float64", shape=(2,))
        d = view.to_dict()
        assert d == {
            "buffer_id": "buf1",
            "offset": 0,
            "size": 16,
            "dtype": "float64",
            "shape": [2],
        }

    def test_from_dict(self):
        """from_dict should reconstruct an identical BufferView."""
        d = {
            "buffer_id": "buf2",
            "offset": 64,
            "size": 128,
            "dtype": "int32",
            "shape": [32],
        }
        view = BufferView.from_dict(d)
        assert view.buffer_id == "buf2"
        assert view.offset == 64
        assert view.size == 128
        assert view.dtype == "int32"
        assert view.shape == (32,)

    def test_round_trip_dict(self):
        """to_dict -> from_dict should produce an identical BufferView."""
        original = BufferView(buffer_id="x", offset=10, size=20, dtype="float32", shape=(5,))
        d = original.to_dict()
        restored = BufferView.from_dict(d)
        assert restored == original


class TestZeroCopyBuffer:
    """Tests for ZeroCopyBuffer."""

    def test_creation(self):
        """ZeroCopyBuffer should initialize with correct capacity."""
        buf = ZeroCopyBuffer("test", capacity_bytes=1024)
        assert buf.buffer_id == "test"
        assert buf.capacity == 1024
        assert buf.free_space == 1024

    def test_allocate_reduces_free_space(self):
        """allocate should reduce free_space by the allocated size."""
        buf = ZeroCopyBuffer("test", capacity_bytes=1024)
        view = buf.allocate(100, "float32", (25,))
        assert view.size == 100
        assert buf.free_space == 924

    def test_write_and_read(self):
        """write followed by read should return the original data."""
        buf = ZeroCopyBuffer("test", capacity_bytes=1024)
        arr = np.arange(10, dtype=np.float32)
        view = buf.allocate(arr.nbytes, "float32", (10,))
        buf.write(view, arr)
        result = buf.read(view)
        assert np.array_equal(result, arr)

    def test_read_returns_view(self):
        """read should return a view into the buffer (zero-copy)."""
        buf = ZeroCopyBuffer("test", capacity_bytes=1024)
        arr = np.ones(5, dtype=np.float64)
        view = buf.allocate(arr.nbytes, "float64", (5,))
        buf.write(view, arr)
        result = buf.read(view)
        # Modifying result should affect the buffer if it's a true view
        result[0] = 99.0
        re_read = buf.read(view)
        assert re_read[0] == 99.0

    def test_allocate_out_of_space(self):
        """allocate should raise MemoryError when out of space."""
        buf = ZeroCopyBuffer("test", capacity_bytes=100)
        buf.allocate(80, "float32", (20,))
        with pytest.raises(MemoryError):
            buf.allocate(30, "float32", (7,))

    def test_write_wrong_buffer_id(self):
        """write should raise ValueError for mismatched buffer_id."""
        buf = ZeroCopyBuffer("buf1", capacity_bytes=1024)
        view = BufferView(buffer_id="buf2", offset=0, size=10, dtype="float32", shape=(2,))
        with pytest.raises(ValueError, match="does not match"):
            buf.write(view, np.zeros(2, dtype=np.float32))

    def test_read_wrong_buffer_id(self):
        """read should raise ValueError for mismatched buffer_id."""
        buf = ZeroCopyBuffer("buf1", capacity_bytes=1024)
        view = BufferView(buffer_id="buf2", offset=0, size=10, dtype="float32", shape=(2,))
        with pytest.raises(ValueError, match="does not match"):
            buf.read(view)

    def test_reset(self):
        """reset should restore free_space to capacity."""
        buf = ZeroCopyBuffer("test", capacity_bytes=1024)
        buf.allocate(500, "float32", (125,))
        assert buf.free_space == 524
        buf.reset()
        assert buf.free_space == 1024


class TestSharedMemoryTransport:
    """Tests for SharedMemoryTransport."""

    def test_send_receive_tensor(self):
        """send/receive should round-trip Tensor correctly."""
        transport = SharedMemoryTransport(buffer_capacity=1024 * 1024)
        original = Tensor.random((4, 5))
        view = transport.send(original, buffer_id="t1")
        received = transport.receive(view, "Tensor")
        assert isinstance(received, Tensor)
        assert received.shape == original.shape
        assert np.allclose(received.data, original.data)

    def test_send_receive_state_vector(self):
        """send/receive should round-trip StateVector correctly."""
        transport = SharedMemoryTransport(buffer_capacity=1024 * 1024)
        original = StateVector.random(64)
        view = transport.send(original, buffer_id="sv1")
        received = transport.receive(view, "StateVector")
        assert isinstance(received, StateVector)
        assert received.dimension == original.dimension
        assert np.allclose(received.data, original.data)

    def test_send_receive_signal(self):
        """send/receive should round-trip Signal correctly."""
        transport = SharedMemoryTransport(buffer_capacity=1024 * 1024)
        original = Signal(value=-0.75, timestamp=123.45)
        view = transport.send(original, buffer_id="sig1")
        received = transport.receive(view, "Signal")
        assert isinstance(received, Signal)
        assert received.value == original.value
        assert received.timestamp == original.timestamp

    def test_send_receive_message(self):
        """send/receive should round-trip Message correctly."""
        transport = SharedMemoryTransport(buffer_capacity=1024 * 1024)
        original = Message(topic="cmd", payload={"action": "reset"}, timestamp=1.0)
        view = transport.send(original, buffer_id="msg1")
        received = transport.receive(view, "Message")
        assert isinstance(received, Message)
        assert received.topic == original.topic
        assert received.payload == original.payload
        assert received.timestamp == original.timestamp

    def test_send_receive_spike_train(self):
        """send/receive should round-trip SpikeTrain correctly."""
        transport = SharedMemoryTransport(buffer_capacity=1024 * 1024)
        original = SpikeTrain(
            timestamps=np.array([0.1, 0.3, 0.5], dtype=np.float64),
            neuron_ids=np.array([0, 5, 3], dtype=np.int64),
            duration=1.0,
        )
        view = transport.send(original, buffer_id="st1")
        received = transport.receive(view, "SpikeTrain")
        assert isinstance(received, SpikeTrain)
        assert received.spike_count == original.spike_count
        assert np.allclose(received.timestamps, original.timestamps)
        assert np.array_equal(received.neuron_ids, original.neuron_ids)

    def test_multiple_buffers(self):
        """Transport should support multiple independent buffers."""
        transport = SharedMemoryTransport(buffer_capacity=1024)
        t1 = Tensor.zeros((10,))
        t2 = Tensor(data=np.ones((10,), dtype=np.float32), shape=(10,))
        view1 = transport.send(t1, buffer_id="buf_a")
        view2 = transport.send(t2, buffer_id="buf_b")
        received1 = transport.receive(view1, "Tensor")
        received2 = transport.receive(view2, "Tensor")
        assert np.allclose(received1.data, t1.data)
        assert np.allclose(received2.data, t2.data)

    def test_reset_buffer(self):
        """reset_buffer should clear a specific buffer."""
        transport = SharedMemoryTransport(buffer_capacity=1024)
        t = Tensor.zeros((100,))
        transport.send(t, buffer_id="buf_x")
        transport.reset_buffer("buf_x")
        # After reset, we should be able to send again without overflow
        transport.send(t, buffer_id="buf_x")

    def test_reset_all(self):
        """reset_all should clear all buffers."""
        transport = SharedMemoryTransport(buffer_capacity=1024)
        t = Tensor.zeros((100,))
        transport.send(t, buffer_id="buf_a")
        transport.send(t, buffer_id="buf_b")
        transport.reset_all()
        transport.send(t, buffer_id="buf_a")
        transport.send(t, buffer_id="buf_b")

    def test_unsupported_send_type(self):
        """send should raise TypeError for unsupported types."""
        transport = SharedMemoryTransport()
        with pytest.raises(TypeError, match="Unsupported BrainData type"):
            transport.send("not a brain data")

    def test_unknown_receive_type(self):
        """receive should raise TypeError for unknown data types."""
        transport = SharedMemoryTransport()
        view = BufferView(buffer_id="default", offset=0, size=10, dtype="float32", shape=(2,))
        with pytest.raises(TypeError, match="Unknown BrainData type"):
            transport.receive(view, "UnknownType")
