"""Tests for error handling hardening (P1-10).

Tests protocol queue limits, MemoryError handling, and graceful degradation.
"""
import pytest
from collections import deque

from brain_agent.protocols.async_push import AsyncPushProtocol, MAX_PROTOCOL_QUEUE_SIZE as ASYNC_MAX
from brain_agent.protocols.pub_sub import PubSubProtocol, MAX_PROTOCOL_QUEUE_SIZE as PUBSUB_MAX
from brain_agent.protocols.sync_call import SyncCallProtocol
from brain_agent.types import Signal, Tensor
from brain_agent.node import BrainNode, NodeConfig, RegionType, Protocol


class _DummyNode(BrainNode):
    """Minimal node for protocol testing."""

    def process(self, data):
        return data

    def _setup_ports(self):
        pass


def _make_node(node_id: int = 0) -> _DummyNode:
    config = NodeConfig(
        node_id=node_id,
        name_cn=f"测试节点{node_id}",
        short_name=f"Test{node_id}",
        folder_name=f"test_{node_id}",
        layer=0,
        parent_id=None,
        responsibility="test",
        region_type=RegionType.SENSORY,
        tensor_shape="(1,1,8,8)",
        frequency="100",
        protocol=Protocol.ASYNC_PUSH,
        algorithm="test",
    )
    return _DummyNode(config)


class TestProtocolQueueLimits:
    """Test that protocol queues have bounded sizes."""

    def test_async_push_queue_has_max_size(self):
        """AsyncPush queue should have maxlen set."""
        proto = AsyncPushProtocol()
        source = _make_node(0)
        target = _make_node(1)
        proto.send(source, target, Signal(value=1.0, timestamp=0.0))
        assert proto._queues[1].maxlen == ASYNC_MAX

    def test_pub_sub_queue_has_max_size(self):
        """PubSub queue should have maxlen set."""
        proto = PubSubProtocol()
        source = _make_node(0)
        target = _make_node(1)
        proto.send(source, target, Signal(value=1.0, timestamp=0.0))
        assert proto._queues[1].maxlen == PUBSUB_MAX

    def test_async_push_drops_oldest_when_full(self):
        """When queue is full, oldest items are dropped (deque maxlen behavior)."""
        proto = AsyncPushProtocol()
        source = _make_node(0)
        target = _make_node(1)

        # Fill queue beyond capacity
        for i in range(ASYNC_MAX + 10):
            proto.send(source, target, Signal(value=float(i), timestamp=float(i)))

        # Queue should not exceed maxlen
        assert len(proto._queues[1]) == ASYNC_MAX

    def test_pub_sub_drops_oldest_when_full(self):
        """When queue is full, oldest items are dropped (deque maxlen behavior)."""
        proto = PubSubProtocol()
        source = _make_node(0)
        target = _make_node(1)

        for i in range(PUBSUB_MAX + 10):
            proto.send(source, target, Signal(value=float(i), timestamp=float(i)))

        assert len(proto._queues[1]) == PUBSUB_MAX


class TestProtocolMemoryError:
    """Test MemoryError handling in protocols."""

    def test_async_push_memory_error_increments_error_count(self):
        """MemoryError in AsyncPush should increment error_count."""
        proto = AsyncPushProtocol()
        source = _make_node(0)
        target = _make_node(1)

        # Replace queue with a mock that raises MemoryError on append
        class FailingQueue:
            def __init__(self):
                self.maxlen = ASYNC_MAX

            def append(self, item):
                raise MemoryError("test")

            def __len__(self):
                return 0

        proto._queues[target.node_id] = FailingQueue()

        with pytest.raises(MemoryError):
            proto.send(source, target, Signal(value=1.0, timestamp=0.0))

        assert proto._error_count == 1

    def test_pub_sub_memory_error_increments_error_count(self):
        """MemoryError in PubSub should increment error_count."""
        proto = PubSubProtocol()
        source = _make_node(0)
        target = _make_node(1)

        # Replace queue with a mock that raises MemoryError on append
        class FailingQueue:
            def __init__(self):
                self.maxlen = PUBSUB_MAX

            def append(self, item):
                raise MemoryError("test")

            def __len__(self):
                return 0

        proto._queues[target.node_id] = FailingQueue()

        with pytest.raises(MemoryError):
            proto.send(source, target, Signal(value=2.0, timestamp=1.0))

        assert proto._error_count == 1

    def test_sync_call_memory_error_increments_error_count(self):
        """MemoryError in SyncCall should increment error_count."""
        proto = SyncCallProtocol()
        source = _make_node(0)
        target = _make_node(1)

        # Mock process to raise MemoryError
        original_process = target.process
        target.process = lambda data: (_ for _ in ()).throw(MemoryError("test"))

        with pytest.raises(MemoryError):
            proto.send(source, target, Signal(value=1.0, timestamp=0.0))

        assert proto._error_count == 1
        target.process = original_process


class TestSafeProcessErrorRecording:
    """Test that safe_process records errors for scheduler capture."""

    def test_safe_process_records_error(self):
        """safe_process should record error in _last_process_error."""
        node = _make_node(0)

        def failing_process(data):
            raise RuntimeError("Test error")

        node.process = failing_process
        result = node.safe_process(Signal(value=1.0, timestamp=0.0))

        assert node._last_process_error is not None
        assert "Node 0" in node._last_process_error
        assert "Test error" in node._last_process_error

    def test_safe_process_clears_error_on_success(self):
        """safe_process should clear _last_process_error on success."""
        node = _make_node(0)

        # First, cause an error
        def failing_process(data):
            raise RuntimeError("Test error")

        node.process = failing_process
        node.safe_process(Signal(value=1.0, timestamp=0.0))
        assert node._last_process_error is not None

        # Then, succeed
        node.process = lambda data: data
        node.safe_process(Signal(value=1.0, timestamp=0.0))
        assert node._last_process_error is None

    def test_safe_process_returns_default_on_error(self):
        """safe_process should return default output on process() failure."""
        node = _make_node(0)

        def failing_process(data):
            raise RuntimeError("Test error")

        node.process = failing_process
        result = node.safe_process(Signal(value=1.0, timestamp=0.0))

        # Should return some BrainData (default output)
        assert result is not None
