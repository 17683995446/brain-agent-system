"""Tests for error handling hardening.

Tests node queue limits, safe_process graceful degradation, and
backpressure behavior in the runtime node implementation.
"""
import logging
import pytest

from brain_agent.types import Signal
from brain_agent.node import BrainNode, NodeConfig, RegionType, Protocol, DEFAULT_MAX_QUEUE_SIZE


class _DummyNode(BrainNode):
    """Minimal node for testing."""

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


class TestNodeQueueLimits:
    """Test that node input queues have bounded sizes."""

    def test_input_queue_has_max_size(self):
        """Node input queue should have maxlen set."""
        node = _make_node(0)
        assert node._input_queue.maxlen == DEFAULT_MAX_QUEUE_SIZE

    def test_input_queue_drops_oldest_when_full(self):
        """When queue is full, oldest items are dropped (deque maxlen behavior)."""
        node = _make_node(0)

        for i in range(DEFAULT_MAX_QUEUE_SIZE + 10):
            node.receive(Signal(value=float(i), timestamp=float(i)))

        assert len(node._input_queue) == DEFAULT_MAX_QUEUE_SIZE

    def test_input_queue_preserves_latest(self):
        """When queue is full, latest items are preserved."""
        node = _make_node(0)

        for i in range(DEFAULT_MAX_QUEUE_SIZE + 5):
            node.receive(Signal(value=float(i), timestamp=float(i)))

        last_item = node._input_queue[-1]
        assert last_item.value == float(DEFAULT_MAX_QUEUE_SIZE + 4)


class TestSafeProcessErrorRecording:
    """Test that safe_process records errors for scheduler capture."""

    def test_safe_process_records_error(self):
        """safe_process should record error in _last_process_error."""
        node = _make_node(0)

        def failing_process(data):
            raise RuntimeError("Test error")

        node.process = failing_process
        node.safe_process(Signal(value=1.0, timestamp=0.0))

        assert node._last_process_error is not None
        assert "Node 0" in node._last_process_error
        assert "Test error" in node._last_process_error

    def test_safe_process_clears_error_on_success(self):
        """safe_process should clear _last_process_error on success."""
        node = _make_node(0)

        def failing_process(data):
            raise RuntimeError("Test error")

        node.process = failing_process
        node.safe_process(Signal(value=1.0, timestamp=0.0))
        assert node._last_process_error is not None

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
        assert result is not None

    def test_safe_process_does_not_raise(self):
        """safe_process should never raise, even on catastrophic errors."""
        node = _make_node(0)

        def catastrophic_process(data):
            raise MemoryError("Out of memory")

        node.process = catastrophic_process
        result = node.safe_process(Signal(value=1.0, timestamp=0.0))
        assert result is not None
        assert node._last_process_error is not None


class TestNodeBackpressure:
    """Test node backpressure via receive()."""

    def test_receive_adds_to_queue(self):
        """receive() should add data to input queue."""
        node = _make_node(0)
        node.receive(Signal(value=1.0, timestamp=0.0))
        assert len(node._input_queue) == 1

    def test_receive_logs_warning_when_full(self, caplog):
        """receive() should log warning when queue is full."""
        node = _make_node(0)

        for i in range(DEFAULT_MAX_QUEUE_SIZE):
            node.receive(Signal(value=float(i), timestamp=float(i)))

        with caplog.at_level(logging.WARNING):
            node.receive(Signal(value=999.0, timestamp=0.0))

        assert any(
            "queue full" in r.message.lower() or "dropping" in r.message.lower()
            for r in caplog.records
        )
