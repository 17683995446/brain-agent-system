"""Tests for BrainNode base class (Phase 3)."""

import pytest
import numpy as np

from brain_agent.types import BrainData, Tensor, StateVector, SpikeTrain, Signal, Message
from brain_agent.node import (
    BrainNode, NodeConfig, PortSpec, RegionType, Protocol,
)


# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------

class PassThroughNode(BrainNode):
    """Test node that passes data through unchanged."""
    def process(self, data: BrainData) -> BrainData:
        return data


class DoubleNode(BrainNode):
    """Test node that doubles signal values."""
    def process(self, data: BrainData) -> BrainData:
        if isinstance(data, Signal):
            return Signal(value=data.value * 2, timestamp=data.timestamp)
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    node_id: int = 1,
    name_cn: str = "测试节点",
    short_name: str = "TestNode",
    folder_name: str = "TestNode",
    layer: int = 0,
    parent_id: int = None,
    responsibility: str = "测试用途",
    region_type: RegionType = RegionType.RELAY,
    tensor_shape: str = "[1,64,64]",
    frequency: str = "100Hz",
    protocol: Protocol = Protocol.ASYNC_PUSH,
    algorithm: str = "pass_through",
    children_ids: list = None,
) -> NodeConfig:
    return NodeConfig(
        node_id=node_id,
        name_cn=name_cn,
        short_name=short_name,
        folder_name=folder_name,
        layer=layer,
        parent_id=parent_id,
        responsibility=responsibility,
        region_type=region_type,
        tensor_shape=tensor_shape,
        frequency=frequency,
        protocol=protocol,
        algorithm=algorithm,
        children_ids=children_ids or [],
    )


# ---------------------------------------------------------------------------
# 1. Test BrainNode creation with NodeConfig
# ---------------------------------------------------------------------------

class TestNodeCreation:
    """Test that BrainNode can be instantiated with a valid NodeConfig."""

    def test_create_pass_through_node(self):
        config = _make_config(node_id=1, short_name="PTNode")
        node = PassThroughNode(config)
        assert node.node_id == 1
        assert node.short_name == "PTNode"
        assert node.name_cn == "测试节点"
        assert node.folder_name == "TestNode"
        assert node.layer == 0
        assert node.region_type == RegionType.RELAY
        assert node.protocol == Protocol.ASYNC_PUSH

    def test_create_double_node(self):
        config = _make_config(node_id=2, short_name="DNode", protocol=Protocol.SYNC_CALL)
        node = DoubleNode(config)
        assert node.node_id == 2
        assert node.protocol == Protocol.SYNC_CALL

    def test_cannot_instantiate_abstract_class(self):
        """BrainNode itself is abstract and cannot be instantiated."""
        config = _make_config()
        with pytest.raises(TypeError):
            BrainNode(config)

    def test_default_ports_created(self):
        config = _make_config()
        node = PassThroughNode(config)
        assert "IN" in node._input_ports
        assert "OUT" in node._output_ports
        assert node._input_ports["IN"].data_type == Tensor
        assert node._output_ports["OUT"].data_type == Tensor

    def test_repr(self):
        config = _make_config(node_id=42, short_name="ReprNode", layer=3)
        node = PassThroughNode(config)
        r = repr(node)
        assert "id=42" in r
        assert "ReprNode" in r
        assert "L03" in r


# ---------------------------------------------------------------------------
# 2. Test creating a concrete subclass that implements process()
# ---------------------------------------------------------------------------

class TestProcessMethod:
    """Test that concrete subclasses implement process() correctly."""

    def test_pass_through_returns_same_data(self):
        config = _make_config()
        node = PassThroughNode(config)
        data = Signal(value=5.0, timestamp=1.0)
        result = node.process(data)
        assert result is data  # same object returned
        assert result.value == 5.0

    def test_double_node_doubles_signal(self):
        config = _make_config()
        node = DoubleNode(config)
        data = Signal(value=3.0, timestamp=0.0)
        result = node.process(data)
        assert isinstance(result, Signal)
        assert result.value == 6.0

    def test_double_node_passes_through_non_signal(self):
        config = _make_config()
        node = DoubleNode(config)
        tensor = Tensor.zeros((2, 2))
        result = node.process(tensor)
        assert result is tensor

    def test_process_tensor(self):
        config = _make_config()
        node = PassThroughNode(config)
        tensor = Tensor.random((3, 3))
        result = node.process(tensor)
        assert np.array_equal(result.data, tensor.data)

    def test_process_state_vector(self):
        config = _make_config()
        node = PassThroughNode(config)
        sv = StateVector.zeros(128)
        result = node.process(sv)
        assert result is sv

    def test_process_message(self):
        config = _make_config()
        node = PassThroughNode(config)
        msg = Message(topic="test", payload={"key": "val"})
        result = node.process(msg)
        assert result is msg


# ---------------------------------------------------------------------------
# 3. Test connect_parent / connect_child
# ---------------------------------------------------------------------------

class TestParentChildManagement:
    """Test parent-child relationship management."""

    def test_connect_child_sets_bidirectional_link(self):
        parent_config = _make_config(node_id=1, short_name="Parent")
        child_config = _make_config(node_id=2, short_name="Child")
        parent = PassThroughNode(parent_config)
        child = PassThroughNode(child_config)

        parent.connect_child(child)

        assert child.node_id in parent.children
        assert parent is child.parent

    def test_connect_parent_directly(self):
        parent_config = _make_config(node_id=1, short_name="Parent")
        child_config = _make_config(node_id=2, short_name="Child")
        parent = PassThroughNode(parent_config)
        child = PassThroughNode(child_config)

        child.connect_parent(parent)
        assert child.parent is parent
        # parent's children dict should NOT contain child (connect_parent is one-way)
        assert child.node_id not in parent.children

    def test_disconnect_child(self):
        parent_config = _make_config(node_id=1, short_name="Parent")
        child_config = _make_config(node_id=2, short_name="Child")
        parent = PassThroughNode(parent_config)
        child = PassThroughNode(child_config)

        parent.connect_child(child)
        parent.disconnect_child(child.node_id)

        assert child.node_id not in parent.children
        assert child.parent is None

    def test_disconnect_nonexistent_child_is_noop(self):
        parent_config = _make_config(node_id=1)
        parent = PassThroughNode(parent_config)
        parent.disconnect_child(999)  # should not raise

    def test_connect_multiple_children(self):
        parent_config = _make_config(node_id=1, short_name="Root")
        parent = PassThroughNode(parent_config)

        children = []
        for i in range(5):
            c = PassThroughNode(_make_config(node_id=10 + i, short_name=f"Child{i}"))
            parent.connect_child(c)
            children.append(c)

        assert len(parent.children) == 5
        for c in children:
            assert c.parent is parent


# ---------------------------------------------------------------------------
# 4. Test push_async propagates to children
# ---------------------------------------------------------------------------

class TestPushAsync:
    """Test ASYNC_PUSH protocol."""

    def test_push_async_to_no_children(self):
        config = _make_config()
        node = PassThroughNode(config)
        data = Signal(value=1.0)
        node.push_async(data)  # should not raise
        assert node._process_count == 1

    def test_push_async_propagates_to_children(self):
        parent_config = _make_config(node_id=1, short_name="P")
        parent = PassThroughNode(parent_config)

        child1 = PassThroughNode(_make_config(node_id=2, short_name="C1"))
        child2 = PassThroughNode(_make_config(node_id=3, short_name="C2"))
        parent.connect_child(child1)
        parent.connect_child(child2)

        data = Signal(value=42.0, timestamp=0.0)
        parent.push_async(data)

        # Each child should have received the data in its input queue
        assert len(child1._input_queue) == 1
        assert len(child2._input_queue) == 1
        assert child1._input_queue[0].value == 42.0
        assert child2._input_queue[0].value == 42.0

    def test_push_async_updates_metrics(self):
        config = _make_config()
        node = PassThroughNode(config)
        data = Signal(value=1.0)
        node.push_async(data)
        assert node._process_count == 1
        assert node._last_process_time > 0
        assert node._total_process_time > 0


# ---------------------------------------------------------------------------
# 5. Test call_sync collects results from children
# ---------------------------------------------------------------------------

class TestCallSync:
    """Test SYNC_CALL protocol."""

    def test_call_sync_with_no_children(self):
        config = _make_config()
        node = PassThroughNode(config)
        data = Signal(value=1.0)
        results = node.call_sync(data)
        assert results == []

    def test_call_sync_collects_child_results(self):
        parent_config = _make_config(node_id=1, short_name="P")
        parent = PassThroughNode(parent_config)

        child1 = DoubleNode(_make_config(node_id=2, short_name="C1"))
        child2 = DoubleNode(_make_config(node_id=3, short_name="C2"))
        parent.connect_child(child1)
        parent.connect_child(child2)

        data = Signal(value=5.0, timestamp=0.0)
        results = parent.call_sync(data)

        assert len(results) == 2
        # Each child doubles: 5 -> 10
        assert isinstance(results[0], Signal)
        assert results[0].value == 10.0
        assert isinstance(results[1], Signal)
        assert results[1].value == 10.0

    def test_call_sync_updates_parent_metrics(self):
        parent_config = _make_config(node_id=1, short_name="P")
        parent = PassThroughNode(parent_config)

        child = PassThroughNode(_make_config(node_id=2, short_name="C"))
        parent.connect_child(child)

        data = Signal(value=1.0)
        parent.call_sync(data)

        assert parent._process_count == 1


# ---------------------------------------------------------------------------
# 6. Test publish broadcasts to children
# ---------------------------------------------------------------------------

class TestPublish:
    """Test PUB_SUB protocol."""

    def test_publish_to_no_children(self):
        config = _make_config()
        node = PassThroughNode(config)
        data = Signal(value=1.0)
        node.publish(data)  # should not raise
        assert node._process_count == 1

    def test_publish_broadcasts_to_children(self):
        parent_config = _make_config(node_id=1, short_name="P")
        parent = PassThroughNode(parent_config)

        child1 = PassThroughNode(_make_config(node_id=2, short_name="C1"))
        child2 = PassThroughNode(_make_config(node_id=3, short_name="C2"))
        child3 = PassThroughNode(_make_config(node_id=4, short_name="C3"))
        parent.connect_child(child1)
        parent.connect_child(child2)
        parent.connect_child(child3)

        msg = Message(topic="modulation", payload={"dopamine": 0.8})
        parent.publish(msg)

        assert len(child1._input_queue) == 1
        assert len(child2._input_queue) == 1
        assert len(child3._input_queue) == 1
        assert child1._input_queue[0].topic == "modulation"

    def test_publish_updates_metrics(self):
        config = _make_config()
        node = PassThroughNode(config)
        node.publish(Signal(value=1.0))
        assert node._process_count == 1


# ---------------------------------------------------------------------------
# 7. Test metrics tracking
# ---------------------------------------------------------------------------

class TestMetrics:
    """Test processing metrics."""

    def test_initial_metrics(self):
        config = _make_config(node_id=10, short_name="M")
        node = PassThroughNode(config)
        metrics = node.get_metrics()

        assert metrics["node_id"] == 10
        assert metrics["short_name"] == "M"
        assert metrics["layer"] == 0
        assert metrics["region_type"] == "relay"
        assert metrics["process_count"] == 0
        assert metrics["last_process_time_ms"] == 0.0
        assert metrics["avg_process_time_ms"] == 0.0
        assert metrics["queue_depth"] == 0
        assert metrics["children_count"] == 0
        assert metrics["is_leaf"] is True

    def test_metrics_after_processing(self):
        config = _make_config(node_id=5, short_name="Proc")
        node = PassThroughNode(config)
        data = Signal(value=1.0)
        node.push_async(data)

        metrics = node.get_metrics()
        assert metrics["process_count"] == 1
        assert metrics["last_process_time_ms"] > 0
        assert metrics["avg_process_time_ms"] > 0

    def test_avg_process_time_accumulates(self):
        config = _make_config()
        node = PassThroughNode(config)
        for _ in range(10):
            node.push_async(Signal(value=1.0))

        metrics = node.get_metrics()
        assert metrics["process_count"] == 10
        assert metrics["avg_process_time_ms"] > 0

    def test_queue_depth_increases_on_receive(self):
        config = _make_config()
        node = PassThroughNode(config)
        node.receive(Signal(value=1.0))
        node.receive(Signal(value=2.0))
        node.receive(Signal(value=3.0))

        metrics = node.get_metrics()
        assert metrics["queue_depth"] == 3


# ---------------------------------------------------------------------------
# 8. Test reset functionality
# ---------------------------------------------------------------------------

class TestReset:
    """Test node reset."""

    def test_reset_clears_state(self):
        config = _make_config()
        node = PassThroughNode(config)
        node.set_state("key1", "value1")
        node.set_state("key2", 42)

        node.reset()

        assert node.get_state("key1") is None
        assert node.get_state("key2") is None

    def test_reset_clears_input_queue(self):
        config = _make_config()
        node = PassThroughNode(config)
        node.receive(Signal(value=1.0))
        node.receive(Signal(value=2.0))
        assert len(node._input_queue) == 2

        node.reset()
        assert len(node._input_queue) == 0

    def test_reset_clears_metrics(self):
        config = _make_config()
        node = PassThroughNode(config)
        node.push_async(Signal(value=1.0))
        node.push_async(Signal(value=2.0))
        assert node._process_count == 2

        node.reset()

        metrics = node.get_metrics()
        assert metrics["process_count"] == 0
        assert metrics["last_process_time_ms"] == 0.0
        assert metrics["avg_process_time_ms"] == 0.0

    def test_reset_preserves_connections(self):
        parent_config = _make_config(node_id=1, short_name="P")
        child_config = _make_config(node_id=2, short_name="C")
        parent = PassThroughNode(parent_config)
        child = PassThroughNode(child_config)
        parent.connect_child(child)

        parent.reset()
        child.reset()

        assert child.node_id in parent.children
        assert child.parent is parent


# ---------------------------------------------------------------------------
# 9. Test is_leaf property
# ---------------------------------------------------------------------------

class TestIsLeaf:
    """Test is_leaf property."""

    def test_new_node_is_leaf(self):
        config = _make_config()
        node = PassThroughNode(config)
        assert node.is_leaf is True

    def test_node_with_children_is_not_leaf(self):
        parent_config = _make_config(node_id=1, short_name="P")
        parent = PassThroughNode(parent_config)
        child = PassThroughNode(_make_config(node_id=2, short_name="C"))
        parent.connect_child(child)

        assert parent.is_leaf is False

    def test_child_is_leaf(self):
        parent_config = _make_config(node_id=1, short_name="P")
        parent = PassThroughNode(parent_config)
        child = PassThroughNode(_make_config(node_id=2, short_name="C"))
        parent.connect_child(child)

        assert child.is_leaf is True

    def test_becomes_leaf_after_disconnect(self):
        parent_config = _make_config(node_id=1, short_name="P")
        parent = PassThroughNode(parent_config)
        child = PassThroughNode(_make_config(node_id=2, short_name="C"))
        parent.connect_child(child)
        assert parent.is_leaf is False

        parent.disconnect_child(child.node_id)
        assert parent.is_leaf is True


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

class TestStateManagement:
    """Test internal state get/set."""

    def test_set_and_get_state(self):
        config = _make_config()
        node = PassThroughNode(config)
        node.set_state("threshold", 0.5)
        assert node.get_state("threshold") == 0.5

    def test_get_nonexistent_state_returns_default(self):
        config = _make_config()
        node = PassThroughNode(config)
        assert node.get_state("missing") is None
        assert node.get_state("missing", "fallback") == "fallback"

    def test_state_overwrite(self):
        config = _make_config()
        node = PassThroughNode(config)
        node.set_state("x", 1)
        node.set_state("x", 2)
        assert node.get_state("x") == 2


class TestPortSpec:
    """Test PortSpec dataclass."""

    def test_port_spec_creation(self):
        port = PortSpec(
            port_id="IN",
            name="输入信号",
            data_type=Tensor,
            shape=(1, 64, 64),
            frequency="100Hz",
            protocol=Protocol.ASYNC_PUSH,
            required=True,
        )
        assert port.port_id == "IN"
        assert port.data_type == Tensor
        assert port.shape == (1, 64, 64)
        assert port.protocol == Protocol.ASYNC_PUSH
        assert port.required is True


class TestNodeConfig:
    """Test NodeConfig dataclass."""

    def test_node_config_defaults(self):
        config = _make_config()
        assert config.children_ids == []

    def test_node_config_with_children(self):
        config = _make_config(children_ids=[2, 3])
        assert config.children_ids == [2, 3]

    def test_node_config_all_fields(self):
        config = _make_config(
            node_id=100,
            name_cn="视觉皮层",
            short_name="VisualCortex",
            folder_name="VisualCortex",
            layer=5,
            parent_id=50,
            responsibility="视觉处理",
            region_type=RegionType.SENSORY,
            tensor_shape="[1,100,100]",
            frequency="60Hz",
            protocol=Protocol.ASYNC_PUSH,
            algorithm="conv2d",
        )
        assert config.node_id == 100
        assert config.name_cn == "视觉皮层"
        assert config.region_type == RegionType.SENSORY
        assert config.algorithm == "conv2d"


class TestReceiveMethod:
    """Test the receive() method."""

    def test_receive_appends_to_queue(self):
        config = _make_config()
        node = PassThroughNode(config)
        node.receive(Signal(value=1.0))
        node.receive(Signal(value=2.0))
        node.receive(Message(topic="t"))

        assert len(node._input_queue) == 3
        assert node._input_queue[0].value == 1.0
        assert node._input_queue[1].value == 2.0
        assert node._input_queue[2].topic == "t"


class TestRegionTypeAndProtocolEnums:
    """Test enum values."""

    def test_region_type_values(self):
        assert RegionType.SENSORY.value == "sensory"
        assert RegionType.MOTOR.value == "motor"
        assert RegionType.MEMORY.value == "memory"
        assert RegionType.MODULATION.value == "modulation"
        assert RegionType.RELAY.value == "relay"
        assert RegionType.UNKNOWN.value == "unknown"

    def test_protocol_values(self):
        assert Protocol.ASYNC_PUSH.value == "ASYNC_PUSH"
        assert Protocol.SYNC_CALL.value == "SYNC_CALL"
        assert Protocol.PUB_SUB.value == "PUB_SUB"
