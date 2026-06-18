"""Tests for communication protocols (Phase 4)."""

import pytest
import numpy as np

from brain_agent.types import BrainData, Tensor, StateVector, SpikeTrain, Signal, Message
from brain_agent.node import (
    BrainNode, NodeConfig, PortSpec, RegionType, Protocol,
)
from brain_agent.protocols import (
    BaseProtocol,
    AsyncPushProtocol,
    SyncCallProtocol,
    PubSubProtocol,
)


# ---------------------------------------------------------------------------
# Concrete test node
# ---------------------------------------------------------------------------

class PassThroughNode(BrainNode):
    """Test node that passes data through unchanged."""
    def process(self, data: BrainData) -> BrainData:
        return data


class DoubleSignalNode(BrainNode):
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


def _make_node(node_id: int, short_name: str = "Node") -> PassThroughNode:
    return PassThroughNode(_make_config(node_id=node_id, short_name=short_name))


# ===========================================================================
# 1. AsyncPushProtocol tests
# ===========================================================================

class TestAsyncPushProtocol:
    """Test ASYNC_PUSH protocol: send/receive, queue ordering, pending_count, stats."""

    def test_protocol_name(self):
        proto = AsyncPushProtocol()
        assert proto.name == "ASYNC_PUSH"

    def test_send_and_receive_single_message(self):
        proto = AsyncPushProtocol()
        source = _make_node(1, "Src")
        target = _make_node(2, "Tgt")

        data = Signal(value=42.0, timestamp=1.0)
        proto.send(source, target, data)

        received = proto.receive(target)
        assert received is not None
        assert isinstance(received, Signal)
        assert received.value == 42.0
        assert received.timestamp == 1.0

    def test_receive_from_empty_queue_returns_none(self):
        proto = AsyncPushProtocol()
        target = _make_node(1, "Empty")
        assert proto.receive(target) is None

    def test_queue_ordering_fifo(self):
        """Messages should be received in FIFO order."""
        proto = AsyncPushProtocol()
        source = _make_node(1, "Src")
        target = _make_node(2, "Tgt")

        for i in range(5):
            proto.send(source, target, Signal(value=float(i), timestamp=0.0))

        for i in range(5):
            received = proto.receive(target)
            assert received.value == float(i)

    def test_multiple_senders_to_same_target(self):
        """Multiple sources can send to the same target."""
        proto = AsyncPushProtocol()
        src1 = _make_node(1, "S1")
        src2 = _make_node(2, "S2")
        target = _make_node(3, "Tgt")

        proto.send(src1, target, Signal(value=10.0))
        proto.send(src2, target, Signal(value=20.0))

        first = proto.receive(target)
        second = proto.receive(target)
        assert first.value == 10.0
        assert second.value == 20.0

    def test_pending_count(self):
        proto = AsyncPushProtocol()
        source = _make_node(1, "Src")
        target = _make_node(2, "Tgt")

        assert proto.pending_count(target.node_id) == 0

        proto.send(source, target, Signal(value=1.0))
        proto.send(source, target, Signal(value=2.0))
        proto.send(source, target, Signal(value=3.0))

        assert proto.pending_count(target.node_id) == 3

        proto.receive(target)
        assert proto.pending_count(target.node_id) == 2

    def test_pending_count_creates_queue_if_missing(self):
        proto = AsyncPushProtocol()
        assert proto.pending_count(999) == 0

    def test_stats_after_messages(self):
        proto = AsyncPushProtocol()
        source = _make_node(1, "Src")
        target = _make_node(2, "Tgt")

        proto.send(source, target, Signal(value=1.0))
        proto.send(source, target, Signal(value=2.0))

        stats = proto.get_stats()
        assert stats["protocol"] == "ASYNC_PUSH"
        assert stats["message_count"] == 2
        assert stats["error_count"] == 0
        assert stats["avg_latency_ms"] > 0.0

    def test_stats_initial(self):
        proto = AsyncPushProtocol()
        stats = proto.get_stats()
        assert stats["message_count"] == 0
        assert stats["avg_latency_ms"] == 0.0

    def test_send_different_data_types(self):
        """Protocol should handle all BrainData types."""
        proto = AsyncPushProtocol()
        source = _make_node(1, "Src")
        target = _make_node(2, "Tgt")

        # Tensor
        proto.send(source, target, Tensor.zeros((2, 2)))
        # Signal
        proto.send(source, target, Signal(value=5.0))
        # Message
        proto.send(source, target, Message(topic="test", payload={"k": "v"}))

        r1 = proto.receive(target)
        assert isinstance(r1, Tensor)
        r2 = proto.receive(target)
        assert isinstance(r2, Signal)
        r3 = proto.receive(target)
        assert isinstance(r3, Message)

    def test_send_to_multiple_targets(self):
        """Same source sends to different targets."""
        proto = AsyncPushProtocol()
        source = _make_node(1, "Src")
        t1 = _make_node(2, "T1")
        t2 = _make_node(3, "T2")

        data = Signal(value=99.0)
        proto.send(source, t1, data)
        proto.send(source, t2, data)

        assert proto.receive(t1).value == 99.0
        assert proto.receive(t2).value == 99.0


# ===========================================================================
# 2. SyncCallProtocol tests
# ===========================================================================

class TestSyncCallProtocol:
    """Test SYNC_CALL protocol: send, call_and_collect, get_last_results, stats."""

    def test_protocol_name(self):
        proto = SyncCallProtocol()
        assert proto.name == "SYNC_CALL"

    def test_send_calls_target_process(self):
        proto = SyncCallProtocol()
        source = _make_node(1, "Src")
        target = DoubleSignalNode(_make_config(node_id=2, short_name="Dbl"))

        data = Signal(value=5.0, timestamp=0.0)
        proto.send(source, target, data)

        # Result should be stored in last_results
        results = proto.get_last_results(source.node_id)
        assert len(results) == 1
        assert isinstance(results[0], Signal)
        assert results[0].value == 10.0  # doubled

    def test_receive_always_returns_none(self):
        proto = SyncCallProtocol()
        node = _make_node(1, "Node")
        assert proto.receive(node) is None

    def test_call_and_collect_single_target(self):
        proto = SyncCallProtocol()
        source = _make_node(1, "Src")
        target = DoubleSignalNode(_make_config(node_id=2, short_name="Dbl"))

        data = Signal(value=3.0)
        results = proto.call_and_collect(source, [target], data)

        assert len(results) == 1
        assert results[0].value == 6.0

    def test_call_and_collect_multiple_targets(self):
        proto = SyncCallProtocol()
        source = _make_node(1, "Src")
        t1 = DoubleSignalNode(_make_config(node_id=2, short_name="D1"))
        t2 = DoubleSignalNode(_make_config(node_id=3, short_name="D2"))
        t3 = DoubleSignalNode(_make_config(node_id=4, short_name="D3"))

        data = Signal(value=7.0)
        results = proto.call_and_collect(source, [t1, t2, t3], data)

        assert len(results) == 3
        for r in results:
            assert r.value == 14.0

    def test_call_and_collect_empty_targets(self):
        proto = SyncCallProtocol()
        source = _make_node(1, "Src")

        data = Signal(value=1.0)
        results = proto.call_and_collect(source, [], data)

        assert results == []

    def test_call_and_collect_stores_last_results(self):
        proto = SyncCallProtocol()
        source = _make_node(1, "Src")
        target = PassThroughNode(_make_config(node_id=2, short_name="PT"))

        data = Signal(value=42.0)
        proto.call_and_collect(source, [target], data)

        last = proto.get_last_results(source.node_id)
        assert len(last) == 1
        assert last[0].value == 42.0

    def test_get_last_results_empty(self):
        proto = SyncCallProtocol()
        assert proto.get_last_results(999) == []

    def test_stats_after_send(self):
        proto = SyncCallProtocol()
        source = _make_node(1, "Src")
        target = _make_node(2, "Tgt")

        proto.send(source, target, Signal(value=1.0))
        proto.send(source, target, Signal(value=2.0))

        stats = proto.get_stats()
        assert stats["protocol"] == "SYNC_CALL"
        assert stats["message_count"] == 2
        assert stats["avg_latency_ms"] > 0.0

    def test_stats_after_call_and_collect(self):
        proto = SyncCallProtocol()
        source = _make_node(1, "Src")
        targets = [_make_node(i, f"T{i}") for i in range(2, 5)]

        proto.call_and_collect(source, targets, Signal(value=1.0))

        stats = proto.get_stats()
        assert stats["message_count"] == 3  # 3 targets

    def test_stats_initial(self):
        proto = SyncCallProtocol()
        stats = proto.get_stats()
        assert stats["message_count"] == 0
        assert stats["avg_latency_ms"] == 0.0

    def test_send_calls_target_process_directly(self):
        """Sending via SYNC_CALL calls target.process() directly (no metric update on target).

        The protocol calls process() directly, bypassing the node's internal
        metric tracking. Metric tracking is the node's own responsibility when
        using its push_async/call_sync/publish methods.
        """
        proto = SyncCallProtocol()
        source = _make_node(1, "Src")
        target = PassThroughNode(_make_config(node_id=2, short_name="Tgt"))

        proto.send(source, target, Signal(value=1.0))
        # process() is called directly by the protocol, so node metrics are NOT updated
        metrics = target.get_metrics()
        assert metrics["process_count"] == 0


# ===========================================================================
# 3. PubSubProtocol tests
# ===========================================================================

class TestPubSubProtocol:
    """Test PUB_SUB protocol: subscribe/unsubscribe, publish to topic, receive, stats."""

    def test_protocol_name(self):
        proto = PubSubProtocol()
        assert proto.name == "PUB_SUB"

    def test_subscribe_and_get_subscribers(self):
        proto = PubSubProtocol()
        node = _make_node(1, "Sub")

        proto.subscribe(node, "dopamine")
        subs = proto.get_subscribers("dopamine")
        assert 1 in subs

    def test_subscribe_multiple_nodes(self):
        proto = PubSubProtocol()
        n1 = _make_node(1, "N1")
        n2 = _make_node(2, "N2")
        n3 = _make_node(3, "N3")

        proto.subscribe(n1, "topic_a")
        proto.subscribe(n2, "topic_a")
        proto.subscribe(n3, "topic_b")

        assert len(proto.get_subscribers("topic_a")) == 2
        assert len(proto.get_subscribers("topic_b")) == 1

    def test_unsubscribe(self):
        proto = PubSubProtocol()
        node = _make_node(1, "Sub")

        proto.subscribe(node, "serotonin")
        assert 1 in proto.get_subscribers("serotonin")

        proto.unsubscribe(node, "serotonin")
        assert 1 not in proto.get_subscribers("serotonin")

    def test_unsubscribe_nonexistent_is_noop(self):
        proto = PubSubProtocol()
        node = _make_node(1, "Sub")
        proto.unsubscribe(node, "nonexistent")  # should not raise

    def test_get_topics(self):
        proto = PubSubProtocol()
        n1 = _make_node(1, "N1")
        n2 = _make_node(2, "N2")

        proto.subscribe(n1, "topic_a")
        proto.subscribe(n2, "topic_b")

        topics = proto.get_topics()
        assert "topic_a" in topics
        assert "topic_b" in topics

    def test_publish_to_subscribers(self):
        proto = PubSubProtocol()
        n1 = _make_node(1, "Sub1")
        n2 = _make_node(2, "Sub2")
        publisher = _make_node(99, "Pub")

        proto.subscribe(n1, "modulation")
        proto.subscribe(n2, "modulation")

        data = Signal(value=0.8, timestamp=1.0)
        count = proto.publish(publisher, "modulation", data)

        assert count == 2

        r1 = proto.receive(n1)
        r2 = proto.receive(n2)
        assert r1.value == 0.8
        assert r2.value == 0.8

    def test_publish_to_empty_topic(self):
        proto = PubSubProtocol()
        publisher = _make_node(1, "Pub")

        count = proto.publish(publisher, "empty_topic", Signal(value=1.0))
        assert count == 0

    def test_publish_multiple_messages(self):
        """Multiple publishes accumulate in subscriber queues."""
        proto = PubSubProtocol()
        sub = _make_node(1, "Sub")
        pub = _make_node(2, "Pub")

        proto.subscribe(sub, "events")

        proto.publish(pub, "events", Signal(value=1.0))
        proto.publish(pub, "events", Signal(value=2.0))
        proto.publish(pub, "events", Signal(value=3.0))

        assert proto.receive(sub).value == 1.0
        assert proto.receive(sub).value == 2.0
        assert proto.receive(sub).value == 3.0
        assert proto.receive(sub) is None

    def test_receive_from_empty_queue(self):
        proto = PubSubProtocol()
        node = _make_node(1, "Node")
        assert proto.receive(node) is None

    def test_direct_send(self):
        """Direct send (used for tree propagation)."""
        proto = PubSubProtocol()
        source = _make_node(1, "Src")
        target = _make_node(2, "Tgt")

        proto.send(source, target, Message(topic="direct", payload={"x": 1}))
        received = proto.receive(target)

        assert received is not None
        assert received.topic == "direct"

    def test_send_to_unsubscribed_node(self):
        """Direct send works even without subscription."""
        proto = PubSubProtocol()
        source = _make_node(1, "Src")
        target = _make_node(2, "Tgt")  # not subscribed to anything

        proto.send(source, target, Signal(value=5.0))
        assert proto.receive(target).value == 5.0

    def test_stats_after_publish(self):
        proto = PubSubProtocol()
        n1 = _make_node(1, "S1")
        n2 = _make_node(2, "S2")
        pub = _make_node(99, "Pub")

        proto.subscribe(n1, "t")
        proto.subscribe(n2, "t")
        proto.publish(pub, "t", Signal(value=1.0))

        stats = proto.get_stats()
        assert stats["protocol"] == "PUB_SUB"
        assert stats["message_count"] == 2
        assert stats["avg_latency_ms"] > 0.0

    def test_stats_after_direct_send(self):
        proto = PubSubProtocol()
        source = _make_node(1, "Src")
        target = _make_node(2, "Tgt")

        proto.send(source, target, Signal(value=1.0))

        stats = proto.get_stats()
        assert stats["message_count"] == 1

    def test_stats_initial(self):
        proto = PubSubProtocol()
        stats = proto.get_stats()
        assert stats["message_count"] == 0
        assert stats["avg_latency_ms"] == 0.0

    def test_multiple_topics_independent(self):
        """Different topics have independent subscriber sets."""
        proto = PubSubProtocol()
        n1 = _make_node(1, "N1")
        n2 = _make_node(2, "N2")
        pub = _make_node(99, "Pub")

        proto.subscribe(n1, "topic_a")
        proto.subscribe(n2, "topic_b")

        proto.publish(pub, "topic_a", Signal(value=10.0))
        proto.publish(pub, "topic_b", Signal(value=20.0))

        assert proto.receive(n1).value == 10.0
        assert proto.receive(n2).value == 20.0
        assert proto.receive(n1) is None
        assert proto.receive(n2) is None


# ===========================================================================
# 4. Integration tests: protocols with actual BrainNode instances
# ===========================================================================

class TestIntegration:
    """Integration: protocols with actual BrainNode instances."""

    def test_async_push_with_tree(self):
        """AsyncPushProtocol with a parent and two children."""
        proto = AsyncPushProtocol()
        parent = PassThroughNode(_make_config(node_id=1, short_name="Parent"))
        child1 = PassThroughNode(_make_config(node_id=2, short_name="Child1"))
        child2 = PassThroughNode(_make_config(node_id=3, short_name="Child2"))

        parent.connect_child(child1)
        parent.connect_child(child2)

        data = Signal(value=100.0)
        # Simulate parent pushing to children via protocol
        proto.send(parent, child1, data)
        proto.send(parent, child2, data)

        r1 = proto.receive(child1)
        r2 = proto.receive(child2)
        assert r1.value == 100.0
        assert r2.value == 100.0

    def test_sync_call_with_tree(self):
        """SyncCallProtocol: parent calls children and collects results."""
        proto = SyncCallProtocol()
        parent = PassThroughNode(_make_config(node_id=1, short_name="Parent"))
        child1 = DoubleSignalNode(_make_config(node_id=2, short_name="C1"))
        child2 = DoubleSignalNode(_make_config(node_id=3, short_name="C2"))

        parent.connect_child(child1)
        parent.connect_child(child2)

        data = Signal(value=5.0)
        targets = list(parent.children.values())
        results = proto.call_and_collect(parent, targets, data)

        assert len(results) == 2
        assert results[0].value == 10.0
        assert results[1].value == 10.0

    def test_pub_sub_with_modulation_tree(self):
        """PubSubProtocol: modulation broadcast to subscribers."""
        proto = PubSubProtocol()
        publisher = PassThroughNode(_make_config(
            node_id=1, short_name="Modulator",
            region_type=RegionType.MODULATION,
        ))
        sub1 = PassThroughNode(_make_config(node_id=2, short_name="Target1"))
        sub2 = PassThroughNode(_make_config(node_id=3, short_name="Target2"))
        sub3 = PassThroughNode(_make_config(node_id=4, short_name="Target3"))

        proto.subscribe(sub1, "dopamine")
        proto.subscribe(sub2, "dopamine")
        proto.subscribe(sub3, "serotonin")

        # Publish dopamine signal
        data = Signal(value=0.9, timestamp=0.5)
        count = proto.publish(publisher, "dopamine", data)

        assert count == 2
        assert proto.receive(sub1).value == 0.9
        assert proto.receive(sub2).value == 0.9
        assert proto.receive(sub3) is None  # not subscribed to dopamine

    def test_mixed_protocols_same_nodes(self):
        """Nodes can participate in multiple protocols simultaneously."""
        async_proto = AsyncPushProtocol()
        sync_proto = SyncCallProtocol()
        pubsub_proto = PubSubProtocol()

        parent = PassThroughNode(_make_config(node_id=1, short_name="Hub"))
        child = DoubleSignalNode(_make_config(node_id=2, short_name="Leaf"))

        # Use async push
        async_proto.send(parent, child, Signal(value=1.0))
        r = async_proto.receive(child)
        assert r.value == 1.0

        # Use sync call
        sync_proto.send(parent, child, Signal(value=3.0))
        results = sync_proto.get_last_results(parent.node_id)
        assert results[-1].value == 6.0  # doubled

        # Use pub/sub
        pubsub_proto.subscribe(child, "alerts")
        pubsub_proto.publish(parent, "alerts", Signal(value=7.0))
        r = pubsub_proto.receive(child)
        assert r.value == 7.0

    def test_deep_tree_async_propagation(self):
        """AsyncPushProtocol through a 3-level tree."""
        proto = AsyncPushProtocol()
        root = PassThroughNode(_make_config(node_id=1, short_name="Root", layer=0))
        mid = PassThroughNode(_make_config(node_id=2, short_name="Mid", layer=1))
        leaf = PassThroughNode(_make_config(node_id=3, short_name="Leaf", layer=2))

        root.connect_child(mid)
        mid.connect_child(leaf)

        # Root -> Mid
        proto.send(root, mid, Signal(value=10.0))
        assert proto.receive(mid).value == 10.0

        # Mid -> Leaf
        proto.send(mid, leaf, Signal(value=20.0))
        assert proto.receive(leaf).value == 20.0

    def test_protocol_stats_independent(self):
        """Each protocol instance tracks its own stats independently."""
        p1 = AsyncPushProtocol()
        p2 = AsyncPushProtocol()
        src = _make_node(1, "S")
        tgt = _make_node(2, "T")

        p1.send(src, tgt, Signal(value=1.0))
        p2.send(src, tgt, Signal(value=2.0))
        p2.send(src, tgt, Signal(value=3.0))

        assert p1.get_stats()["message_count"] == 1
        assert p2.get_stats()["message_count"] == 2


# ===========================================================================
# 5. BaseProtocol tests
# ===========================================================================

class TestBaseProtocol:
    """Test BaseProtocol abstract class."""

    def test_cannot_instantiate_base_protocol(self):
        """BaseProtocol is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseProtocol("test")

    def test_subclass_has_name(self):
        proto = AsyncPushProtocol()
        assert proto.name == "ASYNC_PUSH"

    def test_subclass_has_stats(self):
        proto = SyncCallProtocol()
        stats = proto.get_stats()
        assert "protocol" in stats
        assert "message_count" in stats
        assert "error_count" in stats
        assert "avg_latency_ms" in stats
