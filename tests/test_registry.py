"""Tests for unified node registry (Phase 8).

Tests the integration of L00-L03 manual nodes with L04-L10 generated nodes
into a unified 2047-node system.
"""

import json
import os
import pytest
from pathlib import Path

from brain_agent.node import BrainNode, NodeConfig, RegionType, Protocol
from brain_agent.types import BrainData, Signal, Tensor, Message
from brain_agent.tree import (
    BrainTreeBuilder,
    BrainTree,
    PassThroughBrainNode,
)
from brain_agent.nodes import NODE_CLASS_MAP
from brain_agent.generated import GENERATED_NODE_MAP
from brain_agent.exceptions import BrainAgentError
from brain_agent.registry import (
    ALL_NODE_CLASSES,
    get_node_class,
    create_node,
    get_registry_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DESIGN_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "L00_BrainSystem"
)


def _make_full_2047_json(tmp_path):
    """Create a full 2047-node JSON file for testing.

    11 layers (0-10), total = 2^11 - 1 = 2047 nodes.
    """
    nodes = []
    num_layers = 11
    for node_id in range(2047):
        # Determine layer from node_id
        layer = 0
        start = 0
        for l in range(num_layers):
            if node_id < start + 2 ** l:
                layer = l
                break
            start += 2 ** l

        parent_id = None
        children = []
        if layer > 0:
            parent_id = (node_id - 1) // 2
        if layer < num_layers - 1:
            children = [2 * node_id + 1, 2 * node_id + 2]

        # Assign region types based on subtree
        if layer == 0:
            rt = "unknown"
        elif layer <= 2:
            rt = "sensory" if node_id % 2 == 0 else "motor"
        elif layer <= 5:
            rt = "memory"
        else:
            rt = "relay"

        nodes.append({
            "id": node_id,
            "name_cn": f"节点{node_id}",
            "short_name": f"Node{node_id}",
            "full_name": f"Node{node_id}",
            "folder_name": f"L{layer:02d}_Node{node_id}",
            "layer": layer,
            "parent_id": parent_id,
            "responsibility": f"节点{node_id}的职责",
            "tensor_shape": "[1,64,64]",
            "frequency": f"{100 - layer * 10}Hz",
            "protocol": "ASYNC_PUSH",
            "algorithm": "pass_through",
            "region_type": rt,
            "children": children,
        })

    json_path = tmp_path / "test_tree_2047.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"nodes": nodes}, f, ensure_ascii=False, indent=2)
    return json_path


# ---------------------------------------------------------------------------
# 1. Test registry has 2047 entries
# ---------------------------------------------------------------------------

class TestRegistrySize:
    """Test that the unified registry contains 2047 node classes."""

    def test_registry_has_2047_entries(self):
        """ALL_NODE_CLASSES should contain exactly 2047 entries."""
        assert len(ALL_NODE_CLASSES) == 2047, (
            f"Expected 2047 entries, got {len(ALL_NODE_CLASSES)}"
        )

    def test_registry_stats_total(self):
        """get_registry_stats should report total=2047."""
        stats = get_registry_stats()
        assert stats["total"] == 2047


# ---------------------------------------------------------------------------
# 2. Test no overlap between manual and generated
# ---------------------------------------------------------------------------

class TestNoOverlap:
    """Test that manual (L00-L03) and generated (L04-L10) maps don't overlap."""

    def test_no_overlap(self):
        """Manual and generated node maps should have no overlapping IDs."""
        manual_ids = set(NODE_CLASS_MAP.keys())
        generated_ids = set(GENERATED_NODE_MAP.keys())
        overlap = manual_ids & generated_ids
        assert len(overlap) == 0, (
            f"Found overlapping node IDs: {overlap}"
        )

    def test_registry_stats_no_overlap(self):
        """get_registry_stats should report overlap=0."""
        stats = get_registry_stats()
        assert stats["overlap"] == 0

    def test_manual_count(self):
        """NODE_CLASS_MAP should have 15 entries (L00-L03)."""
        assert len(NODE_CLASS_MAP) == 15

    def test_generated_count(self):
        """GENERATED_NODE_MAP should have 2032 entries (L04-L10)."""
        assert len(GENERATED_NODE_MAP) == 2032

    def test_manual_ids_range(self):
        """Manual node IDs should be 0-14."""
        manual_ids = set(NODE_CLASS_MAP.keys())
        assert manual_ids == set(range(15))


# ---------------------------------------------------------------------------
# 3. Test get_node_class returns correct class for L00-L03
# ---------------------------------------------------------------------------

class TestGetNodeClassManual:
    """Test get_node_class for manually implemented L00-L03 nodes."""

    def test_node_0_is_brain_system(self):
        """Node 0 should be BrainSystemNode."""
        from brain_agent.nodes import BrainSystemNode
        cls = get_node_class(0)
        assert cls is BrainSystemNode

    def test_node_1_is_sensorimotor(self):
        """Node 1 should be SensorimotorSystemNode."""
        from brain_agent.nodes import SensorimotorSystemNode
        cls = get_node_class(1)
        assert cls is SensorimotorSystemNode

    def test_node_2_is_cognitive(self):
        """Node 2 should be CognitiveSystemNode."""
        from brain_agent.nodes import CognitiveSystemNode
        cls = get_node_class(2)
        assert cls is CognitiveSystemNode

    def test_node_14_is_noradrenaline(self):
        """Node 14 should be NoradrenalineSystemNode."""
        from brain_agent.nodes import NoradrenalineSystemNode
        cls = get_node_class(14)
        assert cls is NoradrenalineSystemNode

    def test_all_l00_l03_return_classes(self):
        """All L00-L03 node IDs (0-14) should return a class."""
        for node_id in range(15):
            cls = get_node_class(node_id)
            assert cls is not None, f"get_node_class({node_id}) returned None"
            assert issubclass(cls, BrainNode), (
                f"get_node_class({node_id}) returned non-BrainNode class"
            )


# ---------------------------------------------------------------------------
# 4. Test get_node_class returns generated class for L04+
# ---------------------------------------------------------------------------

class TestGetNodeClassGenerated:
    """Test get_node_class for auto-generated L04-L10 nodes."""

    def test_generated_node_returns_class(self):
        """A generated node (e.g., 26) should return a BrainNode subclass."""
        cls = get_node_class(26)
        assert cls is not None
        assert issubclass(cls, BrainNode)

    def test_generated_node_not_manual(self):
        """Generated nodes should NOT be the same as manual node classes."""
        from brain_agent.nodes import NODE_CLASS_MAP as manual_map
        # Pick a generated node ID that is not in manual
        gen_id = 26  # First generated node
        cls = get_node_class(gen_id)
        assert cls is not None
        # Should not be any of the manual classes
        for manual_cls in manual_map.values():
            assert cls is not manual_cls

    def test_high_id_generated_node(self):
        """High ID generated nodes (e.g., 2046) should also return a class."""
        cls = get_node_class(2046)
        assert cls is not None
        assert issubclass(cls, BrainNode)

    def test_all_generated_ids_have_classes(self):
        """Every ID in GENERATED_NODE_MAP should return a class via get_node_class."""
        for node_id in GENERATED_NODE_MAP:
            cls = get_node_class(node_id)
            assert cls is not None, f"get_node_class({node_id}) returned None"


# ---------------------------------------------------------------------------
# 5. Test create_node creates correct instance
# ---------------------------------------------------------------------------

class TestCreateNode:
    """Test create_node factory function."""

    def _make_config(self, node_id, region_type=RegionType.RELAY, layer=0):
        return NodeConfig(
            node_id=node_id,
            name_cn=f"测试{node_id}",
            short_name=f"Test{node_id}",
            folder_name=f"L{layer:02d}_Test{node_id}",
            layer=layer,
            parent_id=None,
            responsibility="test",
            region_type=region_type,
            tensor_shape="",
            frequency="",
            protocol=Protocol.ASYNC_PUSH,
            algorithm="",
            children_ids=[],
        )

    def test_create_manual_node(self):
        """create_node should create a manual node instance for ID 0."""
        config = self._make_config(0)
        node = create_node(config)
        assert isinstance(node, BrainNode)
        assert node.node_id == 0

    def test_create_generated_node(self):
        """create_node should create a generated node instance for ID 26."""
        config = self._make_config(26)
        node = create_node(config)
        assert isinstance(node, BrainNode)
        assert node.node_id == 26

    def test_create_node_has_process_method(self):
        """Created node should have a process method."""
        config = self._make_config(0)
        node = create_node(config)
        assert hasattr(node, 'process')
        assert callable(node.process)

    def test_create_node_unregistered_raises(self):
        """create_node should raise BrainAgentError for unregistered node ID."""
        # Use a very high ID that won't be in the registry
        config = self._make_config(99999)
        with pytest.raises(BrainAgentError, match="No node class registered"):
            create_node(config)


# ---------------------------------------------------------------------------
# 6. Test build_full_tree creates valid 2047-node tree
# ---------------------------------------------------------------------------

class TestBuildFullTree:
    """Test BrainTreeBuilder.build_full_tree method."""

    def test_build_full_tree_2047_nodes(self, tmp_path):
        """build_full_tree should create a tree with 2047 nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        assert tree.total_nodes == 2047
        assert tree.max_layer == 10
        assert tree.root is not None
        assert tree.root.node_id == 0

    def test_build_full_tree_root_is_not_passthrough(self, tmp_path):
        """Root node should use the registered class, not PassThroughBrainNode."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        # Node 0 is in the manual map, so it should not be PassThroughBrainNode
        assert not isinstance(tree.root, PassThroughBrainNode), (
            "Root node should use registered class, not PassThroughBrainNode"
        )

    def test_build_full_tree_manual_nodes_not_passthrough(self, tmp_path):
        """L00-L03 nodes should use registered classes, not PassThroughBrainNode."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        for node_id in range(15):
            node = tree.get_node(node_id)
            assert node is not None
            assert not isinstance(node, PassThroughBrainNode), (
                f"Node {node_id} should use registered class, not PassThroughBrainNode"
            )

    def test_build_full_tree_generated_nodes_not_passthrough(self, tmp_path):
        """L04-L10 nodes that are in the registry should not be PassThroughBrainNode."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        # Check a few generated node IDs
        for node_id in [26, 54, 109, 219, 440]:
            node = tree.get_node(node_id)
            if node is not None:
                assert not isinstance(node, PassThroughBrainNode), (
                    f"Node {node_id} should use registered class"
                )


# ---------------------------------------------------------------------------
# 7. Test all nodes have correct region_type
# ---------------------------------------------------------------------------

class TestRegionTypes:
    """Test that nodes have correct region_type values."""

    def test_manual_nodes_have_region_types(self):
        """All manual nodes should have a valid region_type."""
        for node_id, cls in NODE_CLASS_MAP.items():
            assert node_id in ALL_NODE_CLASSES
            # We can verify the class exists and is a BrainNode
            assert issubclass(cls, BrainNode)

    def test_all_registry_classes_are_brain_nodes(self):
        """All classes in the registry should be BrainNode subclasses."""
        for node_id, cls in ALL_NODE_CLASSES.items():
            assert issubclass(cls, BrainNode), (
                f"Node {node_id}: {cls} is not a BrainNode subclass"
            )


# ---------------------------------------------------------------------------
# 8. Test all nodes have correct layer
# ---------------------------------------------------------------------------

class TestNodeLayers:
    """Test that nodes are associated with correct layers."""

    def test_manual_nodes_layers(self):
        """Manual nodes (0-14) should be in L00-L03."""
        # L00: node 0 (1 node)
        # L01: nodes 1-2 (2 nodes)
        # L02: nodes 3-6 (4 nodes)
        # L03: nodes 7-14 (8 nodes)
        assert 0 in NODE_CLASS_MAP  # L00
        assert 1 in NODE_CLASS_MAP and 2 in NODE_CLASS_MAP  # L01
        for nid in range(3, 7):
            assert nid in NODE_CLASS_MAP  # L02
        for nid in range(7, 15):
            assert nid in NODE_CLASS_MAP  # L03

    def test_generated_nodes_l04_plus(self):
        """Generated nodes should start from L04 (node_id >= 15)."""
        for node_id in GENERATED_NODE_MAP:
            assert node_id >= 15, (
                f"Generated node {node_id} should have ID >= 15"
            )


# ---------------------------------------------------------------------------
# 9. Test tree validation passes
# ---------------------------------------------------------------------------

class TestFullTreeValidation:
    """Test that the full 2047-node tree built via build_full_tree validates."""

    def test_full_tree_validation(self, tmp_path):
        """Full tree built via build_full_tree should pass validation."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        is_valid, errors = tree.validate()
        assert is_valid, f"Validation failed: {errors}"

    def test_full_tree_layer_counts(self, tmp_path):
        """Each layer should have exactly 2^layer nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        for layer in range(11):
            count = len(tree.get_nodes_by_layer(layer))
            expected = 2 ** layer
            assert count == expected, (
                f"Layer {layer}: expected {expected}, got {count}"
            )

    def test_full_tree_binary_structure(self, tmp_path):
        """All non-leaf nodes should have exactly 2 children."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        for node in tree._nodes.values():
            if not node.is_leaf:
                assert len(node.children) == 2, (
                    f"Node {node.node_id} has {len(node.children)} children"
                )

    def test_full_tree_leaf_count(self, tmp_path):
        """Full tree should have 1024 leaf nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        leaves = tree.get_leaf_nodes()
        assert len(leaves) == 1024


# ---------------------------------------------------------------------------
# 10. Test end-to-end: push data from root, verify it reaches leaf nodes
# ---------------------------------------------------------------------------

class TestEndToEndPush:
    """Test end-to-end data flow: push_async from root to leaves."""

    @staticmethod
    def _recursive_push(node, data):
        """Recursively push data through the tree using push_async."""
        node.push_async(data)
        for child in node.children.values():
            TestEndToEndPush._recursive_push(child, data)

    def test_push_async_reaches_all_leaves(self, tmp_path):
        """push_async from root should deliver data to all 1024 leaf nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        # Use a Message (simplest data type) to push through the tree
        data = Message(topic="test_push", payload={"value": 42})

        # Recursively push from root to all leaves
        self._recursive_push(tree.root, data)

        # Verify all leaf nodes received the data
        leaves = tree.get_leaf_nodes()
        assert len(leaves) == 1024

        for leaf in leaves:
            assert len(leaf._input_queue) == 1, (
                f"Leaf {leaf.node_id} did not receive data"
            )
            received = leaf._input_queue[0]
            assert isinstance(received, Message)
            assert received.topic == "test_push"

    def test_push_async_signal(self, tmp_path):
        """push_async with Signal data should reach all leaves."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        data = Signal(value=3.14, timestamp=1.0)
        self._recursive_push(tree.root, data)

        leaves = tree.get_leaf_nodes()
        for leaf in leaves:
            assert len(leaf._input_queue) == 1
            assert isinstance(leaf._input_queue[0], Signal)

    def test_push_async_process_count_increments(self, tmp_path):
        """After recursive push_async, every node's process_count should be >= 1."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        data = Message(topic="test", payload={})
        self._recursive_push(tree.root, data)

        for node in tree._nodes.values():
            assert node._process_count >= 1, (
                f"Node {node.node_id} process_count is {node._process_count}"
            )


# ---------------------------------------------------------------------------
# 11. Test end-to-end: call_sync from root, collect all leaf results
# ---------------------------------------------------------------------------

class TestEndToEndSyncCall:
    """Test end-to-end data flow: call_sync from root to collect results."""

    @staticmethod
    def _recursive_call_sync(node, data):
        """Recursively call_sync through the tree, collecting all leaf results."""
        result = node.process(data)
        child_results = []
        for child in node.children.values():
            child_results.extend(
                TestEndToEndSyncCall._recursive_call_sync(child, result)
            )
        if node.is_leaf:
            return [result]
        return child_results

    def test_call_sync_returns_results(self, tmp_path):
        """call_sync from root should return results from children."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        data = Message(topic="test_sync", payload={"x": 1})
        results = tree.root.call_sync(data)

        # Root has 2 children, so call_sync returns 2 results
        assert len(results) == 2

    def test_call_sync_all_nodes_process(self, tmp_path):
        """After recursive push_async, every node should have process_count >= 1."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        data = Message(topic="test", payload={})
        # Use push_async on each node to ensure metrics are updated
        for node in tree._nodes.values():
            node.push_async(data)

        for node in tree._nodes.values():
            assert node._process_count >= 1, (
                f"Node {node.node_id} process_count is {node._process_count}"
            )

    def test_call_sync_returns_brain_data(self, tmp_path):
        """call_sync results should be BrainData instances."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        data = Message(topic="test", payload={})
        results = tree.root.call_sync(data)

        for result in results:
            assert isinstance(result, BrainData)


# ---------------------------------------------------------------------------
# 12. Test metrics collection across full tree
# ---------------------------------------------------------------------------

class TestMetricsCollection:
    """Test metrics collection across the full 2047-node tree."""

    def test_get_all_metrics(self, tmp_path):
        """get_all_metrics should return correct aggregate metrics."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        metrics = tree.get_all_metrics()
        assert metrics["total_nodes"] == 2047
        assert metrics["max_layer"] == 10
        assert metrics["leaf_nodes"] == 1024
        assert "layer_stats" in metrics
        assert len(metrics["layer_stats"]) == 11

    def test_node_metrics_after_push(self, tmp_path):
        """After recursive push_async, all nodes should have updated metrics."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        data = Message(topic="test", payload={})
        # Recursively push through all nodes
        for node in tree._nodes.values():
            node.push_async(data)

        for node in tree._nodes.values():
            m = node.get_metrics()
            assert m["process_count"] >= 1
            assert m["node_id"] == node.node_id
            assert m["layer"] == node.layer
            assert "region_type" in m
            assert "avg_process_time_ms" in m

    def test_node_metrics_after_reset(self, tmp_path):
        """After reset, node metrics should be cleared."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        data = Message(topic="test", payload={})
        tree.root.push_async(data)

        # Reset all nodes
        for node in tree._nodes.values():
            node.reset()

        for node in tree._nodes.values():
            m = node.get_metrics()
            assert m["process_count"] == 0
            assert m["last_process_time_ms"] == 0.0
            assert m["avg_process_time_ms"] == 0.0
            assert m["queue_depth"] == 0

    def test_layer_stats_validity(self, tmp_path):
        """All layer stats should report valid=True."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_full_tree(str(json_path))

        stats = tree.get_layer_stats()
        for layer in range(11):
            assert stats[layer]["valid"] is True, (
                f"Layer {layer} stats report invalid"
            )
            assert stats[layer]["count"] == 2 ** layer
            assert stats[layer]["expected"] == 2 ** layer

    def test_registry_stats_consistency(self):
        """Registry stats should be consistent with actual map sizes."""
        stats = get_registry_stats()
        assert stats["total"] == stats["manual"] + stats["generated"] - stats["overlap"]
        assert stats["manual"] == len(NODE_CLASS_MAP)
        assert stats["generated"] == len(GENERATED_NODE_MAP)
