"""Tests for binary tree builder (Phase 5).

Tests DesignDocParser, BrainTreeBuilder, and BrainTree classes.
"""

import json
import os
import pytest
from pathlib import Path

from brain_agent.node import BrainNode, NodeConfig, RegionType, Protocol
from brain_agent.types import BrainData, Signal, Tensor
from brain_agent.tree import (
    DesignDocParser,
    BrainTreeBuilder,
    BrainTree,
    PassThroughBrainNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Path to the DESIGN.md root directory
DESIGN_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "L00_BrainSystem"
)


def _make_small_tree_json(tmp_path, num_layers=4):
    """Create a small JSON file with a complete binary tree of num_layers layers.

    Layer 0: 1 node, Layer 1: 2 nodes, ..., Layer num_layers-1: 2^(num_layers-1) nodes
    Total nodes = 2^num_layers - 1
    """
    nodes = []
    node_id = 0
    for layer in range(num_layers):
        nodes_in_layer = 2 ** layer
        for i in range(nodes_in_layer):
            parent_id = None
            children = []
            if layer > 0:
                parent_id = (node_id - 1) // 2
            if layer < num_layers - 1:
                children = [2 * node_id + 1, 2 * node_id + 2]

            nodes.append({
                "id": node_id,
                "name_cn": f"节点{node_id}",
                "short_name": f"Node{node_id}",
                "full_name": f"Node{node_id}",
                "folder_name": f"L{layer:02d}_Node{node_id}",
                "layer": layer,
                "parent_id": parent_id,
                "responsibility": f"节点{node_id}的职责",
                "tensor_shape": "[1,64,64]" if layer < 3 else "[1,32,32]",
                "frequency": f"{100 - layer * 10}Hz",
                "protocol": "ASYNC_PUSH",
                "algorithm": "pass_through",
                "region_type": "sensory" if layer < 2 else "relay",
                "children": children,
            })
            node_id += 1

    json_path = tmp_path / "test_tree.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"nodes": nodes}, f, ensure_ascii=False, indent=2)
    return json_path


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
# 1. Test DesignDocParser with actual DESIGN.md files
# ---------------------------------------------------------------------------

class TestDesignDocParser:
    """Test DesignDocParser parsing of DESIGN.md files."""

    def test_parse_root_design_md(self):
        """Parse the root DESIGN.md (L00_BrainSystem)."""
        filepath = os.path.join(DESIGN_ROOT, "DESIGN.md")
        if not os.path.exists(filepath):
            pytest.skip("DESIGN.md not found at expected path")

        metadata = DesignDocParser.parse(filepath)
        assert metadata["node_id"] == 0
        assert metadata["folder_name"] == "L00_BrainSystem"
        assert metadata["layer"] == 0
        assert metadata["region_type"] == RegionType.UNKNOWN
        assert metadata["protocol"] == Protocol.PUB_SUB

    def test_parse_layer1_design_md(self):
        """Parse a layer 1 DESIGN.md."""
        filepath = os.path.join(
            DESIGN_ROOT,
            "L01_Brain--SensorimotorSystem",
            "DESIGN.md"
        )
        if not os.path.exists(filepath):
            pytest.skip("DESIGN.md not found at expected path")

        metadata = DesignDocParser.parse(filepath)
        assert metadata["node_id"] == 1
        assert metadata["folder_name"] == "L01_Brain--SensorimotorSystem"
        assert metadata["layer"] == 1
        assert metadata["region_type"] == RegionType.SENSORY
        assert metadata["protocol"] == Protocol.ASYNC_PUSH

    def test_parse_layer2_design_md(self):
        """Parse a layer 2 DESIGN.md."""
        filepath = os.path.join(
            DESIGN_ROOT,
            "L01_Brain--SensorimotorSystem",
            "L02_Sensorimotor--SensoryPathway",
            "DESIGN.md"
        )
        if not os.path.exists(filepath):
            pytest.skip("DESIGN.md not found at expected path")

        metadata = DesignDocParser.parse(filepath)
        assert metadata["node_id"] == 3
        assert metadata["layer"] == 2
        assert metadata["region_type"] == RegionType.SENSORY

    def test_parse_leaf_node_design_md(self):
        """Parse a leaf node DESIGN.md (layer 10)."""
        # Find a leaf node DESIGN.md
        leaf_path = os.path.join(
            DESIGN_ROOT,
            "L01_Brain--SensorimotorSystem",
            "L02_Sensorimotor--MotorPathway",
            "L03_Motor--SomatosensoryPathway",
            "L04_Somato--OnChannelDetector",
            "L05_OnChannel--HighFreqFilter",
            "L06_HighFreq--CenterReceptor",
            "L07_CenterRF--FastPathway",
            "L08_FastPath--MagnocellularUnit",
            "L09_MagnoCell--OrientationSelector",
            "L10_OrientSel--LuminanceEncoder",
            "DESIGN.md"
        )
        if not os.path.exists(leaf_path):
            pytest.skip("Leaf DESIGN.md not found at expected path")

        metadata = DesignDocParser.parse(leaf_path)
        assert metadata["node_id"] == 1280
        assert metadata["layer"] == 10
        assert metadata["is_leaf"] is True
        assert metadata["children_count"] == 0

    def test_extract_field(self):
        """Test _extract_field with known content."""
        content = "- **节点ID**: 42\n- **层级**: 第3层 (L03)\n"
        assert DesignDocParser._extract_field(content, "节点ID") == "42"
        assert DesignDocParser._extract_field(content, "层级") == "第3层 (L03)"

    def test_extract_section(self):
        """Test _extract_section with known content."""
        content = "## 职责功能\n处理感觉输入\n\n## 接口规范\n"
        result = DesignDocParser._extract_section(content, "职责功能")
        assert "处理感觉输入" in result

    def test_parse_layer(self):
        """Test _parse_layer with various formats."""
        assert DesignDocParser._parse_layer("第0层 (L00)") == 0
        assert DesignDocParser._parse_layer("第1层 (L01)") == 1
        assert DesignDocParser._parse_layer("第10层 (L10)") == 10

    def test_parse_region_type(self):
        """Test _parse_region_type mapping."""
        assert DesignDocParser._parse_region_type("sensory") == RegionType.SENSORY
        assert DesignDocParser._parse_region_type("motor") == RegionType.MOTOR
        assert DesignDocParser._parse_region_type("memory") == RegionType.MEMORY
        assert DesignDocParser._parse_region_type("modulation") == RegionType.MODULATION
        assert DesignDocParser._parse_region_type("relay") == RegionType.RELAY
        assert DesignDocParser._parse_region_type("unknown") == RegionType.UNKNOWN
        assert DesignDocParser._parse_region_type("invalid") == RegionType.UNKNOWN

    def test_derive_short_name(self):
        """Test _derive_short_name."""
        assert DesignDocParser._derive_short_name("L00_BrainSystem", 0) == "BrainSystem"
        assert DesignDocParser._derive_short_name("L01_Brain--SensorimotorSystem", 1) == "SensorimotorSystem"
        assert DesignDocParser._derive_short_name("L02_Sensorimotor--SensoryPathway", 2) == "SensoryPathway"


# ---------------------------------------------------------------------------
# 2. Test BrainTreeBuilder.build_from_json
# ---------------------------------------------------------------------------

class TestBrainTreeBuilderFromJson:
    """Test building tree from JSON data."""

    def test_build_small_tree(self, tmp_path):
        """Build a small 15-node tree (4 layers)."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        assert tree.total_nodes == 15  # 2^4 - 1
        assert tree.max_layer == 3
        assert tree.root is not None
        assert tree.root.node_id == 0

    def test_build_full_2047_node_tree(self, tmp_path):
        """Build the full 2047-node tree."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        assert tree.total_nodes == 2047
        assert tree.max_layer == 10
        assert tree.root is not None
        assert tree.root.node_id == 0

    def test_build_tree_nodes_have_correct_layers(self, tmp_path):
        """Verify each node is assigned to the correct layer."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        for layer in range(11):
            nodes = tree.get_nodes_by_layer(layer)
            assert len(nodes) == 2 ** layer, (
                f"Layer {layer}: expected {2 ** layer} nodes, got {len(nodes)}"
            )
            for node in nodes:
                assert node.layer == layer

    def test_build_tree_parent_child_links(self, tmp_path):
        """Verify parent-child links are correctly established."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        # Root (id=0) should have 2 children: 1, 2
        assert len(tree.root.children) == 2
        assert 1 in tree.root.children
        assert 2 in tree.root.children

        # Node 1 should have children 3, 4
        node1 = tree.get_node(1)
        assert len(node1.children) == 2
        assert 3 in node1.children
        assert 4 in node1.children

        # Node 3 should have children 7, 8
        node3 = tree.get_node(3)
        assert len(node3.children) == 2
        assert 7 in node3.children
        assert 8 in node3.children

    def test_build_tree_child_parent_backlink(self, tmp_path):
        """Verify child -> parent backlinks are correct."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        # Node 1's parent should be node 0
        node1 = tree.get_node(1)
        assert node1.parent is tree.root

        # Node 3's parent should be node 1
        node3 = tree.get_node(3)
        assert node3.parent is node1

    def test_build_tree_leaf_nodes(self, tmp_path):
        """Verify leaf nodes have no children."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        leaves = tree.get_leaf_nodes()
        # Layer 3 has 8 nodes, all should be leaves
        assert len(leaves) == 8
        for leaf in leaves:
            assert leaf.is_leaf is True
            assert len(leaf.children) == 0

    def test_build_tree_with_custom_node_class(self, tmp_path):
        """Build tree with a custom node class."""
        class CustomNode(BrainNode):
            def process(self, data: BrainData) -> BrainData:
                return data

        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path), node_class=CustomNode)

        assert tree.total_nodes == 7
        for node in tree._nodes.values():
            assert isinstance(node, CustomNode)


# ---------------------------------------------------------------------------
# 3. Test tree validation
# ---------------------------------------------------------------------------

class TestTreeValidation:
    """Test tree structure validation."""

    def test_full_tree_validation_passes(self, tmp_path):
        """Full 2047-node tree should pass validation."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        is_valid, errors = tree.validate()
        assert is_valid, f"Validation failed: {errors}"

    def test_small_tree_validation_passes(self, tmp_path):
        """Small 15-node tree should pass its own validation."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        # For a 4-layer tree, we expect 15 nodes and max layer 3
        # The validate() method checks for 2047 nodes, so it will fail
        # Instead, let's check the structural properties manually
        assert tree.total_nodes == 15
        assert tree.max_layer == 3

        # Check each layer has correct count
        for layer in range(4):
            count = len(tree.get_nodes_by_layer(layer))
            expected = 2 ** layer
            assert count == expected

        # Check binary tree property
        for node in tree._nodes.values():
            if not node.is_leaf:
                assert len(node.children) == 2

    def test_validation_detects_wrong_node_count(self, tmp_path):
        """Validation should detect incorrect node count."""
        # Create a tree with only 1 node
        nodes = [{"id": 0, "name_cn": "Root", "short_name": "Root",
                   "folder_name": "L00_Root", "layer": 0, "parent_id": None,
                   "responsibility": "root", "tensor_shape": "",
                   "frequency": "", "protocol": "ASYNC_PUSH",
                   "algorithm": "", "region_type": "unknown", "children": []}]
        json_path = tmp_path / "tiny.json"
        with open(json_path, 'w') as f:
            json.dump({"nodes": nodes}, f)

        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        is_valid, errors = tree.validate()
        assert not is_valid
        assert any("2047" in e for e in errors)

    def test_validate_binary_structure(self, tmp_path):
        """Verify all non-leaf nodes have exactly 2 children."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        for node in tree._nodes.values():
            if not node.is_leaf:
                assert len(node.children) == 2, (
                    f"Node {node.node_id} has {len(node.children)} children"
                )

    def test_validate_layer_counts(self, tmp_path):
        """Verify each layer has exactly 2^layer nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        for layer in range(11):
            count = len(tree.get_nodes_by_layer(layer))
            expected = 2 ** layer
            assert count == expected, (
                f"Layer {layer}: expected {expected}, got {count}"
            )


# ---------------------------------------------------------------------------
# 4. Test traversal methods
# ---------------------------------------------------------------------------

class TestTraversal:
    """Test BFS and DFS traversal methods."""

    def test_bfs_traverse_small_tree(self, tmp_path):
        """BFS should visit nodes level by level."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        bfs_order = list(tree.bfs_traverse())
        ids = [n.node_id for n in bfs_order]

        # BFS order: 0, 1, 2, 3, 4, 5, 6
        assert ids == [0, 1, 2, 3, 4, 5, 6]

    def test_bfs_traverse_full_tree(self, tmp_path):
        """BFS should visit all 2047 nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        bfs_order = list(tree.bfs_traverse())
        assert len(bfs_order) == 2047

        # First node should be root
        assert bfs_order[0].node_id == 0

        # Layer 0 first, then layer 1, etc.
        for i in range(len(bfs_order) - 1):
            assert bfs_order[i].layer <= bfs_order[i + 1].layer

    def test_dfs_traverse_small_tree(self, tmp_path):
        """DFS should visit in pre-order (root, left, right)."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        dfs_order = list(tree.dfs_traverse())
        ids = [n.node_id for n in dfs_order]

        # Pre-order DFS: 0, 1, 3, 4, 2, 5, 6
        # (reversed children means left child pushed last, popped first)
        assert ids == [0, 1, 3, 4, 2, 5, 6]

    def test_dfs_traverse_full_tree(self, tmp_path):
        """DFS should visit all 2047 nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        dfs_order = list(tree.dfs_traverse())
        assert len(dfs_order) == 2047

        # First node should be root
        assert dfs_order[0].node_id == 0

        # All nodes should be visited exactly once
        visited_ids = set(n.node_id for n in dfs_order)
        assert len(visited_ids) == 2047

    def test_bfs_and_dfs_visit_same_nodes(self, tmp_path):
        """BFS and DFS should visit the same set of nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        bfs_ids = set(n.node_id for n in tree.bfs_traverse())
        dfs_ids = set(n.node_id for n in tree.dfs_traverse())

        assert bfs_ids == dfs_ids
        assert len(bfs_ids) == 2047

    def test_traverse_empty_tree(self):
        """Traversal of empty tree should yield nothing."""
        tree = BrainTree({})
        assert list(tree.bfs_traverse()) == []
        assert list(tree.dfs_traverse()) == []


# ---------------------------------------------------------------------------
# 5. Test get_path_to_root
# ---------------------------------------------------------------------------

class TestPathToRoot:
    """Test get_path_to_root method."""

    def test_root_path_to_root(self, tmp_path):
        """Root node's path to root is just itself."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        path = tree.get_path_to_root(0)
        assert len(path) == 1
        assert path[0].node_id == 0

    def test_child_path_to_root(self, tmp_path):
        """Child node's path goes from child up to root."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        path = tree.get_path_to_root(5)
        ids = [n.node_id for n in path]
        # 5 -> 2 -> 0
        assert ids == [5, 2, 0]

    def test_deep_leaf_path_to_root(self, tmp_path):
        """Deep leaf node's path has correct length."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        # Node 14 is at layer 3: 14 -> 6 -> 2 -> 0
        path = tree.get_path_to_root(14)
        ids = [n.node_id for n in path]
        assert ids == [14, 6, 2, 0]
        assert len(path) == 4  # layer 3 + root

    def test_nonexistent_node_path(self, tmp_path):
        """Path for nonexistent node should be empty."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        path = tree.get_path_to_root(999)
        assert path == []

    def test_path_ends_at_root(self, tmp_path):
        """Every path should end at the root node."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        # Check a few random nodes
        for node_id in [0, 1, 42, 500, 1023, 2046]:
            path = tree.get_path_to_root(node_id)
            assert path[-1].node_id == 0


# ---------------------------------------------------------------------------
# 6. Test get_nodes_by_layer and get_nodes_by_type
# ---------------------------------------------------------------------------

class TestNodeQueries:
    """Test node query methods."""

    def test_get_nodes_by_layer(self, tmp_path):
        """get_nodes_by_layer returns correct nodes."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        layer0 = tree.get_nodes_by_layer(0)
        assert len(layer0) == 1
        assert layer0[0].node_id == 0

        layer1 = tree.get_nodes_by_layer(1)
        assert len(layer1) == 2
        assert set(n.node_id for n in layer1) == {1, 2}

        layer3 = tree.get_nodes_by_layer(3)
        assert len(layer3) == 8

    def test_get_nodes_by_layer_full_tree(self, tmp_path):
        """get_nodes_by_layer for full 2047-node tree."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        for layer in range(11):
            nodes = tree.get_nodes_by_layer(layer)
            assert len(nodes) == 2 ** layer
            for node in nodes:
                assert node.layer == layer

    def test_get_nodes_by_type(self, tmp_path):
        """get_nodes_by_type returns nodes of the correct type."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        sensory_nodes = tree.get_nodes_by_type(RegionType.SENSORY)
        for node in sensory_nodes:
            assert node.region_type == RegionType.SENSORY

        motor_nodes = tree.get_nodes_by_type(RegionType.MOTOR)
        for node in motor_nodes:
            assert node.region_type == RegionType.MOTOR

    def test_get_nodes_by_type_nonexistent(self, tmp_path):
        """get_nodes_by_type returns empty list for type with no nodes."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        # No modulation nodes in the test tree
        mod_nodes = tree.get_nodes_by_type(RegionType.MODULATION)
        assert mod_nodes == []

    def test_get_node(self, tmp_path):
        """get_node returns the correct node or None."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        node = tree.get_node(0)
        assert node is not None
        assert node.node_id == 0

        node = tree.get_node(6)
        assert node is not None
        assert node.node_id == 6

        node = tree.get_node(999)
        assert node is None

    def test_get_leaf_nodes(self, tmp_path):
        """get_leaf_nodes returns only leaf nodes."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        leaves = tree.get_leaf_nodes()
        assert len(leaves) == 8  # Layer 3 has 8 leaves
        for leaf in leaves:
            assert leaf.is_leaf is True

    def test_get_leaf_nodes_full_tree(self, tmp_path):
        """Full 2047-node tree should have 1024 leaf nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        leaves = tree.get_leaf_nodes()
        assert len(leaves) == 1024


# ---------------------------------------------------------------------------
# 7. Test get_layer_stats
# ---------------------------------------------------------------------------

class TestLayerStats:
    """Test get_layer_stats method."""

    def test_layer_stats_small_tree(self, tmp_path):
        """get_layer_stats returns correct stats for small tree."""
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        stats = tree.get_layer_stats()
        assert len(stats) == 4  # layers 0-3

        for layer in range(4):
            assert stats[layer]["count"] == 2 ** layer
            assert stats[layer]["expected"] == 2 ** layer
            assert stats[layer]["valid"] is True
            assert "types" in stats[layer]

    def test_layer_stats_full_tree(self, tmp_path):
        """get_layer_stats returns correct stats for full 2047-node tree."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        stats = tree.get_layer_stats()
        assert len(stats) == 11  # layers 0-10

        for layer in range(11):
            assert stats[layer]["count"] == 2 ** layer
            assert stats[layer]["valid"] is True

    def test_layer_stats_types(self, tmp_path):
        """Layer stats should include type breakdown."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        stats = tree.get_layer_stats()
        for layer in range(11):
            types = stats[layer]["types"]
            # Sum of type counts should equal total count
            assert sum(types.values()) == 2 ** layer


# ---------------------------------------------------------------------------
# 8. Test BrainTree properties and get_all_metrics
# ---------------------------------------------------------------------------

class TestBrainTreeProperties:
    """Test BrainTree container properties."""

    def test_root_property(self, tmp_path):
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        assert tree.root is not None
        assert tree.root.node_id == 0

    def test_root_property_empty_tree(self):
        tree = BrainTree({})
        assert tree.root is None

    def test_total_nodes(self, tmp_path):
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        assert tree.total_nodes == 7

    def test_max_layer(self, tmp_path):
        json_path = _make_small_tree_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        assert tree.max_layer == 3

    def test_get_all_metrics(self, tmp_path):
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        metrics = tree.get_all_metrics()
        assert metrics["total_nodes"] == 2047
        assert metrics["max_layer"] == 10
        assert metrics["leaf_nodes"] == 1024
        assert "layer_stats" in metrics
        assert len(metrics["layer_stats"]) == 11


# ---------------------------------------------------------------------------
# 9. Test PassThroughBrainNode
# ---------------------------------------------------------------------------

class TestPassThroughBrainNode:
    """Test the PassThroughBrainNode default implementation."""

    def test_pass_through_returns_same_data(self):
        config = NodeConfig(
            node_id=0, name_cn="测试", short_name="Test",
            folder_name="Test", layer=0, parent_id=None,
            responsibility="", region_type=RegionType.RELAY,
            tensor_shape="", frequency="", protocol=Protocol.ASYNC_PUSH,
            algorithm="", children_ids=[],
        )
        node = PassThroughBrainNode(config)
        data = Signal(value=42.0)
        result = node.process(data)
        assert result is data
        assert result.value == 42.0

    def test_pass_through_tensor(self):
        config = NodeConfig(
            node_id=0, name_cn="测试", short_name="Test",
            folder_name="Test", layer=0, parent_id=None,
            responsibility="", region_type=RegionType.RELAY,
            tensor_shape="", frequency="", protocol=Protocol.ASYNC_PUSH,
            algorithm="", children_ids=[],
        )
        node = PassThroughBrainNode(config)
        tensor = Tensor.zeros((3, 3))
        result = node.process(tensor)
        assert result is tensor


# ---------------------------------------------------------------------------
# 10. Test build_from_filesystem with actual DESIGN.md files
# ---------------------------------------------------------------------------

class TestBuildFromFilesystem:
    """Test building tree from filesystem DESIGN.md files."""

    def test_build_from_filesystem_creates_2047_nodes(self):
        """Build from filesystem should create all 2047 nodes."""
        root = os.path.abspath(DESIGN_ROOT)
        if not os.path.exists(os.path.join(root, "DESIGN.md")):
            pytest.skip("DESIGN.md filesystem not available")

        builder = BrainTreeBuilder()
        tree = builder.build_from_filesystem(root)

        assert tree.total_nodes == 2047
        assert tree.max_layer == 10

    def test_filesystem_tree_has_root(self):
        """Tree built from filesystem should have root node."""
        root = os.path.abspath(DESIGN_ROOT)
        if not os.path.exists(os.path.join(root, "DESIGN.md")):
            pytest.skip("DESIGN.md filesystem not available")

        builder = BrainTreeBuilder()
        tree = builder.build_from_filesystem(root)

        assert tree.root is not None
        assert tree.root.node_id == 0

    def test_filesystem_tree_layer_counts(self):
        """Tree built from filesystem should have correct layer counts."""
        root = os.path.abspath(DESIGN_ROOT)
        if not os.path.exists(os.path.join(root, "DESIGN.md")):
            pytest.skip("DESIGN.md filesystem not available")

        builder = BrainTreeBuilder()
        tree = builder.build_from_filesystem(root)

        for layer in range(11):
            count = len(tree.get_nodes_by_layer(layer))
            expected = 2 ** layer
            assert count == expected, (
                f"Layer {layer}: expected {expected}, got {count}"
            )

    def test_filesystem_tree_validation(self):
        """Tree built from filesystem should pass validation."""
        root = os.path.abspath(DESIGN_ROOT)
        if not os.path.exists(os.path.join(root, "DESIGN.md")):
            pytest.skip("DESIGN.md filesystem not available")

        builder = BrainTreeBuilder()
        tree = builder.build_from_filesystem(root)

        is_valid, errors = tree.validate()
        assert is_valid, f"Filesystem tree validation failed: {errors}"
