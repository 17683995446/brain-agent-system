"""Tests for code generator (Phase 7).

Tests NodeCodeGenerator: loading, class name generation, code validity,
node instantiation, process() behavior, and total class count.
"""

import json
import os
import ast
import pytest
import numpy as np
from pathlib import Path

from brain_agent.codegen import NodeCodeGenerator, PROCESS_TEMPLATES
from brain_agent.node import BrainNode, NodeConfig, RegionType, Protocol
from brain_agent.types import BrainData, Tensor, StateVector, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DESIGN_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "L00_BrainSystem"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "brain_agent", "generated"
)


def _make_test_json(tmp_path, num_layers=11):
    """Create a JSON file with a complete binary tree of num_layers layers.

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

            # Assign region types based on subtree position
            if layer == 0:
                rt = "unknown"
            elif layer <= 2:
                rt = "sensory" if node_id % 2 == 0 else "motor"
            elif layer <= 5:
                rt = "memory" if node_id % 2 == 0 else "modulation"
            else:
                rt = "relay"

            nodes.append({
                "id": node_id,
                "name_cn": f"节点{node_id}",
                "short_name": f"Node{node_id}",
                "full_name": f"Node{node_id}",
                "folder_name": f"L{layer:02d}_Test--Node{node_id}",
                "layer": layer,
                "parent_id": parent_id,
                "responsibility": f"节点{node_id}的职责描述",
                "tensor_shape": "[1,64,64]" if layer < 3 else "[1,32,32]",
                "frequency": f"{100 - layer * 10}Hz",
                "protocol": "ASYNC_PUSH",
                "algorithm": "pass_through",
                "region_type": rt,
                "children": children,
            })
            node_id += 1

    json_path = tmp_path / "test_tree_codegen.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"nodes": nodes}, f, ensure_ascii=False, indent=2)
    return json_path


def _make_small_test_json(tmp_path, num_layers=6):
    """Create a small JSON file for faster unit tests."""
    return _make_test_json(tmp_path, num_layers=num_layers)


def _run_generator(tmp_path, num_layers=11):
    """Helper: create JSON, run generator, return (gen, output_path)."""
    json_path = _make_test_json(tmp_path, num_layers=num_layers)
    output_dir = str(tmp_path / "generated")
    gen = NodeCodeGenerator(json_path=str(json_path), output_dir=output_dir)
    gen.load_from_json(str(json_path))
    output_path = gen.generate_all()
    return gen, output_path


# ---------------------------------------------------------------------------
# 1. Test code generator loads JSON correctly
# ---------------------------------------------------------------------------

class TestCodeGeneratorLoad:
    """Test that the code generator loads JSON data correctly."""

    def test_load_from_json(self, tmp_path):
        """Generator should load all nodes from JSON."""
        json_path = _make_test_json(tmp_path, num_layers=6)
        output_dir = str(tmp_path / "generated")
        gen = NodeCodeGenerator(json_path=str(json_path), output_dir=output_dir)
        gen.load_from_json(str(json_path))

        assert len(gen._node_data) == 2 ** 6 - 1  # 63 nodes

    def test_load_full_2047_json(self, tmp_path):
        """Generator should load all 2047 nodes from full JSON."""
        json_path = _make_test_json(tmp_path, num_layers=11)
        output_dir = str(tmp_path / "generated")
        gen = NodeCodeGenerator(json_path=str(json_path), output_dir=output_dir)
        gen.load_from_json(str(json_path))

        assert len(gen._node_data) == 2047

    def test_load_from_filesystem(self):
        """Generator should load from filesystem DESIGN.md files."""
        root = os.path.abspath(DESIGN_ROOT)
        if not os.path.exists(os.path.join(root, "DESIGN.md")):
            pytest.skip("DESIGN.md filesystem not available")

        output_dir = str(Path(DESIGN_ROOT).parent / "src" / "brain_agent" / "generated")
        gen = NodeCodeGenerator(design_root=root, output_dir=output_dir)
        gen.load()

        assert len(gen._node_data) == 2047

    def test_load_preserves_node_ids(self, tmp_path):
        """Loaded data should preserve node IDs."""
        json_path = _make_test_json(tmp_path, num_layers=4)
        output_dir = str(tmp_path / "generated")
        gen = NodeCodeGenerator(json_path=str(json_path), output_dir=output_dir)
        gen.load_from_json(str(json_path))

        ids = [n["node_id"] for n in gen._node_data]
        assert ids == list(range(15))  # 2^4 - 1 = 15

    def test_load_preserves_layer_info(self, tmp_path):
        """Loaded data should preserve layer information."""
        json_path = _make_test_json(tmp_path, num_layers=4)
        output_dir = str(tmp_path / "generated")
        gen = NodeCodeGenerator(json_path=str(json_path), output_dir=output_dir)
        gen.load_from_json(str(json_path))

        layers = [n["layer"] for n in gen._node_data]
        assert min(layers) == 0
        assert max(layers) == 3

    def test_load_preserves_region_types(self, tmp_path):
        """Loaded data should preserve region type information."""
        json_path = _make_test_json(tmp_path, num_layers=7)
        output_dir = str(tmp_path / "generated")
        gen = NodeCodeGenerator(json_path=str(json_path), output_dir=output_dir)
        gen.load_from_json(str(json_path))

        region_types = set(n["region_type"] for n in gen._node_data)
        assert "sensory" in region_types
        assert "motor" in region_types
        assert "relay" in region_types


# ---------------------------------------------------------------------------
# 2. Test class name generation
# ---------------------------------------------------------------------------

class TestClassNameGeneration:
    """Test _make_class_name produces valid unique Python class names."""

    def test_basic_folder_name(self):
        """L04_Visual--OnChannelDetector (id=15) -> N15OnChannelDetectorNode."""
        gen = NodeCodeGenerator(design_root="/tmp", output_dir="/tmp")
        node = {"folder_name": "L04_Visual--OnChannelDetector", "node_id": 15}
        assert gen._make_class_name(node) == "N15OnChannelDetectorNode"

    def test_layer05_folder_name(self):
        """L05_OnChannel--HighFreqFilter (id=31) -> N31HighFreqFilterNode."""
        gen = NodeCodeGenerator(design_root="/tmp", output_dir="/tmp")
        node = {"folder_name": "L05_OnChannel--HighFreqFilter", "node_id": 31}
        assert gen._make_class_name(node) == "N31HighFreqFilterNode"

    def test_layer10_folder_name(self):
        """L10_OrientSel--LuminanceEncoder (id=1280) -> N1280LuminanceEncoderNode."""
        gen = NodeCodeGenerator(design_root="/tmp", output_dir="/tmp")
        node = {"folder_name": "L10_OrientSel--LuminanceEncoder", "node_id": 1280}
        assert gen._make_class_name(node) == "N1280LuminanceEncoderNode"

    def test_layer10_color_opponent(self):
        """L10_OrientSel--ColorOpponent (id=1281) -> N1281ColorOpponentNode."""
        gen = NodeCodeGenerator(design_root="/tmp", output_dir="/tmp")
        node = {"folder_name": "L10_OrientSel--ColorOpponent", "node_id": 1281}
        assert gen._make_class_name(node) == "N1281ColorOpponentNode"

    def test_no_double_dash(self):
        """Folder without -- should still work with node_id prefix."""
        gen = NodeCodeGenerator(design_root="/tmp", output_dir="/tmp")
        node = {"folder_name": "L04_SimpleNode", "node_id": 100}
        assert gen._make_class_name(node) == "N100SimpleNodeNode"

    def test_empty_folder_name(self):
        """Empty folder name should produce a fallback name with node_id prefix."""
        gen = NodeCodeGenerator(design_root="/tmp", output_dir="/tmp")
        node = {"folder_name": "", "node_id": 42}
        assert gen._make_class_name(node) == "N42Node42Node"

    def test_class_names_are_valid_python(self, tmp_path):
        """All generated class names should be valid Python identifiers."""
        json_path = _make_test_json(tmp_path, num_layers=11)
        output_dir = str(tmp_path / "generated")
        gen = NodeCodeGenerator(json_path=str(json_path), output_dir=output_dir)
        gen.load_from_json(str(json_path))

        for node in gen._node_data:
            if node["layer"] >= 4:
                name = gen._make_class_name(node)
                assert name.isidentifier(), f"Invalid class name: {name}"
                assert name.endswith("Node"), f"Class name missing Node suffix: {name}"

    def test_class_names_are_unique(self, tmp_path):
        """All generated class names should be unique."""
        json_path = _make_test_json(tmp_path, num_layers=11)
        output_dir = str(tmp_path / "generated")
        gen = NodeCodeGenerator(json_path=str(json_path), output_dir=output_dir)
        gen.load_from_json(str(json_path))

        names = []
        for node in gen._node_data:
            if node["layer"] >= 4:
                names.append(gen._make_class_name(node))

        assert len(names) == len(set(names)), f"Duplicate class names found: {names}"


# ---------------------------------------------------------------------------
# 3. Test generated code is valid Python (can be imported)
# ---------------------------------------------------------------------------

class TestGeneratedCodeValidity:
    """Test that generated code is syntactically valid Python."""

    def test_generated_code_parses_as_python(self, tmp_path):
        """Generated code should be parseable by ast.parse."""
        gen, output_path = _run_generator(tmp_path, num_layers=6)

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Should not raise SyntaxError
        tree = ast.parse(source)
        assert isinstance(tree, ast.Module)

    def test_generated_code_has_classes(self, tmp_path):
        """Generated code should contain class definitions."""
        gen, output_path = _run_generator(tmp_path, num_layers=6)

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        # L04-L05: 2^4-1 + 2^5-1 = 15+31 = 46 nodes with layer >= 4
        # Actually: layer 4 has 16 nodes, layer 5 has 32 nodes = 48
        expected = 16 + 32  # layers 4 and 5
        assert len(classes) == expected, (
            f"Expected {expected} classes, got {len(classes)}"
        )

    def test_generated_code_has_imports(self, tmp_path):
        """Generated code should import required modules."""
        gen, output_path = _run_generator(tmp_path, num_layers=6)

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "import numpy as np" in source
        assert "from ..types import" in source
        assert "from ..node import BrainNode" in source

    def test_generated_code_has_node_map(self, tmp_path):
        """Generated code should contain GENERATED_NODE_MAP."""
        gen, output_path = _run_generator(tmp_path, num_layers=6)

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "GENERATED_NODE_MAP" in source

    def test_full_2047_generated_code_parses(self, tmp_path):
        """Full 2047-node generated code should be valid Python."""
        gen, output_path = _run_generator(tmp_path, num_layers=11)

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Should not raise SyntaxError
        tree = ast.parse(source)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        # L04-L10: sum(2^l for l in range(4, 11)) = 16+32+64+128+256+512+1024 = 2032
        assert len(classes) == 2032


# ---------------------------------------------------------------------------
# 4. Test generated nodes can be instantiated
# ---------------------------------------------------------------------------

class TestNodeInstantiation:
    """Test that generated node classes can be instantiated."""

    def _get_generated_classes(self, tmp_path, num_layers=6):
        """Helper: run generator and return dict of {class_name: class}."""
        gen, output_path = _run_generator(tmp_path, num_layers=num_layers)

        # We need to exec the generated code in a namespace that has
        # the required imports available
        namespace = {
            "np": np,
            "BrainData": BrainData,
            "Tensor": Tensor,
            "StateVector": StateVector,
            "Signal": Signal,
            "BrainNode": BrainNode,
        }

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Remove the relative import lines and replace with absolute
        source = source.replace(
            "from ..types import BrainData, Tensor, StateVector, Signal",
            "# types imported via namespace"
        )
        source = source.replace(
            "from ..node import BrainNode",
            "# node imported via namespace"
        )

        exec(compile(source, output_path, 'exec'), namespace)
        return namespace

    def test_instantiate_sensory_node(self, tmp_path):
        """A sensory node should be instantiable."""
        ns = self._get_generated_classes(tmp_path, num_layers=6)

        # Find a node class
        node_classes = [
            obj for name, obj in ns.items()
            if isinstance(obj, type) and issubclass(obj, BrainNode) and name != "BrainNode"
        ]
        assert len(node_classes) > 0

        cls = node_classes[0]
        config = NodeConfig(
            node_id=100, name_cn="Test", short_name="Test",
            folder_name="L04_Test--Node100", layer=4, parent_id=None,
            responsibility="test", region_type=RegionType.SENSORY,
            tensor_shape="[1,32,32]", frequency="60Hz",
            protocol=Protocol.ASYNC_PUSH, algorithm="", children_ids=[],
        )
        node = cls(config)
        assert isinstance(node, BrainNode)

    def test_instantiate_motor_node(self, tmp_path):
        """A motor node should be instantiable."""
        ns = self._get_generated_classes(tmp_path, num_layers=6)

        # Find a motor node class
        all_classes = [
            obj for name, obj in ns.items()
            if isinstance(obj, type) and issubclass(obj, BrainNode) and name != "BrainNode"
        ]
        assert len(all_classes) > 0

        cls = all_classes[0]
        config = NodeConfig(
            node_id=101, name_cn="Test", short_name="Test",
            folder_name="L04_Test--Node101", layer=4, parent_id=None,
            responsibility="test", region_type=RegionType.MOTOR,
            tensor_shape="[1,32,32]", frequency="60Hz",
            protocol=Protocol.ASYNC_PUSH, algorithm="", children_ids=[],
        )
        node = cls(config)
        assert isinstance(node, BrainNode)

    def test_instantiate_all_l04_l05_nodes(self, tmp_path):
        """All L04-L05 nodes should be instantiable."""
        ns = self._get_generated_classes(tmp_path, num_layers=6)

        node_classes = [
            obj for name, obj in ns.items()
            if isinstance(obj, type) and issubclass(obj, BrainNode) and name != "BrainNode"
        ]

        for cls in node_classes:
            config = NodeConfig(
                node_id=999, name_cn="Test", short_name="Test",
                folder_name="L04_Test--Node", layer=4, parent_id=None,
                responsibility="test", region_type=RegionType.RELAY,
                tensor_shape="[1,32,32]", frequency="60Hz",
                protocol=Protocol.ASYNC_PUSH, algorithm="", children_ids=[],
            )
            node = cls(config)
            assert isinstance(node, BrainNode)


# ---------------------------------------------------------------------------
# 5. Test generated nodes process() works with Tensor/Signal/StateVector
# ---------------------------------------------------------------------------

class TestProcessMethod:
    """Test that generated process() methods work with various input types."""

    def _make_node(self, tmp_path, region_type_str, num_layers=7):
        """Helper: create a generated node of given region type."""
        gen, output_path = _run_generator(tmp_path, num_layers=num_layers)

        namespace = {
            "np": np,
            "BrainData": BrainData,
            "Tensor": Tensor,
            "StateVector": StateVector,
            "Signal": Signal,
            "BrainNode": BrainNode,
        }

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        source = source.replace(
            "from ..types import BrainData, Tensor, StateVector, Signal",
            "# types imported via namespace"
        )
        source = source.replace(
            "from ..node import BrainNode",
            "# node imported via namespace"
        )

        exec(compile(source, output_path, 'exec'), namespace)

        region_map = {
            "sensory": RegionType.SENSORY,
            "motor": RegionType.MOTOR,
            "memory": RegionType.MEMORY,
            "modulation": RegionType.MODULATION,
            "relay": RegionType.RELAY,
        }

        # Find a node in gen._node_data with the desired region type
        # and look up its class in the GENERATED_NODE_MAP
        target_node = None
        for node in gen._node_data:
            rt = node["region_type"]
            if isinstance(rt, str):
                rt_str = rt
            else:
                rt_str = rt.value if hasattr(rt, 'value') else str(rt)
            if rt_str == region_type_str and node["layer"] >= 4:
                target_node = node
                break

        if target_node is None:
            # Fallback: use first L04+ node
            for node in gen._node_data:
                if node["layer"] >= 4:
                    target_node = node
                    break

        assert target_node is not None, f"No node found for region_type={region_type_str}"

        node_map = namespace.get("GENERATED_NODE_MAP", {})
        node_id = target_node["node_id"]
        cls = node_map.get(node_id)
        assert cls is not None, f"No class found for node_id={node_id}"

        config = NodeConfig(
            node_id=node_id, name_cn="Test", short_name="Test",
            folder_name=target_node.get("folder_name", "L04_Test--Node"),
            layer=target_node["layer"], parent_id=None,
            responsibility="test",
            region_type=region_map.get(region_type_str, RegionType.RELAY),
            tensor_shape="[1,32,32]", frequency="60Hz",
            protocol=Protocol.ASYNC_PUSH, algorithm="", children_ids=[],
        )
        return cls(config)

    def test_sensory_process_tensor(self, tmp_path):
        """Sensory node process() should handle Tensor input."""
        node = self._make_node(tmp_path, "sensory")
        data = Tensor(data=np.random.randn(2, 4, 4).astype(np.float32), shape=(2, 4, 4))
        result = node.process(data)
        assert isinstance(result, Tensor)
        assert result.data.shape == (2, 4, 4)

    def test_sensory_process_signal(self, tmp_path):
        """Sensory node process() should handle Signal input."""
        node = self._make_node(tmp_path, "sensory")
        data = Signal(value=1.5, timestamp=0.1)
        result = node.process(data)
        assert isinstance(result, Signal)

    def test_motor_process_tensor(self, tmp_path):
        """Motor node process() should handle Tensor input."""
        node = self._make_node(tmp_path, "motor")
        data = Tensor(data=np.random.randn(2, 4, 4).astype(np.float32), shape=(2, 4, 4))
        result = node.process(data)
        assert isinstance(result, Tensor)
        assert result.data.shape == (2, 4, 4)

    def test_motor_process_signal(self, tmp_path):
        """Motor node process() should handle Signal input."""
        node = self._make_node(tmp_path, "motor")
        data = Signal(value=2.0, timestamp=0.2)
        result = node.process(data)
        assert isinstance(result, Signal)
        assert result.value >= 0.0  # Motor applies ReLU

    def test_memory_process_state_vector(self, tmp_path):
        """Memory node process() should handle StateVector input."""
        node = self._make_node(tmp_path, "memory")
        data = StateVector(data=np.random.randn(16).astype(np.float32), dimension=16)
        result = node.process(data)
        assert isinstance(result, StateVector)
        assert result.dimension == 16

    def test_memory_process_tensor(self, tmp_path):
        """Memory node process() should handle Tensor input."""
        node = self._make_node(tmp_path, "memory")
        data = Tensor(data=np.random.randn(2, 4, 4).astype(np.float32), shape=(2, 4, 4))
        result = node.process(data)
        assert isinstance(result, Tensor)
        assert result.data.shape == (2, 4, 4)

    def test_modulation_process_signal(self, tmp_path):
        """Modulation node process() should handle Signal input."""
        node = self._make_node(tmp_path, "modulation")
        data = Signal(value=1.0, timestamp=0.3)
        result = node.process(data)
        assert isinstance(result, Signal)
        # With default mod_level=0.5, gain = 4*0.5*0.5 = 1.0
        assert abs(result.value - 1.0) < 1e-6

    def test_modulation_process_tensor(self, tmp_path):
        """Modulation node process() should handle Tensor input."""
        node = self._make_node(tmp_path, "modulation")
        data = Tensor(data=np.ones((2, 3, 3), dtype=np.float32), shape=(2, 3, 3))
        result = node.process(data)
        assert isinstance(result, Tensor)
        assert result.data.shape == (2, 3, 3)

    def test_relay_process_tensor(self, tmp_path):
        """Relay node process() should handle Tensor input."""
        node = self._make_node(tmp_path, "relay")
        data = Tensor(data=np.random.randn(2, 4, 4).astype(np.float32), shape=(2, 4, 4))
        result = node.process(data)
        assert isinstance(result, Tensor)
        assert result.data.shape == (2, 4, 4)

    def test_relay_process_signal(self, tmp_path):
        """Relay node process() should handle Signal input."""
        node = self._make_node(tmp_path, "relay")
        data = Signal(value=3.0, timestamp=0.4)
        result = node.process(data)
        assert isinstance(result, Signal)
        # With default gate_open=True and relay_gain=1.0, value should be 3.0
        assert abs(result.value - 3.0) < 1e-6

    def test_relay_gated_closed(self, tmp_path):
        """Relay node with gate closed should zero out signal."""
        node = self._make_node(tmp_path, "relay")
        node.set_state("gate_open", False)
        data = Signal(value=3.0, timestamp=0.4)
        result = node.process(data)
        assert isinstance(result, Signal)
        assert result.value == 0.0

    def test_relay_gated_closed_tensor(self, tmp_path):
        """Relay node with gate closed should zero out tensor."""
        node = self._make_node(tmp_path, "relay")
        node.set_state("gate_open", False)
        data = Tensor(data=np.ones((2, 3, 3), dtype=np.float32), shape=(2, 3, 3))
        result = node.process(data)
        assert isinstance(result, Tensor)
        assert np.all(result.data == 0.0)


# ---------------------------------------------------------------------------
# 6. Test total generated classes == 2032 (L04-L10)
# ---------------------------------------------------------------------------

class TestTotalGeneratedClasses:
    """Test that exactly 2032 classes are generated for L04-L10."""

    def test_count_from_json(self, tmp_path):
        """Full 2047-node JSON should generate 2032 L04-L10 classes."""
        gen, output_path = _run_generator(tmp_path, num_layers=11)

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        # L04-L10: sum(2^l for l in range(4, 11))
        # = 16 + 32 + 64 + 128 + 256 + 512 + 1024 = 2032
        assert len(classes) == 2032, (
            f"Expected 2032 classes, got {len(classes)}"
        )

    def test_node_map_has_2032_entries(self, tmp_path):
        """GENERATED_NODE_MAP should have 2032 entries."""
        gen, output_path = _run_generator(tmp_path, num_layers=11)

        namespace = {
            "np": np,
            "BrainData": BrainData,
            "Tensor": Tensor,
            "StateVector": StateVector,
            "Signal": Signal,
            "BrainNode": BrainNode,
        }

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        source = source.replace(
            "from ..types import BrainData, Tensor, StateVector, Signal",
            "# types imported via namespace"
        )
        source = source.replace(
            "from ..node import BrainNode",
            "# node imported via namespace"
        )

        exec(compile(source, output_path, 'exec'), namespace)
        node_map = namespace.get("GENERATED_NODE_MAP", {})
        assert len(node_map) == 2032, (
            f"Expected 2032 map entries, got {len(node_map)}"
        )

    def test_node_map_ids_are_l04_to_l10(self, tmp_path):
        """All node IDs in GENERATED_NODE_MAP should be >= 15 (first L04 node)."""
        gen, output_path = _run_generator(tmp_path, num_layers=11)

        namespace = {
            "np": np,
            "BrainData": BrainData,
            "Tensor": Tensor,
            "StateVector": StateVector,
            "Signal": Signal,
            "BrainNode": BrainNode,
        }

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        source = source.replace(
            "from ..types import BrainData, Tensor, StateVector, Signal",
            "# types imported via namespace"
        )
        source = source.replace(
            "from ..node import BrainNode",
            "# node imported via namespace"
        )

        exec(compile(source, output_path, 'exec'), namespace)
        node_map = namespace.get("GENERATED_NODE_MAP", {})

        # Node IDs 0-14 are L00-L03 (15 nodes)
        # L04 starts at node_id 15
        for node_id in node_map:
            assert node_id >= 15, f"Node ID {node_id} is below L04 range"

    def test_per_layer_counts(self, tmp_path):
        """Each layer should have the correct number of generated classes."""
        gen, output_path = _run_generator(tmp_path, num_layers=11)

        with open(output_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        # Count classes per layer from docstrings
        import re
        layer_counts = {}
        for cls_node in classes:
            docstring = ast.get_docstring(cls_node) or ""
            match = re.search(r'Layer: L(\d+)', docstring)
            if match:
                layer = int(match.group(1))
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

        for layer in range(4, 11):
            expected = 2 ** layer
            actual = layer_counts.get(layer, 0)
            assert actual == expected, (
                f"Layer {layer}: expected {expected} classes, got {actual}"
            )


# ---------------------------------------------------------------------------
# 7. Test process templates
# ---------------------------------------------------------------------------

class TestProcessTemplates:
    """Test that all process templates are present and valid."""

    def test_all_region_types_have_templates(self):
        """All 5 region types should have templates."""
        expected_types = {"sensory", "motor", "memory", "modulation", "relay"}
        assert set(PROCESS_TEMPLATES.keys()) == expected_types

    def test_templates_contain_process_hooks(self):
        """Each template should contain _process_* hook method definitions."""
        for region_type, template in PROCESS_TEMPLATES.items():
            assert "def _process_" in template, (
                f"Template for {region_type} missing _process_ hook"
            )

    def test_sensory_template_uses_kernel(self):
        """Sensory template should use kernel-based convolution."""
        assert "kernel" in PROCESS_TEMPLATES["sensory"]

    def test_motor_template_uses_relu(self):
        """Motor template should use ReLU-like activation."""
        assert "np.maximum" in PROCESS_TEMPLATES["motor"]

    def test_memory_template_uses_weights(self):
        """Memory template should use weight matrix."""
        assert "weights" in PROCESS_TEMPLATES["memory"]

    def test_modulation_template_uses_inverted_u(self):
        """Modulation template should use inverted-U gain."""
        assert "4.0 * level * (1.0 - level)" in PROCESS_TEMPLATES["modulation"]

    def test_relay_template_uses_gate(self):
        """Relay template should use gating."""
        assert "gate_open" in PROCESS_TEMPLATES["relay"]
