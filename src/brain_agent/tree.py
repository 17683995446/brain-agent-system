"""Binary tree builder for the brain simulation agent system.

Parses DESIGN.md files from the filesystem to build the 2047-node binary tree.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from collections import deque

from .node import BrainNode, NodeConfig, RegionType, Protocol
from .types import BrainData

logger = logging.getLogger(__name__)

# Field names used in Chinese DESIGN.md files. Encoded as Unicode escapes so
# the source stays ASCII-clean while still matching the Chinese document format.
# Translations are provided in comments for maintainability.
_FIELD_LAYER = "\u5c42\u7ea7"              # Layer
_FIELD_NODE_ID = "\u8282\u70b9ID"          # Node ID
_FIELD_FOLDER_NAME = "\u6587\u4ef6\u5939\u540d"  # Folder name
_FIELD_NODE_TYPE = "\u8282\u70b9\u7c7b\u578b"    # Node type
_LEAF_MARKER = "\u53f6\u8282\u70b9"        # Leaf node marker
_FIELD_REGION_TYPE = "\u8111\u533a\u7c7b\u578b"  # Region type
_FIELD_CHILDREN_COUNT = "\u5b50\u8282\u70b9\u6570"  # Children count
_FIELD_PARENT = "\u7236\u6a21\u5757"       # Parent module
_SECTION_RESPONSIBILITY = "\u804c\u8d23\u529f\u80fd"  # Responsibility section
_FIELD_ALGORITHM = "\u7b97\u6cd5"          # Algorithm
# Characters used in the layer string format, e.g. "Layer 0 (L00)"
_CHAR_DI = "\u7b2c"                         # Ordinal prefix
_CHAR_CENG = "\u5c42"                       # Layer suffix
# Fullwidth colon used in some DESIGN.md field markers
_CHAR_FW_COLON = "\uff1a"


class DesignDocParser:
    """Parse DESIGN.md files to extract node metadata."""

    @staticmethod
    def parse(filepath: str) -> Dict[str, Any]:
        """Parse a DESIGN.md file and return node metadata dict."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract layer number from the layer field, e.g. "Layer 0 (L00)" -> 0
        layer_str = DesignDocParser._extract_field(content, _FIELD_LAYER)
        layer = DesignDocParser._parse_layer(layer_str)

        # Extract node_id
        node_id_str = DesignDocParser._extract_field(content, _FIELD_NODE_ID)
        node_id = int(node_id_str) if node_id_str.isdigit() else 0

        # Extract folder_name, stripping backticks if present
        folder_name = DesignDocParser._extract_field(content, _FIELD_FOLDER_NAME).strip("`")

        # Extract node_type to determine if leaf
        node_type_str = DesignDocParser._extract_field(content, _FIELD_NODE_TYPE)
        is_leaf = _LEAF_MARKER in node_type_str

        # Extract region_type
        region_type_str = DesignDocParser._extract_field(content, _FIELD_REGION_TYPE)
        region_type = DesignDocParser._parse_region_type(region_type_str)

        # Extract children count
        children_count_str = DesignDocParser._extract_field(content, _FIELD_CHILDREN_COUNT)
        children_count = int(children_count_str) if children_count_str.isdigit() else 0

        # Extract parent module
        parent_str = DesignDocParser._extract_field(content, _FIELD_PARENT)

        # Extract responsibility
        responsibility = DesignDocParser._extract_section(content, _SECTION_RESPONSIBILITY)

        # Extract algorithm
        algorithm = DesignDocParser._extract_field(content, _FIELD_ALGORITHM)

        # Extract tensor shape and frequency from input interface table
        tensor_shape = DesignDocParser._extract_tensor_shape(content)
        frequency = DesignDocParser._extract_frequency(content)

        # Extract protocol from input interface table
        protocol = DesignDocParser._extract_protocol(content)

        # Derive short_name and full_name from folder_name
        short_name = DesignDocParser._derive_short_name(folder_name, layer)
        full_name = DesignDocParser._derive_full_name(folder_name, layer)

        # Determine parent_id (will be resolved later during tree building)
        parent_id = None

        metadata = {
            "node_id": node_id,
            "name_cn": "",  # Will be filled from the title line
            "short_name": short_name,
            "full_name": full_name,
            "folder_name": folder_name,
            "layer": layer,
            "parent_id": parent_id,
            "parent_name": parent_str,
            "responsibility": responsibility,
            "region_type": region_type,
            "tensor_shape": tensor_shape,
            "frequency": frequency,
            "protocol": protocol,
            "algorithm": algorithm,
            "is_leaf": is_leaf,
            "children_count": children_count,
        }

        return metadata

    @staticmethod
    def _extract_field(content: str, field_name: str) -> str:
        """Extract a field value from markdown: - **field_name**: value"""
        pattern = rf'\*\*{re.escape(field_name)}\*\*[:{_CHAR_FW_COLON}]\s*(.+?)(?:\n|$)'
        match = re.search(pattern, content)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_section(content: str, section_name: str) -> str:
        """Extract text under a ## section heading."""
        pattern = rf'## {re.escape(section_name)}\s*\n(.+?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _parse_layer(layer_str: str) -> int:
        """Parse layer number from a layer string (e.g. 'L00' or 'L10' format)."""
        match = re.search(_CHAR_DI + r'(\d+)' + _CHAR_CENG, layer_str)
        if match:
            return int(match.group(1))
        return 0

    @staticmethod
    def _parse_region_type(region_str: str) -> RegionType:
        """Parse region type string to RegionType enum."""
        mapping = {
            "sensory": RegionType.SENSORY,
            "motor": RegionType.MOTOR,
            "memory": RegionType.MEMORY,
            "modulation": RegionType.MODULATION,
            "relay": RegionType.RELAY,
            "unknown": RegionType.UNKNOWN,
        }
        return mapping.get(region_str.strip().lower(), RegionType.UNKNOWN)

    @staticmethod
    def _extract_tensor_shape(content: str) -> str:
        """Extract tensor shape from input interface table."""
        # Look for pattern like Tensor[float32, 32x32x8] or Composite[All]
        match = re.search(r'Tensor\[float32,\s*([^\]]+)\]', content)
        if match:
            return f"[{match.group(1).strip()}]"
        match = re.search(r'Composite\[([^\]]+)\]', content)
        if match:
            return f"Composite[{match.group(1).strip()}]"
        return ""

    @staticmethod
    def _extract_frequency(content: str) -> str:
        """Extract frequency from input interface table."""
        # Look for pattern like 60-100Hz or 0.01-200Hz
        match = re.search(r'(\d+\.?\d*-\d+\.?\d*Hz)', content)
        if match:
            return match.group(1)
        match = re.search(r'(\d+\.?\d*Hz)', content)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _extract_protocol(content: str) -> Protocol:
        """Extract protocol from input interface table."""
        if "ASYNC_PUSH" in content:
            return Protocol.ASYNC_PUSH
        elif "SYNC_CALL" in content:
            return Protocol.SYNC_CALL
        elif "PUB_SUB" in content:
            return Protocol.PUB_SUB
        return Protocol.ASYNC_PUSH

    @staticmethod
    def _derive_short_name(folder_name: str, layer: int) -> str:
        """Derive a short name from folder name.

        E.g. 'L00_BrainSystem' -> 'BrainSystem'
              'L01_Brain--SensorimotorSystem' -> 'SensorimotorSystem'
        """
        # Remove the layer prefix (e.g. 'L00_')
        parts = folder_name.split('_', 1)
        if len(parts) > 1:
            name = parts[1]
            # If name contains '--', take the part after the last '--'
            if '--' in name:
                name = name.split('--')[-1]
            return name
        return folder_name

    @staticmethod
    def _derive_full_name(folder_name: str, layer: int) -> str:
        """Derive full name from folder name.

        E.g. 'L00_BrainSystem' -> 'BrainSystem'
        """
        parts = folder_name.split('_', 1)
        if len(parts) > 1:
            return parts[1].replace('--', '_')
        return folder_name


class PassThroughBrainNode(BrainNode):
    """A pass-through brain node that simply returns input data unchanged.

    Used as the default node implementation during tree construction.
    """

    def _process_tensor(self, data: BrainData) -> BrainData:
        return data

    def _process_state_vector(self, data: BrainData) -> BrainData:
        return data

    def _process_signal(self, data: BrainData) -> BrainData:
        return data


class BrainTreeBuilder:
    """Build the 2047-node binary tree from DESIGN.md files or JSON data."""

    def __init__(self) -> None:
        self._nodes: Dict[int, NodeConfig] = {}
        self._tree: Optional["BrainTree"] = None

    def build_from_filesystem(self, root_path: str, node_class: Any = PassThroughBrainNode) -> "BrainTree":
        """Build tree by scanning DESIGN.md files in the filesystem.

        Walks the directory tree under root_path, parses each DESIGN.md file,
        and constructs the full binary tree.

        Args:
            root_path: Path to the root directory containing DESIGN.md files
            node_class: BrainNode subclass to instantiate for each node

        Returns:
            BrainTree instance with all nodes connected
        """
        root_path_obj = Path(root_path)
        if not root_path_obj.exists():
            raise FileNotFoundError(f"Root path does not exist: {root_path_obj}")

        # Phase 1: Walk once, parse all DESIGN.md files, collect metadata
        # Also build mappings between directories and node IDs
        all_metadata: Dict[int, Dict[str, Any]] = {}
        dir_to_node_id: Dict[str, int] = {}
        node_id_to_dir: Dict[int, str] = {}

        for dirpath, dirnames, filenames in os.walk(root_path):
            if "DESIGN.md" in filenames:
                filepath = os.path.join(dirpath, "DESIGN.md")
                try:
                    metadata = DesignDocParser.parse(filepath)
                    node_id = metadata["node_id"]
                    all_metadata[node_id] = metadata
                    dir_to_node_id[dirpath] = node_id
                    node_id_to_dir[node_id] = dirpath
                except Exception as e:
                    logger.warning(f"Failed to parse {filepath}: {e}")

        if not all_metadata:
            raise ValueError(f"No DESIGN.md files found under {root_path}")

        # Phase 2: Resolve parent IDs using directory hierarchy
        # For each node, walk up the directory tree to find the nearest
        # ancestor directory that also contains a DESIGN.md
        for node_id, metadata in all_metadata.items():
            if metadata["layer"] == 0:
                metadata["parent_id"] = None
                continue

            node_dir = node_id_to_dir.get(node_id)
            if node_dir is None:
                continue

            # Walk up directories to find parent
            current = str(Path(node_dir).parent)
            root_str = str(root_path_obj)
            while current != root_str and current != os.path.dirname(root_str):
                if current in dir_to_node_id:
                    metadata["parent_id"] = dir_to_node_id[current]
                    break
                current = str(Path(current).parent)

        # Phase 3: Build NodeConfig for each node
        for node_id, metadata in all_metadata.items():
            config = self._create_node_config(metadata)
            self._nodes[node_id] = config

        # Phase 4: Resolve children IDs based on parent relationships
        self._resolve_children_ids()

        # Phase 5: Create node instances and connect them
        return self._assemble_tree(node_class)

    def build_from_json(self, json_path: str, node_class: Any = PassThroughBrainNode) -> "BrainTree":
        """Build tree from a JSON file with all node definitions.

        Args:
            json_path: Path to JSON file containing node definitions
            node_class: BrainNode subclass to instantiate for each node

        Returns:
            BrainTree instance with all nodes connected
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        nodes_data = data.get("nodes", [])
        if not nodes_data:
            raise ValueError(f"JSON file contains no nodes: {json_path}")

        # Phase 1: Build NodeConfig for each node
        for node_data in nodes_data:
            config = self._create_node_config_from_json(node_data)
            self._nodes[config.node_id] = config

        # Phase 2: Create node instances and connect them
        tree = self._assemble_tree(node_class)

        logger.info(f"Built tree from JSON: {tree.total_nodes} nodes, "
                    f"max layer: {tree.max_layer}")
        return tree

    def _resolve_children_ids(self) -> None:
        """Resolve children_ids based on parent_id relationships."""
        for node_id, config in self._nodes.items():
            if config.parent_id is not None and config.parent_id in self._nodes:
                parent_config = self._nodes[config.parent_id]
                if node_id not in parent_config.children_ids:
                    parent_config.children_ids.append(node_id)

    def _create_node_config(self, metadata: Dict[str, Any]) -> NodeConfig:
        """Convert raw DESIGN.md metadata to NodeConfig."""
        return NodeConfig(
            node_id=metadata["node_id"],
            name_cn=metadata.get("name_cn", ""),
            short_name=metadata["short_name"],
            folder_name=metadata["folder_name"],
            layer=metadata["layer"],
            parent_id=metadata.get("parent_id"),
            responsibility=metadata.get("responsibility", ""),
            region_type=metadata["region_type"],
            tensor_shape=metadata.get("tensor_shape", ""),
            frequency=metadata.get("frequency", ""),
            protocol=metadata["protocol"],
            algorithm=metadata.get("algorithm", ""),
            children_ids=[],
        )

    def _create_node_config_from_json(self, node_data: Dict[str, Any]) -> NodeConfig:
        """Convert JSON node data to NodeConfig."""
        # Parse region_type
        region_type_str = node_data.get("region_type", "unknown")
        try:
            region_type = RegionType(region_type_str)
        except ValueError:
            region_type = RegionType.UNKNOWN

        # Parse protocol
        protocol_str = node_data.get("protocol", "ASYNC_PUSH")
        try:
            protocol = Protocol(protocol_str)
        except ValueError:
            protocol = Protocol.ASYNC_PUSH

        return NodeConfig(
            node_id=node_data["id"],
            name_cn=node_data.get("name_cn", ""),
            short_name=node_data.get("short_name", ""),
            folder_name=node_data.get("folder_name", ""),
            layer=node_data["layer"],
            parent_id=node_data.get("parent_id"),
            responsibility=node_data.get("responsibility", ""),
            region_type=region_type,
            tensor_shape=node_data.get("tensor_shape", ""),
            frequency=node_data.get("frequency", ""),
            protocol=protocol,
            algorithm=node_data.get("algorithm", ""),
            children_ids=node_data.get("children", []),
        )

    def build_full_tree(self, json_path: str) -> "BrainTree":
        """Build the complete 2047-node tree with proper node classes.

        Uses the registry to instantiate each node with its correct class.
        """
        from .registry import create_node

        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create all node configs
        configs = {}
        for node_data in data["nodes"]:
            config = self._create_node_config_from_json(node_data)
            configs[node_data["id"]] = config

        # Create node instances using registry
        nodes = {}
        for node_id, config in configs.items():
            try:
                node = create_node(config)
                nodes[node_id] = node
            except ValueError:
                # Fall back to PassThroughBrainNode
                node = PassThroughBrainNode(config)
                nodes[node_id] = node

        # Assemble tree (connect parent-child)
        for node_id, node in nodes.items():
            config = configs[node_id]
            for child_id in config.children_ids:
                if child_id in nodes:
                    node.connect_child(nodes[child_id])

        return BrainTree(nodes)

    def _assemble_tree(self, node_class: Any) -> "BrainTree":
        """Create node instances and connect parent-child relationships."""
        # Create all node instances
        instances: Dict[int, BrainNode] = {}
        for node_id, config in self._nodes.items():
            instances[node_id] = node_class(config)

        # Connect parent-child relationships
        for node_id, config in self._nodes.items():
            for child_id in config.children_ids:
                if child_id in instances:
                    instances[node_id].connect_child(instances[child_id])

        self._tree = BrainTree(instances)
        return self._tree

    def _validate_tree(self) -> List[str]:
        """Validate the tree structure. Returns list of errors."""
        errors = []
        if not self._tree:
            errors.append("Tree has not been built yet")
            return errors

        # Check total node count
        if self._tree.total_nodes != 2047:
            errors.append(f"Node count: {self._tree.total_nodes}, expected: 2047")

        # Check each layer has correct count
        for layer in range(11):
            count = len(self._tree.get_nodes_by_layer(layer))
            expected = 2 ** layer
            if count != expected:
                errors.append(f"Layer {layer}: {count} nodes, expected: {expected}")

        # Check each non-leaf has exactly 2 children
        for node in self._tree._nodes.values():
            if not node.is_leaf and len(node.children) != 2:
                errors.append(
                    f"Node {node.node_id}: {len(node.children)} children, expected: 2"
                )

        # Check each child's parent_id matches
        for node in self._tree._nodes.values():
            if node.parent is not None:
                if node.node_id not in node.parent.children:
                    errors.append(
                        f"Node {node.node_id}: parent's children does not contain this node"
                    )

        # Check layer depth
        if self._tree.max_layer != 10:
            errors.append(f"Max layer: {self._tree.max_layer}, expected: 10")

        return errors


class BrainTree:
    """Container for the complete 2047-node binary tree.

    Provides traversal, lookup, and statistics methods.
    """

    def __init__(self, nodes: Dict[int, BrainNode]):
        self._nodes = nodes
        self._root = nodes[0] if 0 in nodes else None

    @property
    def root(self) -> Optional[BrainNode]:
        return self._root

    @property
    def total_nodes(self) -> int:
        return len(self._nodes)

    @property
    def max_layer(self) -> int:
        return max(n.layer for n in self._nodes.values()) if self._nodes else 0

    def get_node(self, node_id: int) -> Optional[BrainNode]:
        """Get a node by its ID."""
        return self._nodes.get(node_id)

    def get_nodes_by_layer(self, layer: int) -> List[BrainNode]:
        """Get all nodes at a specific layer."""
        return [n for n in self._nodes.values() if n.layer == layer]

    def get_nodes_by_type(self, region_type: RegionType) -> List[BrainNode]:
        """Get all nodes of a specific region type."""
        return [n for n in self._nodes.values() if n.region_type == region_type]

    def get_leaf_nodes(self) -> List[BrainNode]:
        """Get all leaf nodes (nodes with no children)."""
        return [n for n in self._nodes.values() if n.is_leaf]

    def get_path_to_root(self, node_id: int) -> List[BrainNode]:
        """Get the path from a node to the root.

        Returns a list starting from the given node and ending at the root.
        """
        path = []
        current = self._nodes.get(node_id)
        while current:
            path.append(current)
            current = current.parent
        return path

    def bfs_traverse(self) -> Iterator[BrainNode]:
        """Breadth-first traversal of the tree.

        Yields nodes level by level, left to right.
        """
        if not self._root:
            return
        queue = deque([self._root])
        while queue:
            node = queue.popleft()
            yield node
            for child in node.children.values():
                queue.append(child)

    def dfs_traverse(self) -> Iterator[BrainNode]:
        """Depth-first traversal of the tree (pre-order).

        Yields nodes in pre-order (root, left, right).
        """
        if not self._root:
            return
        stack = [self._root]
        while stack:
            node = stack.pop()
            yield node
            for child in reversed(list(node.children.values())):
                stack.append(child)

    def get_layer_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each layer.

        Returns a dict mapping layer number to stats dict containing:
        - count: actual number of nodes at this layer
        - expected: expected number of nodes (2^layer)
        - valid: whether count matches expected
        - types: dict of region_type -> count
        """
        stats = {}
        for layer in range(self.max_layer + 1):
            nodes = self.get_nodes_by_layer(layer)
            types: Dict[str, int] = {}
            for n in nodes:
                t = n.region_type.value
                types[t] = types.get(t, 0) + 1
            stats[layer] = {
                "count": len(nodes),
                "expected": 2 ** layer,
                "valid": len(nodes) == 2 ** layer,
                "types": types,
            }
        return stats

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the tree structure.

        Returns:
            Tuple of (is_valid, list_of_error_strings)
        """
        errors = []

        # Check node count
        if self.total_nodes != 2047:
            errors.append(f"Node count: {self.total_nodes}, expected: 2047")

        # Check layers
        if self.max_layer != 10:
            errors.append(f"Max layer: {self.max_layer}, expected: 10")

        # Check each layer count
        for layer in range(11):
            count = len(self.get_nodes_by_layer(layer))
            expected = 2 ** layer
            if count != expected:
                errors.append(f"Layer {layer}: {count} nodes, expected: {expected}")

        # Check binary tree property
        for node in self._nodes.values():
            if not node.is_leaf and len(node.children) != 2:
                errors.append(
                    f"Node {node.node_id}: {len(node.children)} children, expected: 2"
                )

        return len(errors) == 0, errors

    def get_all_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics from all nodes.

        Returns a dict containing:
        - total_nodes: total number of nodes
        - max_layer: maximum layer depth
        - leaf_nodes: number of leaf nodes
        - layer_stats: per-layer statistics
        """
        all_metrics = {
            "total_nodes": self.total_nodes,
            "max_layer": self.max_layer,
            "leaf_nodes": len(self.get_leaf_nodes()),
            "layer_stats": self.get_layer_stats(),
        }
        return all_metrics
