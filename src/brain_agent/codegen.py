"""Code generator for brain simulation nodes.

Reads node definitions from the filesystem (DESIGN.md files) and generates
Python source code for L04-L10 node implementations.
"""

import os
import re
import json
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


# Processing logic templates per region type.
# Templates override _process_* hooks (not process()) to use the base class
# template method pattern. Random weights use a deterministic seed based on
# node_id to ensure reproducibility.
PROCESS_TEMPLATES = {
    "sensory": '''    def _process_tensor(self, data: BrainData) -> BrainData:
        """{responsibility}"""
        arr = data.data.copy()
        kernel = self.get_state("kernel")
        if kernel is None:
            h, w = arr.shape[1], arr.shape[2]
            kh, kw = min(3, h), min(3, w)
            rng = np.random.RandomState(self.node_id)
            kernel = rng.randn(kh, kw).astype(np.float32) * 0.1
            self.set_state("kernel", kernel)
        result = arr.copy()
        for c in range(arr.shape[0]):
            for i in range(arr.shape[1] - kernel.shape[0] + 1):
                for j in range(arr.shape[2] - kernel.shape[1] + 1):
                    result[c, i, j] = np.sum(
                        arr[c, i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel
                    )
        return Tensor(data=result, shape=result.shape)

    def _process_signal(self, data: BrainData) -> BrainData:
        return Signal(
            value=data.value * self.get_state("gain", 1.0),
            timestamp=data.timestamp,
        )''',

    "motor": '''    def _process_tensor(self, data: BrainData) -> BrainData:
        """{responsibility}"""
        arr = data.data.copy()
        gain = self.get_state("gain", 1.0)
        threshold = self.get_state("threshold", 0.0)
        result = np.maximum(arr * gain - threshold, 0)
        return Tensor(data=result, shape=result.shape)

    def _process_signal(self, data: BrainData) -> BrainData:
        return Signal(
            value=max(data.value * self.get_state("gain", 1.0), 0.0),
            timestamp=data.timestamp,
        )''',

    "memory": '''    def _process_state_vector(self, data: BrainData) -> BrainData:
        """{responsibility}"""
        arr = data.data.copy()
        weights = self.get_state("weights")
        if weights is None:
            dim = len(arr)
            rng = np.random.RandomState(self.node_id)
            weights = rng.randn(dim, dim).astype(np.float32) * 0.01
            self.set_state("weights", weights)
        result = arr @ weights
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm * np.linalg.norm(arr)
        return StateVector(data=result.astype(np.float32), dimension=len(result))

    def _process_tensor(self, data: BrainData) -> BrainData:
        arr = data.data.copy()
        return Tensor(data=arr * self.get_state("gain", 1.0), shape=arr.shape)''',

    "modulation": '''    def _process_signal(self, data: BrainData) -> BrainData:
        """{responsibility}"""
        level = self.get_state("mod_level", 0.5)
        gain = 4.0 * level * (1.0 - level)
        return Signal(value=data.value * gain, timestamp=data.timestamp)

    def _process_tensor(self, data: BrainData) -> BrainData:
        arr = data.data.copy()
        level = self.get_state("mod_level", 0.5)
        gain = 4.0 * level * (1.0 - level)
        return Tensor(data=arr * gain, shape=arr.shape)''',

    "relay": '''    def _process_tensor(self, data: BrainData) -> BrainData:
        """{responsibility}"""
        gate_open = self.get_state("gate_open", True)
        if not gate_open:
            return Tensor.zeros(data.data.shape)
        gain = self.get_state("relay_gain", 1.0)
        return Tensor(data=data.data * gain, shape=data.data.shape)

    def _process_signal(self, data: BrainData) -> BrainData:
        gate_open = self.get_state("gate_open", True)
        if not gate_open:
            return Signal(value=0.0, timestamp=data.timestamp)
        gain = self.get_state("relay_gain", 1.0)
        return Signal(value=data.value * gain, timestamp=data.timestamp)''',
}


class NodeCodeGenerator:
    """Generate Python source code for brain nodes from DESIGN.md definitions.

    Walks the DESIGN.md directory tree, parses each file, and generates
    a Python class for each L04-L10 node. Classes extend BrainNode and
    implement process() with region-type-specific logic.
    """

    def __init__(self, design_root: str = "", output_dir: str = "",
                 json_path: str = ""):
        """Initialize the code generator.

        Args:
            design_root: Path to the root directory containing DESIGN.md files
            output_dir: Directory where generated Python files will be written
            json_path: Path to a JSON file with node definitions (alternative to design_root)
        """
        self.design_root = design_root
        self.json_path = json_path
        self.output_dir = output_dir
        self._node_data: List[Dict] = []

    def load(self) -> None:
        """Load node definitions by parsing all DESIGN.md files."""
        from .tree import DesignDocParser

        root_path = Path(self.design_root)
        if not root_path.exists():
            raise FileNotFoundError(f"Design root does not exist: {self.design_root}")

        for dirpath, _dirnames, filenames in os.walk(root_path):
            if "DESIGN.md" in filenames:
                filepath = os.path.join(dirpath, "DESIGN.md")
                try:
                    metadata = DesignDocParser.parse(filepath)
                    self._node_data.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to parse {filepath}: {e}")

        logger.info(f"Loaded {len(self._node_data)} node definitions from DESIGN.md files")

    def load_from_json(self, json_path: str) -> None:
        """Load node definitions from a JSON file.

        The JSON file should have the format:
        {"nodes": [{"id": ..., "folder_name": ..., "layer": ..., ...}, ...]}

        Args:
            json_path: Path to the JSON file
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        raw_nodes = data.get("nodes", [])
        for node_data in raw_nodes:
            # Normalize JSON keys to match the internal format
            self._node_data.append({
                "node_id": node_data.get("id", node_data.get("node_id", 0)),
                "name_cn": node_data.get("name_cn", ""),
                "short_name": node_data.get("short_name", ""),
                "full_name": node_data.get("full_name", ""),
                "folder_name": node_data.get("folder_name", ""),
                "layer": node_data.get("layer", 0),
                "parent_id": node_data.get("parent_id"),
                "parent_name": "",
                "responsibility": node_data.get("responsibility", ""),
                "region_type": node_data.get("region_type", "unknown"),
                "tensor_shape": node_data.get("tensor_shape", ""),
                "frequency": node_data.get("frequency", ""),
                "protocol": node_data.get("protocol", "ASYNC_PUSH"),
                "algorithm": node_data.get("algorithm", ""),
                "is_leaf": node_data.get("is_leaf", False),
                "children_count": node_data.get("children_count", 0),
            })

        logger.info(f"Loaded {len(self._node_data)} node definitions from {json_path}")

    def generate_all(self) -> str:
        """Generate code for all L04-L10 nodes.

        Returns:
            Path to the generated output file
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Collect L04-L10 nodes
        generated_classes = []
        node_map_entries = []

        for node in self._node_data:
            layer = node["layer"]
            if layer >= 4:
                class_name = self._make_class_name(node)
                region_type = node["region_type"]
                if isinstance(region_type, str):
                    region_type_str = region_type
                else:
                    region_type_str = region_type.value if hasattr(region_type, 'value') else str(region_type)

                responsibility = node.get("responsibility", "")
                # Clean responsibility for docstring (remove newlines, truncate)
                responsibility_clean = responsibility.replace("\n", " ").strip()[:200]

                template = PROCESS_TEMPLATES.get(
                    region_type_str, PROCESS_TEMPLATES["relay"]
                )
                process_body = template.format(responsibility=responsibility_clean)

                node_id = node["node_id"]
                name_cn = node.get("name_cn", "")
                folder_name = node.get("folder_name", "")

                class_code = (
                    f'class {class_name}(BrainNode):\n'
                    f'    """Node {node_id}: {name_cn}\n'
                    f'\n'
                    f'    Folder: {folder_name}\n'
                    f'    Layer: L{layer:02d}\n'
                    f'    Region: {region_type_str}\n'
                    f'    Responsibility: {responsibility_clean}\n'
                    f'    """\n'
                    f'{process_body}\n'
                )
                generated_classes.append(class_code)
                node_map_entries.append((node_id, class_name))

        # Write to a single generated_nodes.py file
        output_path = os.path.join(self.output_dir, "generated_nodes.py")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self._module_header())
            for cls in generated_classes:
                f.write(cls)
                f.write("\n\n")
            f.write(self._module_footer(node_map_entries))

        logger.info(
            f"Generated {len(generated_classes)} node classes to {output_path}"
        )
        return output_path

    def _generate_node_class(self, node: Dict) -> str:
        """Generate a single node class source code.

        Args:
            node: Node metadata dict

        Returns:
            Python source code string for the class
        """
        class_name = self._make_class_name(node)
        region_type = node["region_type"]
        if isinstance(region_type, str):
            region_type_str = region_type
        else:
            region_type_str = region_type.value if hasattr(region_type, 'value') else str(region_type)

        responsibility = node.get("responsibility", "")
        responsibility_clean = responsibility.replace("\n", " ").strip()[:200]

        template = PROCESS_TEMPLATES.get(region_type_str, PROCESS_TEMPLATES["relay"])
        process_body = template.format(responsibility=responsibility_clean)

        node_id = node["node_id"]
        name_cn = node.get("name_cn", "")
        folder_name = node.get("folder_name", "")
        layer = node["layer"]

        return (
            f'class {class_name}(BrainNode):\n'
            f'    """Node {node_id}: {name_cn}\n'
            f'\n'
            f'    Folder: {folder_name}\n'
            f'    Layer: L{layer:02d}\n'
            f'    Region: {region_type_str}\n'
            f'    Responsibility: {responsibility_clean}\n'
            f'    """\n'
            f'{process_body}\n'
        )

    def _make_class_name(self, node: Dict) -> str:
        """Convert folder name to a unique valid Python class name.

        The class name includes the node_id to guarantee uniqueness
        across all 2032 generated nodes.

        Examples:
            L04_Visual--OnChannelDetector (id=15) -> N15OnChannelDetectorNode
            L05_OnChannel--HighFreqFilter (id=30) -> N30HighFreqFilterNode

        Args:
            node: Node metadata dict

        Returns:
            Valid unique Python class name with 'Node' suffix
        """
        folder = node.get("folder_name", "")
        # Remove layer prefix (e.g. 'L04_')
        parts = folder.split('_', 1)
        if len(parts) >= 2:
            name = parts[1]
        else:
            name = folder

        # If name contains '--', take the part after the last '--'
        if '--' in name:
            name = name.split('--')[-1]

        # Clean up: remove non-alphanumeric characters
        name = re.sub(r'[^a-zA-Z0-9]', '', name)

        # Ensure the name is not empty
        if not name:
            name = f"Node{node.get('node_id', 0)}"

        # Prefix with node_id to guarantee uniqueness
        node_id = node.get("node_id", 0)
        return f"N{node_id}{name}Node"

    def _module_header(self) -> str:
        """Generate the module header with imports."""
        return (
            '"""Auto-generated node implementations for L04-L10.\n'
            '\n'
            'DO NOT EDIT MANUALLY - generated by codegen.py\n'
            '"""\n'
            'import numpy as np\n'
            '\n'
            'from ..types import BrainData, Tensor, StateVector, Signal\n'
            'from ..node import BrainNode\n'
            '\n\n'
        )

    def _module_footer(self, node_map_entries: List) -> str:
        """Generate the module footer with GENERATED_NODE_MAP."""
        lines = ["# Node ID to class mapping\n", "GENERATED_NODE_MAP = {\n"]
        for node_id, class_name in node_map_entries:
            lines.append(f"    {node_id}: {class_name},\n")
        lines.append("}\n")
        return "".join(lines)
