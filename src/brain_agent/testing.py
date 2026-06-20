"""Testing utilities for brain_agent.

Provides helper functions and classes to simplify writing tests for
custom brain nodes and simulation scenarios.

Usage:
    from brain_agent.testing import make_node_config, MockNode, build_small_tree

    # Create a test node config
    config = make_node_config(node_id=0, region_type="sensory")

    # Create a mock node for testing
    node = MockNode(config)

    # Build a small test tree (31 nodes, 5 layers)
    tree = build_small_tree(n_layers=5)
"""

from typing import Optional, List, Dict, Callable

import numpy as np

from .node import BrainNode, NodeConfig, RegionType, Protocol
from .tree import BrainTree
from .types import BrainData, Tensor, StateVector, Signal


def make_node_config(
    node_id: int = 0,
    name_cn: str = "测试节点",
    short_name: str = "TestNode",
    folder_name: str = "test_node",
    layer: int = 0,
    parent_id: Optional[int] = None,
    responsibility: str = "test",
    region_type: str = "sensory",
    tensor_shape: str = "(1,1,8,8)",
    frequency: str = "100",
    protocol: str = "async_push",
    algorithm: str = "test",
) -> NodeConfig:
    """Create a NodeConfig for testing.

    Args:
        node_id: Node ID.
        name_cn: Chinese name.
        short_name: Short name.
        folder_name: Folder name.
        layer: Layer number (0 = root).
        parent_id: Parent node ID (None for root).
        responsibility: Responsibility description.
        region_type: Region type string ("sensory", "motor", "memory",
            "modulation", "relay", "cognitive").
        tensor_shape: Tensor shape string.
        frequency: Frequency string.
        protocol: Protocol string ("async_push", "sync_call", "pub_sub").
        algorithm: Algorithm description.

    Returns:
        A NodeConfig instance suitable for testing.
    """
    region_map = {
        "sensory": RegionType.SENSORY,
        "motor": RegionType.MOTOR,
        "memory": RegionType.MEMORY,
        "modulation": RegionType.MODULATION,
        "relay": RegionType.RELAY,
        "cognitive": RegionType.UNKNOWN,
    }
    protocol_map = {
        "async_push": Protocol.ASYNC_PUSH,
        "sync_call": Protocol.SYNC_CALL,
        "pub_sub": Protocol.PUB_SUB,
    }
    return NodeConfig(
        node_id=node_id,
        name_cn=name_cn,
        short_name=short_name,
        folder_name=folder_name,
        layer=layer,
        parent_id=parent_id,
        responsibility=responsibility,
        region_type=region_map.get(region_type, RegionType.SENSORY),
        tensor_shape=tensor_shape,
        frequency=frequency,
        protocol=protocol_map.get(protocol, Protocol.ASYNC_PUSH),
        algorithm=algorithm,
    )


class MockNode(BrainNode):
    """A mock brain node for testing.

    Records all inputs and outputs, allowing test assertions on what
    data was processed. The process() method simply returns the input
    unchanged (pass-through), or applies a configurable transform.

    Attributes:
        processed_inputs: List of all BrainData received by process().
        process_count: Number of times process() was called.
    """

    def __init__(
        self,
        config: NodeConfig,
        transform: Optional[Callable[[BrainData], BrainData]] = None,
    ) -> None:
        """Initialize mock node.

        Args:
            config: Node configuration.
            transform: Optional callable applied to input data.
                If None, input is returned unchanged.
        """
        super().__init__(config)
        self._transform = transform
        self.processed_inputs: List[BrainData] = []

    def _process_tensor(self, data: BrainData) -> BrainData:
        self.processed_inputs.append(data)
        if self._transform is not None:
            result: BrainData = self._transform(data)
            return result
        return data

    def _process_state_vector(self, data: BrainData) -> BrainData:
        self.processed_inputs.append(data)
        if self._transform is not None:
            result = self._transform(data)
            return result
        return data

    def _process_signal(self, data: BrainData) -> BrainData:
        self.processed_inputs.append(data)
        if self._transform is not None:
            result = self._transform(data)
            return result
        return data

    @property
    def process_count(self) -> int:
        """Number of times process() was called."""
        return len(self.processed_inputs)

    def reset_mock(self) -> None:
        """Clear recorded inputs."""
        self.processed_inputs.clear()


class RecordingNode(BrainNode):
    """A brain node that records all processed data for inspection.

    Unlike MockNode, RecordingNode applies a real processing function
    (default: identity) and stores results alongside inputs.

    Attributes:
        records: List of (input, output) tuples.
    """

    def __init__(
        self,
        config: NodeConfig,
        process_fn: Optional[Callable[[BrainData], BrainData]] = None,
    ) -> None:
        """Initialize recording node.

        Args:
            config: Node configuration.
            process_fn: Optional processing function. Default: identity.
        """
        super().__init__(config)
        self._process_fn = process_fn or (lambda x: x)
        self.records: List[tuple] = []

    def _process_tensor(self, data: BrainData) -> BrainData:
        result: BrainData = self._process_fn(data)
        self.records.append((data, result))
        return result

    def _process_state_vector(self, data: BrainData) -> BrainData:
        result = self._process_fn(data)
        self.records.append((data, result))
        return result

    def _process_signal(self, data: BrainData) -> BrainData:
        result = self._process_fn(data)
        self.records.append((data, result))
        return result


def build_small_tree(n_layers: int = 4) -> BrainTree:
    """Build a small brain tree for testing.

    Creates a complete binary tree with the given number of layers.
    A 4-layer tree has 2^4 - 1 = 15 nodes, much faster than the full
    2047-node tree for unit tests.

    Args:
        n_layers: Number of layers (1-11). Default: 4 (15 nodes).

    Returns:
        A BrainTree with the specified number of layers.

    Example:
        tree = build_small_tree(n_layers=3)  # 7 nodes
        for node in tree.bfs_traverse():
            print(node.node_id, node.short_name)
    """
    if n_layers < 1 or n_layers > 11:
        raise ValueError(
            f"n_layers must be 1-11, got {n_layers}. "
            f"This creates 2^n - 1 nodes."
        )

    from .tree import PassThroughBrainNode

    total_nodes = 2 ** n_layers - 1
    nodes: Dict[int, BrainNode] = {}

    for node_id in range(total_nodes):
        layer = 0
        temp = node_id
        while temp > 0:
            temp = (temp - 1) // 2
            layer += 1

        parent_id = (node_id - 1) // 2 if node_id > 0 else None

        region = "relay"
        if layer == 0:
            region = "cognitive"
        elif layer == 1:
            region = "relay"
        elif layer >= 2:
            region = "sensory"

        config = make_node_config(
            node_id=node_id,
            short_name=f"L{layer}N{node_id}",
            folder_name=f"test_{node_id}",
            layer=layer,
            parent_id=parent_id,
            region_type=region,
        )

        node = PassThroughBrainNode(config)
        nodes[node_id] = node

    # Set up parent-child relationships
    for node_id in range(1, total_nodes):
        parent_id = (node_id - 1) // 2
        nodes[parent_id].connect_child(nodes[node_id])

    return BrainTree(nodes)


def make_test_signal(value: float = 1.0, timestamp: float = 0.0) -> Signal:
    """Create a test Signal with sensible defaults.

    Args:
        value: Signal value.
        timestamp: Timestamp in seconds.

    Returns:
        A Signal instance.
    """
    return Signal(value=value, timestamp=timestamp)


def make_test_tensor(
    shape: tuple = (1, 1, 4, 4),
    fill_value: float = 1.0,
) -> Tensor:
    """Create a test Tensor with sensible defaults.

    Args:
        shape: Tensor shape.
        fill_value: Fill value for all elements.

    Returns:
        A Tensor instance.
    """
    data = np.full(shape, fill_value, dtype=np.float32)
    return Tensor(data=data, shape=shape)


def make_test_state_vector(
    dimension: int = 16,
    fill_value: float = 0.5,
) -> StateVector:
    """Create a test StateVector with sensible defaults.

    Args:
        dimension: Vector dimension.
        fill_value: Fill value for all elements.

    Returns:
        A StateVector instance.
    """
    data = np.full(dimension, fill_value, dtype=np.float32)
    return StateVector(data=data, dimension=dimension)
