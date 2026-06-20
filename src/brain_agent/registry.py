"""Node registry: unified mapping of node_id -> node class for all 2047 nodes.

Combines L00-L03 manually implemented nodes with L04-L10 auto-generated nodes.
"""

from typing import Dict, Type, Optional
from .node import BrainNode, NodeConfig
from .nodes import NODE_CLASS_MAP  # L00-L03 (15 nodes)
from .generated import GENERATED_NODE_MAP  # L04-L10 (2032 nodes)

# Unified registry: node_id -> class
ALL_NODE_CLASSES: Dict[int, Type[BrainNode]] = {}
ALL_NODE_CLASSES.update(NODE_CLASS_MAP)
ALL_NODE_CLASSES.update(GENERATED_NODE_MAP)


def get_node_class(node_id: int) -> Optional[Type[BrainNode]]:
    """Get the node class for a given node ID."""
    return ALL_NODE_CLASSES.get(node_id)


def create_node(config: NodeConfig) -> BrainNode:
    """Create a node instance from its config, using the registry."""
    cls = get_node_class(config.node_id)
    if cls is None:
        raise ValueError(
            f"No node class registered for node_id={config.node_id}. "
            f"Register custom nodes via register_node(node_id, cls) or add to NODE_CLASS_MAP."
        )
    return cls(config)


def get_registry_stats() -> Dict[str, int]:
    """Get statistics about the node registry."""
    manual = set(NODE_CLASS_MAP.keys())
    generated = set(GENERATED_NODE_MAP.keys())
    return {
        "total": len(ALL_NODE_CLASSES),
        "manual": len(manual),
        "generated": len(generated),
        "overlap": len(manual & generated),
    }
