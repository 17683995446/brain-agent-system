"""Checkpoint save/restore for simulation state.

Provides functions to serialize and deserialize the full simulation state
(scheduler step/time + all node internal states) to/from JSON files.

Used by :meth:`Simulation.save_checkpoint` and
:meth:`Simulation.load_checkpoint`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .api import Simulation

logger = logging.getLogger(__name__)


def save_checkpoint(sim: "Simulation", path: str) -> None:
    """Save simulation state to a checkpoint file.

    Serializes the scheduler step/time, all node internal states,
    and recent results count to a JSON file.

    Args:
        sim: The Simulation instance to save.
        path: Output file path (JSON format).
    """
    # Collect node states
    node_states = {}
    for node in sim.tree.bfs_traverse():
        state = {}
        for key, value in node.state.items():
            state[key] = _serialize_value(value)
        node_states[node.node_id] = state

    checkpoint = {
        "version": _get_version(),
        "step": sim.scheduler.step_count,
        "sim_time": sim.scheduler.sim_time,
        "node_states": node_states,
        "results_count": len(sim.results),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

    logger.info(f"Checkpoint saved to {path} (step={checkpoint['step']})")


def load_checkpoint(sim: "Simulation", path: str) -> None:
    """Load simulation state from a checkpoint file.

    Restores scheduler step/time and all node internal states.
    Previous results are cleared.

    Args:
        sim: The Simulation instance to restore into.
        path: Checkpoint file path (JSON format).

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    # Restore scheduler state
    sim.scheduler.set_state(
        step=checkpoint["step"],
        sim_time=checkpoint["sim_time"],
    )

    # Restore node states (JSON keys are strings, convert to int)
    node_states = checkpoint.get("node_states", {})
    node_states_int = {int(k): v for k, v in node_states.items()}
    for node in sim.tree.bfs_traverse():
        if node.node_id in node_states_int:
            for key, value in node_states_int[node.node_id].items():
                node.state[key] = _deserialize_value(value)

    # Clear previous results
    sim._results.clear()

    logger.info(
        f"Checkpoint loaded from {path} "
        f"(step={checkpoint['step']}, sim_time={checkpoint['sim_time']:.4f})"
    )


def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON, handling numpy types recursively."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "data": value.tolist(),
                "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return {"__npscalar__": True, "data": value.item(),
                "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    return value


def _deserialize_value(value: Any) -> Any:
    """Deserialize a value from JSON, restoring numpy types."""
    import numpy as np
    if isinstance(value, dict):
        if value.get("__ndarray__"):
            return np.array(value["data"], dtype=value.get("dtype", "float32"))
        if value.get("__npscalar__"):
            return np.array(value["data"], dtype=value.get("dtype", "float32")).item()
    return value


def _get_version() -> str:
    """Get package version (avoids circular import)."""
    try:
        from importlib.metadata import version as _get_version
        return _get_version("brain-agent-system")
    except Exception:
        return "0.5.0"
