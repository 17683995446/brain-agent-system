"""Centralized constants for BrainAgentSystem.

All magic numbers and configuration defaults that appear in multiple
modules should be defined here and imported, rather than duplicated.
"""

from __future__ import annotations

# ── Tree Structure ─────────────────────────────────────────────

#: Total number of nodes in the full binary tree (2^11 - 1).
TOTAL_NODES: int = 2047

#: Maximum layer index (0-based), i.e. 11 layers total.
MAX_LAYER: int = 10

#: Number of layers with manually implemented nodes (L00-L03).
MANUAL_LAYERS: int = 4

#: Number of manually implemented nodes (L00:1 + L01:2 + L02:4 + L03:8).
MANUAL_NODE_COUNT: int = 15

#: Number of auto-generated nodes (L04-L10).
GENERATED_NODE_COUNT: int = 2032

#: Expected number of children per non-leaf node.
CHILDREN_PER_NODE: int = 2


# ── Frequency / Scheduling ─────────────────────────────────────

#: Fast loop timestep in seconds (5ms = 200Hz nominal, 100Hz effective).
DEFAULT_DT_FAST: float = 0.005

#: Medium loop timestep in seconds (20ms = 50Hz nominal, 20Hz effective).
DEFAULT_DT_MEDIUM: float = 0.02

#: Slow loop timestep in seconds (100ms = 10Hz nominal, 1Hz effective).
DEFAULT_DT_SLOW: float = 0.1

#: Default maximum latency threshold in milliseconds.
DEFAULT_MAX_LATENCY_MS: float = 10.0


# ── Data Dimensions ────────────────────────────────────────────

#: Default state vector dimension (used by memory systems).
DEFAULT_STATE_DIM: int = 128

#: Default tensor shape for fast-band sensory data.
DEFAULT_TENSOR_SHAPE_FAST: tuple = (1, 8, 32, 32)

#: Default tensor shape for medium-band motor data.
DEFAULT_TENSOR_SHAPE_MEDIUM: tuple = (1, 1, 32, 32)

#: Default fallback tensor shape.
DEFAULT_TENSOR_SHAPE: tuple = (1, 1, 8, 8)


# ── Node Queue ─────────────────────────────────────────────────

#: Maximum input queue depth. When full, oldest items are dropped.
DEFAULT_MAX_QUEUE_SIZE: int = 1000


# ── Monitoring ─────────────────────────────────────────────────

#: Default port for the HTTP health/metrics server.
DEFAULT_METRICS_PORT: int = 9090
