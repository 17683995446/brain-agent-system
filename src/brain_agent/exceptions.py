"""Custom exception hierarchy for BrainAgentSystem.

All exceptions raised by the brain_agent package inherit from
:class:`BrainAgentError`, allowing users to catch any package-specific
error with a single ``except`` clause::

    from brain_agent import BrainAgentError

    try:
        sim.run(steps=100)
    except BrainAgentError as e:
        print(f"Simulation failed: {e}")

Each exception also inherits from the most appropriate built-in type
(e.g. ``ValueError``, ``RuntimeError``) so that existing code catching
built-in exceptions continues to work.

Hierarchy::

    BrainAgentError
    ├── ConfigValidationError (also ValueError)
    ├── NodeValidationError (also ValueError)
    ├── NodeProcessError (also RuntimeError)
    ├── TreeValidationError (also ValueError)
    ├── SchedulerError (also RuntimeError)
    │   └── SchedulerConfigError (also ValueError)
    ├── CheckpointError (also RuntimeError)
    ├── SerializationError (also ValueError)
    ├── InputError (also ValueError)
    └── ExportError (also ValueError)
"""

from __future__ import annotations


class BrainAgentError(Exception):
    """Base exception for all BrainAgentSystem errors."""


# ── Configuration ──────────────────────────────────────────────

class ConfigValidationError(BrainAgentError, ValueError):
    """Raised when configuration values are invalid."""


# ── Node ───────────────────────────────────────────────────────

class NodeValidationError(BrainAgentError, ValueError):
    """Raised when node input or output data fails validation."""


class NodeProcessError(BrainAgentError, RuntimeError):
    """Raised when a node's process() method fails."""


# ── Tree ───────────────────────────────────────────────────────

class TreeValidationError(BrainAgentError, ValueError):
    """Raised when the brain tree structure is invalid."""


# ── Scheduler ──────────────────────────────────────────────────

class SchedulerError(BrainAgentError, RuntimeError):
    """Raised for scheduler runtime errors."""


class SchedulerConfigError(SchedulerError, ValueError):
    """Raised when SchedulerConfig values are invalid."""


# ── Checkpoint / Serialization ─────────────────────────────────

class CheckpointError(BrainAgentError, RuntimeError):
    """Raised when checkpoint save/load fails."""


class SerializationError(BrainAgentError, ValueError):
    """Raised when BrainData serialization/deserialization fails."""


# ── I/O ────────────────────────────────────────────────────────

class InputError(BrainAgentError, ValueError):
    """Raised when input adapter encounters an error."""


class ExportError(BrainAgentError, ValueError):
    """Raised when result export fails."""
