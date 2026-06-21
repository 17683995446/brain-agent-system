"""BrainAgentSystem - 大规模脑仿真Agent系统

一个2047节点的全二叉树脑仿真框架，支持多频段调度、三种通信协议、
实时监控和优雅降级。

基本用法:
    from brain_agent import BrainTreeBuilder, FrequencyScheduler

    tree = BrainTreeBuilder().build_full_tree("data/brain_tree_v5.json")
    scheduler = FrequencyScheduler(tree)
    results = scheduler.run(steps=100)

高级用法:
    from brain_agent import Simulation

    sim = Simulation.from_json("data/brain_tree_v5.json")
    results = sim.run(steps=100)
"""

from .types import BrainData, Tensor, StateVector, SpikeTrain, Signal, Message
from .node import BrainNode, NodeConfig, RegionType, Protocol, PortSpec
from .tree import BrainTree, BrainTreeBuilder
from .scheduler import (
    FrequencyScheduler,
    SchedulerConfig,
    StepResult,
    FrequencyBand,
    FrequencySpec,
)
from .monitoring import MetricsCollector, Tracer, TraceSpan
from .config import BrainConfig
from .server import HealthServer
from .lifecycle import GracefulShutdown
from .api import Simulation, create_simulation, run_simulation
from .experiment import Experiment, ExperimentConfig, ParameterSweep
from .io import (
    InputAdapter,
    ListInput,
    CallbackInput,
    ArrayInput,
    FileInput,
    create_input,
    ResultExporter,
    create_exporter,
)
from .exceptions import (
    BrainAgentError,
    ConfigValidationError,
    NodeValidationError,
    NodeProcessError,
    TreeValidationError,
    SchedulerError,
    SchedulerConfigError,
    CheckpointError,
    SerializationError,
    InputError,
    ExportError,
)

# Testing utilities (importable as brain_agent.testing)
from . import testing as testing  # noqa: F401

# Single source of truth: read version from package metadata
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("brain-agent-system")
except Exception:
    __version__ = "0.5.0"  # Fallback for development mode

__all__ = [
    # Version
    "__version__",
    # Types
    "BrainData",
    "Tensor",
    "StateVector",
    "SpikeTrain",
    "Signal",
    "Message",
    # Node
    "BrainNode",
    "NodeConfig",
    "RegionType",
    "Protocol",
    "PortSpec",
    # Tree
    "BrainTree",
    "BrainTreeBuilder",
    # Scheduler
    "FrequencyScheduler",
    "SchedulerConfig",
    "StepResult",
    "FrequencyBand",
    "FrequencySpec",
    # Monitoring
    "MetricsCollector",
    "Tracer",
    "TraceSpan",
    # Config
    "BrainConfig",
    # Server
    "HealthServer",
    # Lifecycle
    "GracefulShutdown",
    # High-level API
    "Simulation",
    "create_simulation",
    "run_simulation",
    # Experiment API
    "Experiment",
    "ExperimentConfig",
    "ParameterSweep",
    # I/O
    "InputAdapter",
    "ListInput",
    "CallbackInput",
    "ArrayInput",
    "FileInput",
    "create_input",
    "ResultExporter",
    "create_exporter",
    # Exceptions
    "BrainAgentError",
    "ConfigValidationError",
    "NodeValidationError",
    "NodeProcessError",
    "TreeValidationError",
    "SchedulerError",
    "SchedulerConfigError",
    "CheckpointError",
    "SerializationError",
    "InputError",
    "ExportError",
]
