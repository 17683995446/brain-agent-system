"""Concrete node implementations for the brain simulation agent system.

Exports all 15 core brain nodes (L00-L03) and provides NODE_CLASS_MAP
for indexed access to node classes.
"""

from .brain import BrainSystemNode
from .sensorimotor import SensorimotorSystemNode
from .cognitive import CognitiveSystemNode
from .sensory import SensoryPathwayNode
from .motor import MotorPathwayNode
from .memory import MemorySystemNode
from .modulation import ModulationSystemNode
from .visual import VisualPathwayNode
from .auditory import AuditoryPathwayNode
from .somatosensory import SomatosensoryPathwayNode
from .thalamic import ThalamicRelayStationNode
from .hippocampal import HippocampalCircuitNode
from .amygdala import AmygdalaCircuitNode
from .dopamine import DopamineSystemNode
from .noradrenaline import NoradrenalineSystemNode

__all__ = [
    "BrainSystemNode",
    "SensorimotorSystemNode",
    "CognitiveSystemNode",
    "SensoryPathwayNode",
    "MotorPathwayNode",
    "MemorySystemNode",
    "ModulationSystemNode",
    "VisualPathwayNode",
    "AuditoryPathwayNode",
    "SomatosensoryPathwayNode",
    "ThalamicRelayStationNode",
    "HippocampalCircuitNode",
    "AmygdalaCircuitNode",
    "DopamineSystemNode",
    "NoradrenalineSystemNode",
    "NODE_CLASS_MAP",
]

NODE_CLASS_MAP = {
    0: BrainSystemNode,
    1: SensorimotorSystemNode,
    2: CognitiveSystemNode,
    3: SensoryPathwayNode,
    4: MotorPathwayNode,
    5: MemorySystemNode,
    6: ModulationSystemNode,
    7: VisualPathwayNode,
    8: AuditoryPathwayNode,
    9: SomatosensoryPathwayNode,
    10: ThalamicRelayStationNode,
    11: HippocampalCircuitNode,
    12: AmygdalaCircuitNode,
    13: DopamineSystemNode,
    14: NoradrenalineSystemNode,
}
