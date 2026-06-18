"""Tests for all 15 core brain nodes (L00-L03).

For each node, tests:
1. Node creation with correct config
2. process() returns correct data type
3. process() with Tensor input produces valid output
4. process() with StateVector input (where applicable)
5. process() with Signal input (where applicable)
6. Metrics are tracked after processing
"""

import pytest
import numpy as np

from brain_agent.node import BrainNode, NodeConfig, RegionType, Protocol
from brain_agent.types import Tensor, StateVector, Signal, Message, BrainData
from brain_agent.nodes import (
    BrainSystemNode,
    SensorimotorSystemNode,
    CognitiveSystemNode,
    SensoryPathwayNode,
    MotorPathwayNode,
    MemorySystemNode,
    ModulationSystemNode,
    VisualPathwayNode,
    AuditoryPathwayNode,
    SomatosensoryPathwayNode,
    ThalamicRelayStationNode,
    HippocampalCircuitNode,
    AmygdalaCircuitNode,
    DopamineSystemNode,
    NoradrenalineSystemNode,
    NODE_CLASS_MAP,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_config(node_id: int, name_cn: str, short_name: str, folder_name: str,
                layer: int, region_type: RegionType = RegionType.UNKNOWN,
                tensor_shape: str = "32x32x8",
                frequency: str = "10Hz",
                protocol: Protocol = Protocol.ASYNC_PUSH,
                algorithm: str = "integration",
                responsibility: str = "test") -> NodeConfig:
    return NodeConfig(
        node_id=node_id,
        name_cn=name_cn,
        short_name=short_name,
        folder_name=folder_name,
        layer=layer,
        parent_id=None,
        responsibility=responsibility,
        region_type=region_type,
        tensor_shape=tensor_shape,
        frequency=frequency,
        protocol=protocol,
        algorithm=algorithm,
    )


# Shared test data
TENSOR_32x32x8 = Tensor.random((32, 32, 8))
TENSOR_1x32x32 = Tensor.random((1, 32, 32))
TENSOR_32x32 = Tensor.random((32, 32))
STATE_VEC_128 = StateVector.random(128)
SIGNAL_POS = Signal(value=0.7, timestamp=0.0)
SIGNAL_NEG = Signal(value=-0.5, timestamp=0.0)
MSG = Message(topic="test", payload={"key": "val"}, timestamp=0.0)


# ===========================================================================
# L00: BrainSystemNode
# ===========================================================================

class TestBrainSystemNode:
    """Tests for L00 BrainSystemNode (root node: global integration)."""

    def setup_method(self):
        self.config = _make_config(
            0, "大脑系统", "BrainSystem", "L00_BrainSystem",
            layer=0, region_type=RegionType.UNKNOWN,
            responsibility="全局整合与协调",
        )
        self.node = BrainSystemNode(self.config)

    def test_creation(self):
        """Node creation with correct config."""
        assert self.node.node_id == 0
        assert self.node.short_name == "BrainSystem"
        assert self.node.layer == 0
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        """process() with Tensor returns Tensor."""
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        """process() with Tensor input produces valid output shape."""
        result = self.node.process(TENSOR_32x32x8)
        assert result.shape == (32, 32, 8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        """process() with StateVector returns StateVector."""
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        """process() with Signal returns Signal."""
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_process_message_passthrough(self):
        """process() with Message passes through unchanged."""
        result = self.node.process(MSG)
        assert isinstance(result, Message)
        assert result.topic == "test"

    def test_metrics_tracked(self):
        """Metrics are tracked after processing."""
        self.node.process(TENSOR_32x32x8)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1
        assert metrics["node_id"] == 0
        assert metrics["short_name"] == "BrainSystem"
        assert metrics["last_process_time_ms"] >= 0


# ===========================================================================
# L01: SensorimotorSystemNode
# ===========================================================================

class TestSensorimotorSystemNode:
    """Tests for L01 SensorimotorSystemNode (sensory-motor integration)."""

    def setup_method(self):
        self.config = _make_config(
            1, "感觉运动系统", "SensorimotorSystem", "L01_SensorimotorSystem",
            layer=1, region_type=RegionType.SENSORY,
            responsibility="感觉-运动整合",
        )
        self.node = SensorimotorSystemNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 1
        assert self.node.layer == 1
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_32x32x8)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L01: CognitiveSystemNode
# ===========================================================================

class TestCognitiveSystemNode:
    """Tests for L01 CognitiveSystemNode (memory, emotion, regulation)."""

    def setup_method(self):
        self.config = _make_config(
            2, "认知系统", "CognitiveSystem", "L01_CognitiveSystem",
            layer=1, region_type=RegionType.MEMORY,
            responsibility="记忆、情绪、调节",
        )
        self.node = CognitiveSystemNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 2
        assert self.node.layer == 1
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_32x32x8)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L02: SensoryPathwayNode
# ===========================================================================

class TestSensoryPathwayNode:
    """Tests for L02 SensoryPathwayNode (initial feature extraction)."""

    def setup_method(self):
        self.config = _make_config(
            3, "感觉通路", "SensoryPathway", "L02_SensoryPathway",
            layer=2, region_type=RegionType.SENSORY,
            responsibility="初始特征提取",
        )
        self.node = SensoryPathwayNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 3
        assert self.node.layer == 2
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_32x32x8)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L02: MotorPathwayNode
# ===========================================================================

class TestMotorPathwayNode:
    """Tests for L02 MotorPathwayNode (movement planning and execution)."""

    def setup_method(self):
        self.config = _make_config(
            4, "运动通路", "MotorPathway", "L02_MotorPathway",
            layer=2, region_type=RegionType.MOTOR,
            responsibility="运动规划与执行",
        )
        self.node = MotorPathwayNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 4
        assert self.node.layer == 2
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_1x32x32)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_1x32x32)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))
        # Motor output should be non-negative (rectified)
        assert np.all(result.data >= 0)

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_1x32x32)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L02: MemorySystemNode
# ===========================================================================

class TestMemorySystemNode:
    """Tests for L02 MemorySystemNode (encoding, storage, retrieval)."""

    def setup_method(self):
        self.config = _make_config(
            5, "记忆系统", "MemorySystem", "L02_MemorySystem",
            layer=2, region_type=RegionType.MEMORY,
            responsibility="编码、存储、检索",
        )
        self.node = MemorySystemNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 5
        assert self.node.layer == 2
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(STATE_VEC_128)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L02: ModulationSystemNode
# ===========================================================================

class TestModulationSystemNode:
    """Tests for L02 ModulationSystemNode (neuromodulator release)."""

    def setup_method(self):
        self.config = _make_config(
            6, "调制系统", "ModulationSystem", "L02_ModulationSystem",
            layer=2, region_type=RegionType.MODULATION,
            responsibility="神经调制释放",
        )
        self.node = ModulationSystemNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 6
        assert self.node.layer == 2
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(SIGNAL_POS)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L03: VisualPathwayNode
# ===========================================================================

class TestVisualPathwayNode:
    """Tests for L03 VisualPathwayNode (retinal to cortical processing)."""

    def setup_method(self):
        self.config = _make_config(
            7, "视觉通路", "VisualPathway", "L03_VisualPathway",
            layer=3, region_type=RegionType.SENSORY,
            responsibility="视网膜到皮层视觉处理",
        )
        self.node = VisualPathwayNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 7
        assert self.node.layer == 3
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_32x32x8)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L03: AuditoryPathwayNode
# ===========================================================================

class TestAuditoryPathwayNode:
    """Tests for L03 AuditoryPathwayNode (cochlear to cortical processing)."""

    def setup_method(self):
        self.config = _make_config(
            8, "听觉通路", "AuditoryPathway", "L03_AuditoryPathway",
            layer=3, region_type=RegionType.SENSORY,
            responsibility="耳蜗到皮层听觉处理",
        )
        self.node = AuditoryPathwayNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 8
        assert self.node.layer == 3
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_32x32x8)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L03: SomatosensoryPathwayNode
# ===========================================================================

class TestSomatosensoryPathwayNode:
    """Tests for L03 SomatosensoryPathwayNode (touch and proprioception)."""

    def setup_method(self):
        self.config = _make_config(
            9, "体感通路", "SomatosensoryPathway", "L03_SomatosensoryPathway",
            layer=3, region_type=RegionType.SENSORY,
            responsibility="触觉和本体感觉",
        )
        self.node = SomatosensoryPathwayNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 9
        assert self.node.layer == 3
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_32x32)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L03: ThalamicRelayStationNode
# ===========================================================================

class TestThalamicRelayStationNode:
    """Tests for L03 ThalamicRelayStationNode (relay with gating)."""

    def setup_method(self):
        self.config = _make_config(
            10, "丘脑中继站", "ThalamicRelay", "L03_ThalamicRelay",
            layer=3, region_type=RegionType.RELAY,
            responsibility="感觉/运动信号中继与门控",
        )
        self.node = ThalamicRelayStationNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 10
        assert self.node.layer == 3
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_32x32x8)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L03: HippocampalCircuitNode
# ===========================================================================

class TestHippocampalCircuitNode:
    """Tests for L03 HippocampalCircuitNode (episodic memory)."""

    def setup_method(self):
        self.config = _make_config(
            11, "海马回路", "HippocampalCircuit", "L03_HippocampalCircuit",
            layer=3, region_type=RegionType.MEMORY,
            responsibility="情景记忆",
        )
        self.node = HippocampalCircuitNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 11
        assert self.node.layer == 3
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(STATE_VEC_128)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L03: AmygdalaCircuitNode
# ===========================================================================

class TestAmygdalaCircuitNode:
    """Tests for L03 AmygdalaCircuitNode (emotion processing)."""

    def setup_method(self):
        self.config = _make_config(
            12, "杏仁核回路", "AmygdalaCircuit", "L03_AmygdalaCircuit",
            layer=3, region_type=RegionType.MODULATION,
            responsibility="情绪处理",
        )
        self.node = AmygdalaCircuitNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 12
        assert self.node.layer == 3
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_32x32x8)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L03: DopamineSystemNode
# ===========================================================================

class TestDopamineSystemNode:
    """Tests for L03 DopamineSystemNode (reward prediction error)."""

    def setup_method(self):
        self.config = _make_config(
            13, "多巴胺系统", "DopamineSystem", "L03_DopamineSystem",
            layer=3, region_type=RegionType.MODULATION,
            responsibility="奖赏预测误差",
        )
        self.node = DopamineSystemNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 13
        assert self.node.layer == 3
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(SIGNAL_POS)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# L03: NoradrenalineSystemNode
# ===========================================================================

class TestNoradrenalineSystemNode:
    """Tests for L03 NoradrenalineSystemNode (arousal and attention)."""

    def setup_method(self):
        self.config = _make_config(
            14, "去甲肾上腺素系统", "NoradrenalineSystem", "L03_NoradrenalineSystem",
            layer=3, region_type=RegionType.MODULATION,
            responsibility="觉醒与注意",
        )
        self.node = NoradrenalineSystemNode(self.config)

    def test_creation(self):
        assert self.node.node_id == 14
        assert self.node.layer == 3
        assert isinstance(self.node, BrainNode)

    def test_process_returns_correct_type_tensor(self):
        result = self.node.process(TENSOR_32x32x8)
        assert isinstance(result, Tensor)

    def test_process_tensor_valid_output(self):
        result = self.node.process(TENSOR_32x32x8)
        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_process_state_vector(self):
        result = self.node.process(STATE_VEC_128)
        assert isinstance(result, StateVector)
        assert result.dimension == 128

    def test_process_signal(self):
        result = self.node.process(SIGNAL_POS)
        assert isinstance(result, Signal)

    def test_metrics_tracked(self):
        self.node.process(TENSOR_32x32x8)
        metrics = self.node.get_metrics()
        assert metrics["process_count"] == 1


# ===========================================================================
# NODE_CLASS_MAP tests
# ===========================================================================

class TestNodeClassMap:
    """Tests for the NODE_CLASS_MAP export."""

    def test_map_has_15_entries(self):
        assert len(NODE_CLASS_MAP) == 15

    def test_map_keys_are_sequential(self):
        assert list(NODE_CLASS_MAP.keys()) == list(range(15))

    def test_all_values_are_brain_node_subclasses(self):
        for cls in NODE_CLASS_MAP.values():
            assert issubclass(cls, BrainNode)
            assert cls is not BrainNode  # Not the abstract base

    def test_map_includes_all_nodes(self):
        expected = {
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
        assert NODE_CLASS_MAP == expected
