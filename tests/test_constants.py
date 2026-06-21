"""Tests for centralized constants (constants.py).

Verifies that constants are internally consistent and match the
actual system structure (e.g. TOTAL_NODES == 2^11 - 1).
"""

import pytest
from brain_agent import constants as C


class TestTreeStructureConstants:
    """Verify tree structure constants are internally consistent."""

    def test_total_nodes_is_full_binary_tree(self):
        """TOTAL_NODES should equal 2^(MAX_LAYER+1) - 1."""
        expected = 2 ** (C.MAX_LAYER + 1) - 1
        assert C.TOTAL_NODES == expected

    def test_total_nodes_value(self):
        assert C.TOTAL_NODES == 2047

    def test_max_layer_value(self):
        assert C.MAX_LAYER == 10

    def test_manual_plus_generated_equals_total(self):
        """MANUAL_NODE_COUNT + GENERATED_NODE_COUNT == TOTAL_NODES."""
        assert C.MANUAL_NODE_COUNT + C.GENERATED_NODE_COUNT == C.TOTAL_NODES

    def test_manual_node_count(self):
        """L00(1) + L01(2) + L02(4) + L03(8) = 15."""
        assert C.MANUAL_NODE_COUNT == 15

    def test_generated_node_count(self):
        assert C.GENERATED_NODE_COUNT == 2032

    def test_children_per_node(self):
        """Binary tree: each non-leaf has exactly 2 children."""
        assert C.CHILDREN_PER_NODE == 2

    def test_manual_layers(self):
        """L00-L03 = 4 layers."""
        assert C.MANUAL_LAYERS == 4


class TestSchedulingConstants:
    """Verify scheduling frequency constants."""

    def test_dt_fast_value(self):
        assert C.DEFAULT_DT_FAST == 0.005

    def test_dt_medium_value(self):
        assert C.DEFAULT_DT_MEDIUM == 0.02

    def test_dt_slow_value(self):
        assert C.DEFAULT_DT_SLOW == 0.1

    def test_dt_ordering(self):
        """dt_fast < dt_medium < dt_slow."""
        assert C.DEFAULT_DT_FAST < C.DEFAULT_DT_MEDIUM < C.DEFAULT_DT_SLOW

    def test_max_latency_positive(self):
        assert C.DEFAULT_MAX_LATENCY_MS > 0


class TestDataDimensionConstants:
    """Verify data dimension constants."""

    def test_state_dim_positive(self):
        assert C.DEFAULT_STATE_DIM > 0

    def test_state_dim_power_of_2(self):
        """128 is a power of 2 (common for neural network dimensions)."""
        assert (C.DEFAULT_STATE_DIM & (C.DEFAULT_STATE_DIM - 1)) == 0

    def test_tensor_shapes_are_tuples(self):
        for shape in [C.DEFAULT_TENSOR_SHAPE_FAST,
                      C.DEFAULT_TENSOR_SHAPE_MEDIUM,
                      C.DEFAULT_TENSOR_SHAPE]:
            assert isinstance(shape, tuple)
            assert len(shape) == 4  # (batch, channel, height, width)


class TestOtherConstants:
    """Verify queue and monitoring constants."""

    def test_max_queue_size_positive(self):
        assert C.DEFAULT_MAX_QUEUE_SIZE > 0

    def test_metrics_port_in_valid_range(self):
        """Port should be in valid TCP range (1024-65535)."""
        assert 1024 <= C.DEFAULT_METRICS_PORT <= 65535


class TestConstantsUsedInCode:
    """Verify that constants are actually used by the modules that should use them."""

    def test_scheduler_uses_dt_constants(self):
        from brain_agent.scheduler import SchedulerConfig
        config = SchedulerConfig()
        assert config.dt_fast == C.DEFAULT_DT_FAST
        assert config.dt_medium == C.DEFAULT_DT_MEDIUM
        assert config.dt_slow == C.DEFAULT_DT_SLOW
        assert config.max_latency_ms == C.DEFAULT_MAX_LATENCY_MS

    def test_node_uses_queue_size_constant(self):
        from brain_agent.nodes.brain import BrainSystemNode
        from brain_agent.node import NodeConfig, RegionType, Protocol
        config = NodeConfig(
            node_id=99999,
            name_cn="测试",
            short_name="Test",
            folder_name="L00_Test",
            layer=0,
            parent_id=None,
            responsibility="Testing",
            region_type=RegionType.SENSORY,
            tensor_shape="1,1,8,8",
            frequency="fast",
            protocol=Protocol.ASYNC_PUSH,
            algorithm="test",
        )
        node = BrainSystemNode(config)
        assert node._input_queue.maxlen == C.DEFAULT_MAX_QUEUE_SIZE
