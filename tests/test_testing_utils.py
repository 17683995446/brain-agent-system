"""Tests for brain_agent.testing utilities (Stage 5 usability)."""

import pytest
import numpy as np

from brain_agent.testing import (
    make_node_config,
    MockNode,
    RecordingNode,
    build_small_tree,
    make_test_signal,
    make_test_tensor,
    make_test_state_vector,
)
from brain_agent.types import Signal, Tensor, StateVector


class TestMakeNodeConfig:
    """Test make_node_config() helper."""

    def test_default_config(self):
        config = make_node_config()
        assert config.node_id == 0
        assert config.short_name == "TestNode"
        assert config.layer == 0

    def test_custom_config(self):
        config = make_node_config(
            node_id=5,
            short_name="Custom",
            layer=2,
            parent_id=2,
            region_type="motor",
        )
        assert config.node_id == 5
        assert config.short_name == "Custom"
        assert config.layer == 2
        assert config.parent_id == 2

    def test_all_region_types(self):
        for region in ["sensory", "motor", "memory", "modulation", "relay", "cognitive"]:
            config = make_node_config(region_type=region)
            assert config.region_type is not None

    def test_all_protocols(self):
        for proto in ["async_push", "sync_call", "pub_sub"]:
            config = make_node_config(protocol=proto)
            assert config.protocol is not None


class TestMockNode:
    """Test MockNode test double."""

    def test_mock_node_records_inputs(self):
        config = make_node_config(node_id=0)
        node = MockNode(config)
        signal = make_test_signal(value=1.0)
        node.process(signal)
        assert len(node.processed_inputs) == 1
        assert node.process_count == 1

    def test_mock_node_pass_through_by_default(self):
        config = make_node_config(node_id=0)
        node = MockNode(config)
        signal = make_test_signal(value=0.5)
        result = node.process(signal)
        assert result.value == 0.5

    def test_mock_node_with_transform(self):
        config = make_node_config(node_id=0)
        node = MockNode(config, transform=lambda d: Signal(value=d.value * 2, timestamp=d.timestamp))
        signal = make_test_signal(value=1.0)
        result = node.process(signal)
        assert result.value == 2.0

    def test_mock_node_reset(self):
        config = make_node_config(node_id=0)
        node = MockNode(config)
        node.process(make_test_signal())
        node.process(make_test_signal())
        assert node.process_count == 2
        node.reset_mock()
        assert node.process_count == 0

    def test_mock_node_handles_tensor(self):
        config = make_node_config(node_id=0)
        node = MockNode(config)
        tensor = make_test_tensor(shape=(1, 1, 4, 4))
        result = node.process(tensor)
        assert result is not None
        assert node.process_count == 1


class TestRecordingNode:
    """Test RecordingNode test double."""

    def test_recording_node_records_input_output(self):
        config = make_node_config(node_id=0)
        node = RecordingNode(config)
        signal = make_test_signal(value=1.0)
        result = node.process(signal)
        assert len(node.records) == 1
        input_data, output_data = node.records[0]
        assert input_data.value == 1.0
        assert output_data.value == 1.0

    def test_recording_node_with_custom_fn(self):
        config = make_node_config(node_id=0)
        node = RecordingNode(config, process_fn=lambda d: Signal(value=d.value * 3, timestamp=d.timestamp))
        signal = make_test_signal(value=2.0)
        result = node.process(signal)
        assert result.value == 6.0
        assert len(node.records) == 1


class TestBuildSmallTree:
    """Test build_small_tree() helper."""

    def test_default_tree(self):
        tree = build_small_tree()
        assert tree.total_nodes == 15  # 2^4 - 1

    def test_3_layer_tree(self):
        tree = build_small_tree(n_layers=3)
        assert tree.total_nodes == 7  # 2^3 - 1

    def test_5_layer_tree(self):
        tree = build_small_tree(n_layers=5)
        assert tree.total_nodes == 31  # 2^5 - 1

    def test_invalid_layers_too_few(self):
        with pytest.raises(ValueError, match="1-11"):
            build_small_tree(n_layers=0)

    def test_invalid_layers_too_many(self):
        with pytest.raises(ValueError, match="1-11"):
            build_small_tree(n_layers=12)

    def test_tree_has_root(self):
        tree = build_small_tree(n_layers=3)
        root = tree.get_node(0)
        assert root is not None
        assert root.node_id == 0

    def test_tree_bfs_traverse(self):
        tree = build_small_tree(n_layers=3)
        nodes = list(tree.bfs_traverse())
        assert len(nodes) == 7


class TestTestDataFactories:
    """Test test data factory functions."""

    def test_make_test_signal_default(self):
        signal = make_test_signal()
        assert signal.value == 1.0
        assert signal.timestamp == 0.0

    def test_make_test_signal_custom(self):
        signal = make_test_signal(value=0.5, timestamp=1.5)
        assert signal.value == 0.5
        assert signal.timestamp == 1.5

    def test_make_test_tensor_default(self):
        tensor = make_test_tensor()
        assert tensor.shape == (1, 1, 4, 4)
        assert np.all(tensor.data == 1.0)

    def test_make_test_tensor_custom(self):
        tensor = make_test_tensor(shape=(2, 3), fill_value=0.7)
        assert tensor.shape == (2, 3)
        assert np.all(tensor.data == 0.7)

    def test_make_test_state_vector_default(self):
        sv = make_test_state_vector()
        assert sv.dimension == 16
        assert np.all(sv.data == 0.5)

    def test_make_test_state_vector_custom(self):
        sv = make_test_state_vector(dimension=32, fill_value=0.0)
        assert sv.dimension == 32
        assert np.all(sv.data == 0.0)
