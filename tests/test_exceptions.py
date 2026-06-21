"""Tests for the custom exception hierarchy (Stage 1 order)."""

import pytest
from brain_agent import (
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
from brain_agent import Simulation, Signal, SchedulerConfig


def _get_json_path():
    import os
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "brain_tree_v5.json"),
        "/workspace/BrainAgentSystem/data/brain_tree_v5.json",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    pytest.skip("brain_tree_v5.json not found")


class TestExceptionHierarchy:
    """Test that all custom exceptions inherit from BrainAgentError."""

    def test_all_inherit_from_brain_agent_error(self):
        exceptions = [
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
        ]
        for exc_cls in exceptions:
            assert issubclass(exc_cls, BrainAgentError), (
                f"{exc_cls.__name__} should inherit from BrainAgentError"
            )

    def test_config_validation_error_is_value_error(self):
        assert issubclass(ConfigValidationError, ValueError)

    def test_node_validation_error_is_value_error(self):
        assert issubclass(NodeValidationError, ValueError)

    def test_scheduler_config_error_is_value_error(self):
        assert issubclass(SchedulerConfigError, ValueError)

    def test_scheduler_config_error_is_scheduler_error(self):
        assert issubclass(SchedulerConfigError, SchedulerError)

    def test_serialization_error_is_value_error(self):
        assert issubclass(SerializationError, ValueError)

    def test_input_error_is_value_error(self):
        assert issubclass(InputError, ValueError)

    def test_export_error_is_value_error(self):
        assert issubclass(ExportError, ValueError)

    def test_checkpoint_error_is_runtime_error(self):
        assert issubclass(CheckpointError, RuntimeError)

    def test_tree_validation_error_is_value_error(self):
        assert issubclass(TreeValidationError, ValueError)


class TestExceptionUsage:
    """Test that exceptions are raised correctly in real code paths."""

    def test_scheduler_config_error_on_invalid_dt(self):
        with pytest.raises(SchedulerConfigError):
            SchedulerConfig(dt_fast=-1.0)

    def test_scheduler_config_error_caught_as_value_error(self):
        """Backward compatibility: SchedulerConfigError is also ValueError."""
        with pytest.raises(ValueError):
            SchedulerConfig(dt_fast=-1.0)

    def test_scheduler_config_error_caught_as_brain_agent_error(self):
        """SchedulerConfigError is also BrainAgentError."""
        with pytest.raises(BrainAgentError):
            SchedulerConfig(dt_fast=-1.0)

    def test_checkpoint_error_on_get_node_result_without_capture(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=1, input_data=Signal(value=1.0))
        with pytest.raises(CheckpointError):
            sim.get_node_result(node_id=0, step=0)

    def test_checkpoint_error_caught_as_runtime_error(self):
        """Backward compatibility: CheckpointError is also RuntimeError."""
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=1, input_data=Signal(value=1.0))
        with pytest.raises(RuntimeError):
            sim.get_node_result(node_id=0, step=0)

    def test_checkpoint_error_caught_as_brain_agent_error(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=1, input_data=Signal(value=1.0))
        with pytest.raises(BrainAgentError):
            sim.get_node_result(node_id=0, step=0)

    def test_serialization_error_on_bad_tensor(self):
        import numpy as np
        from brain_agent import Tensor
        with pytest.raises(SerializationError):
            Tensor(data=np.zeros((3, 3)), shape=(2, 2))

    def test_serialization_error_caught_as_value_error(self):
        """Backward compatibility: SerializationError is also ValueError."""
        import numpy as np
        from brain_agent import Tensor
        with pytest.raises(ValueError):
            Tensor(data=np.zeros((3, 3)), shape=(2, 2))


class TestCatchAllPattern:
    """Test that BrainAgentError catches all package errors."""

    def test_catch_all_with_brain_agent_error(self):
        """Users can catch all package errors with one except clause."""
        caught = []
        try:
            SchedulerConfig(dt_fast=-1.0)
        except BrainAgentError as e:
            caught.append(e)

        try:
            Simulation.from_json(_get_json_path()).get_node_result(0, 0)
        except BrainAgentError as e:
            caught.append(e)

        assert len(caught) == 2
