"""Tests for checkpoint/resume mechanism (Stage 3 quality)."""

import json
import os
import pytest
import numpy as np
from brain_agent import Simulation, Signal


def _get_json_path():
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "brain_tree_v5.json"),
        "/workspace/BrainAgentSystem/data/brain_tree_v5.json",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    pytest.skip("brain_tree_v5.json not found")


class TestSaveCheckpoint:
    """Test save_checkpoint() method."""

    def test_save_checkpoint_creates_file(self, tmp_path):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=5, input_data=Signal(value=1.0))
        path = str(tmp_path / "checkpoint.json")
        sim.save_checkpoint(path)
        assert os.path.exists(path)

    def test_checkpoint_contains_required_fields(self, tmp_path):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=5, input_data=Signal(value=1.0))
        path = str(tmp_path / "checkpoint.json")
        sim.save_checkpoint(path)

        with open(path) as f:
            data = json.load(f)
        assert "version" in data
        assert "step" in data
        assert "sim_time" in data
        assert "node_states" in data
        assert "results_count" in data

    def test_checkpoint_step_matches_simulation(self, tmp_path):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=10, input_data=Signal(value=1.0))
        path = str(tmp_path / "checkpoint.json")
        sim.save_checkpoint(path)

        with open(path) as f:
            data = json.load(f)
        assert data["step"] == 10

    def test_checkpoint_serializes_numpy_arrays(self, tmp_path):
        sim = Simulation.from_json(_get_json_path())
        # Set some numpy array state on root node
        root = sim.tree.root
        root.set_state("test_array", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        sim.run(steps=1, input_data=Signal(value=1.0))
        path = str(tmp_path / "checkpoint.json")
        sim.save_checkpoint(path)

        with open(path) as f:
            data = json.load(f)
        root_state = data["node_states"].get("0", {})
        assert "test_array" in root_state
        assert root_state["test_array"]["__ndarray__"] is True


class TestLoadCheckpoint:
    """Test load_checkpoint() method."""

    def test_load_checkpoint_restores_step(self, tmp_path):
        sim1 = Simulation.from_json(_get_json_path())
        sim1.run(steps=10, input_data=Signal(value=1.0))
        path = str(tmp_path / "checkpoint.json")
        sim1.save_checkpoint(path)

        sim2 = Simulation.from_json(_get_json_path())
        assert sim2.step_count == 0
        sim2.load_checkpoint(path)
        assert sim2.step_count == 10

    def test_load_checkpoint_restores_sim_time(self, tmp_path):
        sim1 = Simulation.from_json(_get_json_path())
        sim1.run(steps=10, input_data=Signal(value=1.0))
        path = str(tmp_path / "checkpoint.json")
        sim1.save_checkpoint(path)

        sim2 = Simulation.from_json(_get_json_path())
        sim2.load_checkpoint(path)
        assert sim2.sim_time == pytest.approx(sim1.sim_time)

    def test_load_checkpoint_restores_node_state(self, tmp_path):
        sim1 = Simulation.from_json(_get_json_path())
        sim1.run(steps=1, input_data=Signal(value=1.0))
        # Set state AFTER running to avoid being overwritten by process()
        root = sim1.tree.root
        root.set_state("test_checkpoint_value", 42.0)
        path = str(tmp_path / "checkpoint.json")
        sim1.save_checkpoint(path)

        sim2 = Simulation.from_json(_get_json_path())
        sim2.load_checkpoint(path)
        root2 = sim2.tree.root
        assert root2.get_state("test_checkpoint_value") == 42.0

    def test_load_checkpoint_restores_numpy_array(self, tmp_path):
        sim1 = Simulation.from_json(_get_json_path())
        sim1.run(steps=1, input_data=Signal(value=1.0))
        root = sim1.tree.root
        original_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        root.set_state("test_checkpoint_array", original_array)
        path = str(tmp_path / "checkpoint.json")
        sim1.save_checkpoint(path)

        sim2 = Simulation.from_json(_get_json_path())
        sim2.load_checkpoint(path)
        root2 = sim2.tree.root
        restored = root2.get_state("test_checkpoint_array")
        np.testing.assert_array_equal(restored, original_array)

    def test_load_checkpoint_clears_results(self, tmp_path):
        sim1 = Simulation.from_json(_get_json_path())
        sim1.run(steps=5, input_data=Signal(value=1.0))
        path = str(tmp_path / "checkpoint.json")
        sim1.save_checkpoint(path)

        sim2 = Simulation.from_json(_get_json_path())
        sim2.run(steps=3, input_data=Signal(value=1.0))
        assert len(sim2.results) == 3
        sim2.load_checkpoint(path)
        assert len(sim2.results) == 0

    def test_load_nonexistent_checkpoint_raises(self):
        sim = Simulation.from_json(_get_json_path())
        with pytest.raises(FileNotFoundError):
            sim.load_checkpoint("/nonexistent/path/checkpoint.json")


class TestCheckpointResume:
    """Test full checkpoint/resume workflow."""

    def test_resume_continues_from_saved_step(self, tmp_path):
        # Run first half
        sim1 = Simulation.from_json(_get_json_path())
        sim1.run(steps=5, input_data=Signal(value=1.0))
        path = str(tmp_path / "checkpoint.json")
        sim1.save_checkpoint(path)

        # Resume in a new simulation
        sim2 = Simulation.from_json(_get_json_path())
        sim2.load_checkpoint(path)
        assert sim2.step_count == 5

        # Continue running
        sim2.run(steps=5, input_data=Signal(value=1.0))
        assert sim2.step_count == 10

    def test_roundtrip_preserves_state_for_continuation(self, tmp_path):
        sim1 = Simulation.from_json(_get_json_path())
        sim1.run(steps=3, input_data=Signal(value=1.0))
        path = str(tmp_path / "checkpoint.json")
        sim1.save_checkpoint(path)

        sim2 = Simulation.from_json(_get_json_path())
        sim2.load_checkpoint(path)

        # Both should have the same sim_time
        assert sim1.sim_time == pytest.approx(sim2.sim_time)
