"""Tests for the high-level API (Phase 1)."""

import pytest
import tempfile
import os

from brain_agent import (
    Simulation,
    create_simulation,
    run_simulation,
    BrainTreeBuilder,
    FrequencyScheduler,
    Signal,
    Tensor,
    BrainData,
    __version__,
)


def _get_json_path():
    """Find the brain tree JSON file."""
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "brain_tree_v5.json"),
        "/workspace/BrainAgentSystem/data/brain_tree_v5.json",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    pytest.skip("brain_tree_v5.json not found")


class TestPackageInterface:
    """Test that the package exports are correct."""

    def test_version_string(self):
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_all_exports_exist(self):
        """Every name in __all__ should be importable."""
        import brain_agent
        for name in brain_agent.__all__:
            assert hasattr(brain_agent, name), f"Missing export: {name}"

    def test_core_types_importable(self):
        from brain_agent import BrainData, Tensor, StateVector, Signal, Message, SpikeTrain
        assert BrainData is not None
        assert Tensor is not None

    def test_core_classes_importable(self):
        from brain_agent import (
            BrainNode,
            BrainTree,
            BrainTreeBuilder,
            FrequencyScheduler,
            SchedulerConfig,
            StepResult,
            FrequencyBand,
        )
        assert BrainNode is not None
        assert FrequencyScheduler is not None

    def test_high_level_api_importable(self):
        from brain_agent import Simulation, create_simulation, run_simulation
        assert Simulation is not None
        assert callable(create_simulation)
        assert callable(run_simulation)


class TestCreateSimulation:
    """Test the create_simulation factory function."""

    def test_create_with_explicit_path(self):
        json_path = _get_json_path()
        sim = create_simulation(json_path)
        assert sim is not None
        assert sim.tree.total_nodes > 0
        assert sim.scheduler is not None

    def test_create_with_auto_discovery(self):
        sim = create_simulation()
        assert sim is not None
        assert sim.tree.total_nodes > 0

    def test_create_with_custom_config(self):
        from brain_agent import SchedulerConfig
        json_path = _get_json_path()
        config = SchedulerConfig(max_steps=50)
        sim = create_simulation(json_path, config)
        assert sim is not None


class TestRunSimulation:
    """Test the run_simulation one-shot function."""

    def test_run_basic(self):
        json_path = _get_json_path()
        results = run_simulation(json_path, steps=5)
        assert len(results) == 5
        for r in results:
            assert r.nodes_processed > 0

    def test_run_with_input(self):
        json_path = _get_json_path()
        input_data = Signal(value=1.0, timestamp=0.0)
        results = run_simulation(json_path, steps=5, input_data=input_data)
        assert len(results) == 5


class TestSimulationClass:
    """Test the Simulation class."""

    @pytest.fixture
    def sim(self):
        json_path = _get_json_path()
        return Simulation.from_json(json_path)

    def test_from_json(self, sim):
        assert sim.tree.total_nodes > 0
        assert sim.scheduler is not None
        assert sim.metrics is not None
        assert sim.tracer is not None

    def test_run_steps(self, sim):
        results = sim.run(steps=10)
        assert len(results) == 10
        assert sim.step_count == 10
        assert sim.sim_time > 0

    def test_step_single(self, sim):
        result = sim.step()
        assert result.step == 1  # Scheduler increments step before returning
        assert result.nodes_processed > 0
        assert sim.step_count == 1

    def test_step_with_input(self, sim):
        input_data = Signal(value=0.5, timestamp=0.0)
        result = sim.step(input_data=input_data)
        assert result.nodes_processed > 0

    def test_run_with_callback(self, sim):
        def input_cb(step):
            return Signal(value=float(step), timestamp=step * 0.005)

        results = sim.run(steps=5, input_callback=input_cb)
        assert len(results) == 5

    def test_reset(self, sim):
        sim.run(steps=5)
        assert sim.step_count == 5
        sim.reset()
        assert sim.step_count == 0
        assert len(sim.results) == 0

    def test_stats_property(self, sim):
        stats = sim.stats
        assert "total_nodes" in stats
        assert "step" in stats
        assert "band_counts" in stats

    def test_results_property(self, sim):
        assert len(sim.results) == 0
        sim.run(steps=3)
        assert len(sim.results) == 3

    def test_export_metrics(self, sim):
        sim.run(steps=3)
        metrics_str = sim.export_metrics()
        assert isinstance(metrics_str, str)
        assert len(metrics_str) > 0

    def test_export_metrics_invalid_format(self, sim):
        with pytest.raises(ValueError, match="Unsupported"):
            sim.export_metrics(format="xml")

    def test_get_node(self, sim):
        node = sim.get_node(0)
        assert node is not None
        assert node.node_id == 0

    def test_summary(self, sim):
        sim.run(steps=5)
        s = sim.summary()
        assert "Brain Simulation Summary" in s
        assert "Total nodes" in s
        assert "Avg latency" in s

    def test_summary_no_results(self, sim):
        s = sim.summary()
        assert "Brain Simulation Summary" in s
        assert "Total nodes" in s

    def test_run_with_shutdown_checker(self, sim):
        call_count = [0]

        def checker():
            call_count[0] += 1
            return call_count[0] > 3  # Stop after 3 checks

        results = sim.run(steps=100, shutdown_checker=checker)
        assert len(results) < 100  # Should have stopped early
