"""Tests for per-node result access (Stage 1 quality)."""

import pytest
from brain_agent import Simulation, Signal
from brain_agent.testing import build_small_tree, make_test_signal


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


class TestCaptureOutputs:
    """Test capture_outputs parameter."""

    def test_capture_outputs_disabled_by_default(self):
        sim = Simulation.from_json(_get_json_path())
        assert sim.scheduler._capture_outputs is False

    def test_capture_outputs_enabled(self):
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        assert sim.scheduler._capture_outputs is True

    def test_step_result_has_empty_outputs_when_disabled(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3, input_data=Signal(value=1.0))
        for result in sim.results:
            assert result.node_outputs == {}
            assert result.node_latencies == {}

    def test_step_result_has_outputs_when_enabled(self):
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=3, input_data=Signal(value=1.0))
        for result in sim.results:
            assert len(result.node_outputs) > 0
            assert len(result.node_latencies) > 0

    def test_node_outputs_contain_brain_data(self):
        from brain_agent.types import BrainData
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=1, input_data=Signal(value=1.0))
        result = sim.results[0]
        for nid, output in result.node_outputs.items():
            assert isinstance(output, BrainData)

    def test_node_latencies_are_floats(self):
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=1, input_data=Signal(value=1.0))
        result = sim.results[0]
        for nid, latency in result.node_latencies.items():
            assert isinstance(latency, float)
            assert latency >= 0.0

    def test_captured_node_ids_match_processed_nodes(self):
        """Node IDs in outputs should match nodes_processed count."""
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=1, input_data=Signal(value=1.0))
        result = sim.results[0]
        assert len(result.node_outputs) <= result.nodes_processed


class TestGetNodeResult:
    """Test get_node_result() method."""

    def test_get_node_result_valid(self):
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=5, input_data=Signal(value=1.0))
        # Find a node that was actually processed in step 0
        step0_outputs = sim.results[0].node_outputs
        assert len(step0_outputs) > 0
        first_node_id = next(iter(step0_outputs))
        result = sim.get_node_result(node_id=first_node_id, step=0)
        assert result is not None

    def test_get_node_result_nonexistent_node(self):
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=3, input_data=Signal(value=1.0))
        result = sim.get_node_result(node_id=99999, step=0)
        assert result is None

    def test_get_node_result_out_of_range_step(self):
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=3, input_data=Signal(value=1.0))
        with pytest.raises(IndexError, match="out of range"):
            sim.get_node_result(node_id=0, step=100)

    def test_get_node_result_negative_step(self):
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=3, input_data=Signal(value=1.0))
        with pytest.raises(IndexError, match="out of range"):
            sim.get_node_result(node_id=0, step=-1)

    def test_get_node_result_without_capture_raises(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3, input_data=Signal(value=1.0))
        with pytest.raises(RuntimeError, match="capture_outputs"):
            sim.get_node_result(node_id=0, step=0)


class TestGetNodeLatency:
    """Test get_node_latency() method."""

    def test_get_node_latency_valid(self):
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=5, input_data=Signal(value=1.0))
        # Find a node that was actually processed
        step0_latencies = sim.results[0].node_latencies
        assert len(step0_latencies) > 0
        first_node_id = next(iter(step0_latencies))
        latency = sim.get_node_latency(node_id=first_node_id, step=0)
        assert latency is not None
        assert isinstance(latency, float)
        assert latency >= 0.0

    def test_get_node_latency_without_capture_raises(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3, input_data=Signal(value=1.0))
        with pytest.raises(RuntimeError, match="capture_outputs"):
            sim.get_node_latency(node_id=0, step=0)


class TestToDataFrame:
    """Test to_dataframe() method."""

    def test_to_dataframe_without_capture_raises(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3, input_data=Signal(value=1.0))
        with pytest.raises(RuntimeError, match="capture_outputs"):
            sim.to_dataframe()

    def test_to_dataframe_without_pandas_raises(self):
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=3, input_data=Signal(value=1.0))
        # Mock pandas not being available
        import sys
        old_pandas = sys.modules.get("pandas")
        sys.modules["pandas"] = None
        try:
            with pytest.raises(RuntimeError, match="pandas"):
                sim.to_dataframe()
        finally:
            if old_pandas is not None:
                sys.modules["pandas"] = old_pandas
            else:
                del sys.modules["pandas"]

    def test_to_dataframe_with_pandas(self):
        pytest.importorskip("pandas")
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=5, input_data=Signal(value=1.0))
        df = sim.to_dataframe()
        assert len(df) > 0
        assert "step" in df.columns
        assert "node_id" in df.columns
        assert "latency_ms" in df.columns
        assert "output_type" in df.columns

    def test_to_dataframe_filtered_by_node_ids(self):
        pytest.importorskip("pandas")
        sim = Simulation.from_json(_get_json_path(), capture_outputs=True)
        sim.run(steps=5, input_data=Signal(value=1.0))
        # Get a node ID that actually has output
        first_node_id = next(iter(sim.results[0].node_outputs))
        df = sim.to_dataframe(node_ids=[first_node_id])
        assert len(df) > 0
        assert set(df["node_id"].unique()) == {first_node_id}
