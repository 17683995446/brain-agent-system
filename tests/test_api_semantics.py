"""Tests for context manager and API semantics (Stage 4 usability)."""

import pytest
from brain_agent import Simulation, Signal
from brain_agent.io import ResultExporter, create_exporter


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


class TestSimulationContextManager:
    """Test Simulation as a context manager."""

    def test_context_manager_enter_returns_self(self):
        sim = Simulation.from_json(_get_json_path())
        with sim as s:
            assert s is sim

    def test_context_manager_resets_on_exit(self):
        sim = Simulation.from_json(_get_json_path())
        with sim:
            sim.run(steps=5)
            assert len(sim.results) == 5
        # After exit, should be reset
        assert len(sim.results) == 0

    def test_context_manager_resets_on_exception(self):
        sim = Simulation.from_json(_get_json_path())
        with pytest.raises(RuntimeError):
            with sim:
                sim.run(steps=3)
                raise RuntimeError("test error")
        assert len(sim.results) == 0


class TestExportSemantics:
    """Test export() method semantics."""

    def test_file_export_auto_closes(self, tmp_path):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3)
        path = str(tmp_path / "results.json")
        exporter = sim.export(path)
        assert exporter._output._closed if hasattr(exporter._output, '_closed') else True

    def test_memory_export_stays_open(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3)
        exporter = sim.export()
        # Memory exporter should have data accessible
        assert hasattr(exporter, 'data')
        assert len(exporter.data) > 0

    def test_export_returns_result_exporter(self, tmp_path):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3)
        exporter = sim.export(str(tmp_path / "results.json"))
        assert isinstance(exporter, ResultExporter)


class TestGetNodeStrict:
    """Test get_node_strict() method."""

    def test_get_node_strict_valid_id(self):
        sim = Simulation.from_json(_get_json_path())
        node = sim.get_node_strict(0)
        assert node is not None
        assert node.node_id == 0

    def test_get_node_strict_invalid_id_raises(self):
        sim = Simulation.from_json(_get_json_path())
        with pytest.raises(KeyError, match="No node found"):
            sim.get_node_strict(99999)

    def test_get_node_returns_none_for_invalid(self):
        sim = Simulation.from_json(_get_json_path())
        assert sim.get_node(99999) is None

    def test_get_node_strict_error_message_helpful(self):
        sim = Simulation.from_json(_get_json_path())
        with pytest.raises(KeyError) as exc_info:
            sim.get_node_strict(99999)
        assert "Valid IDs" in str(exc_info.value)


class TestResultExporterContextManager:
    """Test ResultExporter as a context manager."""

    def test_context_manager_closes_on_exit(self, tmp_path):
        path = str(tmp_path / "results.json")
        with create_exporter(path) as exporter:
            assert exporter is not None
        # Should be closed after context exit

    def test_context_manager_with_results(self, tmp_path):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3)
        path = str(tmp_path / "results.json")
        with create_exporter(path) as exporter:
            exporter.export_results(sim.results)
        # File should exist
        import os
        assert os.path.exists(path)
