"""Tests for Tracer integration and /progress endpoint (Stage 2 quality)."""

import json
import time
import pytest
from brain_agent import Simulation, Signal
from brain_agent.monitoring.tracer import Tracer


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


class TestTracerIntegration:
    """Test Tracer integration with the scheduler."""

    def test_tracer_is_set_on_scheduler(self):
        sim = Simulation.from_json(_get_json_path())
        assert sim.scheduler.tracer is not None
        assert sim.scheduler.tracer is sim.tracer

    def test_tracer_records_spans_after_run(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3, input_data=Signal(value=1.0))
        traces = sim.tracer.get_traces()
        assert len(traces) > 0

    def test_tracer_spans_have_node_info(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=1, input_data=Signal(value=1.0))
        traces = sim.tracer.get_traces()
        assert len(traces) > 0
        root_span = traces[0]["root"]
        assert hasattr(root_span, "node_id")
        assert hasattr(root_span, "node_name")
        assert hasattr(root_span, "duration_ms")
        assert root_span.duration_ms >= 0.0

    def test_tracer_format_trace(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=1, input_data=Signal(value=1.0))
        formatted = sim.tracer.format_trace()
        assert isinstance(formatted, str)
        assert "Trace:" in formatted

    def test_tracer_clear(self):
        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=1, input_data=Signal(value=1.0))
        assert len(sim.tracer.get_traces()) > 0
        sim.tracer.clear()
        assert len(sim.tracer.get_traces()) == 0

    def test_set_tracer_none_disables_tracing(self):
        sim = Simulation.from_json(_get_json_path())
        sim.scheduler.set_tracer(None)
        sim.run(steps=1, input_data=Signal(value=1.0))
        # Tracer object still exists but wasn't used
        assert len(sim.tracer.get_traces()) == 0


class TestProgressEndpoint:
    """Test the /progress HTTP endpoint."""

    def test_progress_endpoint_returns_json(self):
        from brain_agent.server import HealthServer
        import urllib.request

        sim = Simulation.from_json(_get_json_path())
        sim.run(steps=3, input_data=Signal(value=1.0))

        server = HealthServer(
            port=19191,
            scheduler=sim.scheduler,
            tree=sim.tree,
            metrics=sim.metrics,
            sim_total_steps=100,
            sim_start_time=time.time() - 5.0,
        )
        server.start()
        try:
            time.sleep(0.5)
            response = urllib.request.urlopen("http://localhost:19191/progress")
            assert response.status == 200
            data = json.loads(response.read())
            assert "step" in data
            assert "total_steps" in data
            assert "progress_pct" in data
            assert "sim_time" in data
            assert "elapsed_seconds" in data
            assert "eta_seconds" in data
            assert data["step"] == 3
            assert data["total_steps"] == 100
        finally:
            server.stop()

    def test_progress_endpoint_without_scheduler(self):
        from brain_agent.server import HealthServer
        import urllib.error

        server = HealthServer(port=19192, scheduler=None)
        server.start()
        try:
            time.sleep(0.5)
            try:
                urllib.request.urlopen("http://localhost:19192/progress")
                assert False, "Should have raised HTTPError"
            except urllib.error.HTTPError as e:
                assert e.code == 503
                data = json.loads(e.read())
                assert "error" in data
        finally:
            server.stop()

    def test_set_progress_info(self):
        from brain_agent.server import HealthServer

        server = HealthServer(port=19193)
        server.set_progress_info(total_steps=500, start_time=1000.0)
        assert server._handler_cls._sim_total_steps == 500
        assert server._handler_cls._sim_start_time == 1000.0
