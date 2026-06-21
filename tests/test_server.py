"""Tests for HTTP health check server."""

import time
import urllib.request
import pytest

from brain_agent.server import HealthServer
from brain_agent.monitoring.metrics import MetricsCollector


def _get_free_port() -> int:
    """Find a free port for testing."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestHealthServer:
    """Tests for the HTTP health check server."""

    @pytest.fixture
    def server(self):
        """Create and start a health server on a free port."""
        port = _get_free_port()
        metrics = MetricsCollector()
        srv = HealthServer(port=port, metrics=metrics)
        srv.start()
        time.sleep(0.3)  # wait for server to bind
        yield srv
        srv.stop()

    def _get(self, server: HealthServer, path: str) -> tuple:
        """Make a GET request and return (status_code, body)."""
        port = server._port
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=5)
            return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def test_health_endpoint(self, server):
        """GET /health should return 200 with alive status."""
        status, body = self._get(server, "/health")
        assert status == 200
        assert "alive" in body

    def test_ready_not_ready(self, server):
        """GET /ready should return 503 when not ready."""
        server.set_ready(False)
        status, body = self._get(server, "/ready")
        assert status == 503
        assert "not_ready" in body

    def test_ready_when_ready(self, server):
        """GET /ready should return 200 when ready."""
        server.set_ready(True)
        status, body = self._get(server, "/ready")
        assert status == 200
        assert "ready" in body

    def test_metrics_endpoint(self, server):
        """GET /metrics should return 200 with text content."""
        status, body = self._get(server, "/metrics")
        assert status == 200

    def test_unknown_path_returns_404(self, server):
        """Unknown paths should return 404."""
        status, _ = self._get(server, "/unknown")
        assert status == 404

    def test_server_start_stop(self):
        """Server should start and stop without errors."""
        port = _get_free_port()
        srv = HealthServer(port=port)
        srv.start()
        time.sleep(0.2)
        srv.stop()

    def test_set_ready_toggle(self, server):
        """set_ready should toggle readiness state."""
        server.set_ready(False)
        status, _ = self._get(server, "/ready")
        assert status == 503

        server.set_ready(True)
        status, _ = self._get(server, "/ready")
        assert status == 200
