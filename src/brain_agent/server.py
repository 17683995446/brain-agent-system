"""HTTP server for health checks, readiness probes, metrics, and progress.

Provides four endpoints:
- GET /health   - Liveness probe (200 if process is alive)
- GET /ready    - Readiness probe (200 if tree is built and scheduler is running)
- GET /metrics  - Prometheus-format metrics export
- GET /progress - Simulation progress (step, sim_time, nodes_processed, ETA)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, cast

from .scheduler import FrequencyScheduler
from .tree import BrainTree
from .monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health check endpoints."""

    # Injected by the server factory; stores shared state
    _scheduler: Optional[FrequencyScheduler] = None
    _tree: Optional[BrainTree] = None
    _metrics: Optional[MetricsCollector] = None
    _ready: bool = False
    _sim_start_time: float = 0.0
    _sim_total_steps: int = 0

    def do_GET(self) -> None:
        """Route GET requests to appropriate handlers."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/ready":
            self._handle_ready()
        elif self.path == "/metrics":
            self._handle_metrics()
        elif self.path == "/progress":
            self._handle_progress()
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _handle_health(self) -> None:
        """Liveness probe: always 200 if the process is alive."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = b'{"status":"alive"}'
        self.wfile.write(body)

    def _handle_ready(self) -> None:
        """Readiness probe: 200 if system is ready to serve."""
        if self._ready:
            node_count = 0
            if self._tree is not None:
                node_count = len(self._tree._nodes) if hasattr(self._tree, '_nodes') else 0
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = f'{{"status":"ready","nodes":{node_count}}}'.encode()
            self.wfile.write(body)
        else:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"not_ready"}')

    def _handle_metrics(self) -> None:
        """Prometheus metrics endpoint."""
        if self._metrics is not None:
            # Collect latest metrics before export
            self._metrics.collect_node_metrics()
            prometheus_text = self._metrics.export_prometheus()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(prometheus_text.encode("utf-8"))
        else:
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Metrics collector not initialized")

    def _handle_progress(self) -> None:
        """Simulation progress endpoint.

        Returns JSON with current step, sim_time, elapsed wall-clock time,
        estimated time remaining, and nodes processed.
        """
        if self._scheduler is None:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"scheduler not initialized"}')
            return

        current_step = self._scheduler.step_count
        sim_time = self._scheduler.sim_time
        elapsed = time.time() - self._sim_start_time if self._sim_start_time > 0 else 0.0

        # Estimate ETA
        eta_seconds = 0.0
        if self._sim_total_steps > 0 and current_step > 0:
            steps_per_sec = current_step / elapsed if elapsed > 0 else 0
            remaining = self._sim_total_steps - current_step
            eta_seconds = remaining / steps_per_sec if steps_per_sec > 0 else 0.0

        progress_pct = 0.0
        if self._sim_total_steps > 0:
            progress_pct = (current_step / self._sim_total_steps) * 100.0

        body = json.dumps({
            "step": current_step,
            "total_steps": self._sim_total_steps,
            "progress_pct": round(progress_pct, 1),
            "sim_time": round(sim_time, 4),
            "elapsed_seconds": round(elapsed, 2),
            "eta_seconds": round(eta_seconds, 2),
            "nodes_processed": current_step,
        })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default logging; use structured logger instead."""
        logger.debug(f"HTTP {args[0] if args else 'unknown'}")


class ReusableHTTPServer(HTTPServer):
    """HTTPServer with SO_REUSEADDR to avoid 'Address already in use' errors."""

    allow_reuse_address = True


class HealthServer:
    """Manages the HTTP health check server lifecycle.

    Runs the server in a background daemon thread so it does not
    block the main simulation loop.
    """

    def __init__(
        self,
        port: int = 9090,
        host: str = "0.0.0.0",
        scheduler: Optional[FrequencyScheduler] = None,
        tree: Optional[BrainTree] = None,
        metrics: Optional[MetricsCollector] = None,
        sim_total_steps: int = 0,
        sim_start_time: float = 0.0,
    ):
        self._port = port
        self._host = host
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        # Create a handler subclass with injected state
        handler_cls = cast(
            "type[HealthCheckHandler]",
            type(
                "InjectedHandler",
                (HealthCheckHandler,),
                {
                    "_scheduler": scheduler,
                    "_tree": tree,
                    "_metrics": metrics,
                    "_ready": False,
                    "_sim_total_steps": sim_total_steps,
                    "_sim_start_time": sim_start_time,
                },
            ),
        )
        self._handler_cls: type[HealthCheckHandler] = handler_cls

    def set_ready(self, ready: bool) -> None:
        """Update readiness state."""
        self._handler_cls._ready = ready

    def set_progress_info(self, total_steps: int, start_time: float) -> None:
        """Update progress tracking info for /progress endpoint.

        Args:
            total_steps: Total number of steps planned for this run.
            start_time: Wall-clock start time (time.time()).
        """
        self._handler_cls._sim_total_steps = total_steps
        self._handler_cls._sim_start_time = start_time

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        self._server = ReusableHTTPServer(
            (self._host, self._port), self._handler_cls
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="health-server",
        )
        self._thread.start()
        logger.info(f"Health server listening on {self._host}:{self._port}")

    def stop(self) -> None:
        """Stop the HTTP server gracefully."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            logger.info("Health server stopped")
        if self._thread is not None:
            self._thread.join(timeout=5.0)
