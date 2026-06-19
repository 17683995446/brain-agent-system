"""Graceful shutdown management for brain simulation.

Handles SIGTERM and SIGINT signals to ensure the simulation
completes the current step, flushes metrics, and exits cleanly.
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Manages graceful shutdown via signal handlers.

    Registers callbacks that are executed in reverse order when
    a shutdown signal is received. The simulation loop can check
    ``is_shutting_down`` to decide when to stop.
    """

    def __init__(self) -> None:
        self._shutdown_event = threading.Event()
        self._callbacks: List[Callable[[], None]] = []
        self._registered = False

    @property
    def is_shutting_down(self) -> bool:
        """True if a shutdown signal has been received."""
        return self._shutdown_event.is_set()

    def register_callback(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback to run on shutdown.

        Callbacks are executed in reverse registration order (LIFO).
        """
        self._callbacks.append(callback)

    def install_signal_handlers(self) -> None:
        """Install SIGTERM and SIGINT handlers.

        Safe to call from the main thread only.
        """
        if self._registered:
            return
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        self._registered = True
        logger.info("Graceful shutdown handlers installed (SIGTERM, SIGINT)")

    def _handle_signal(self, signum: int, frame: object) -> None:
        """Signal handler: set the shutdown event and run callbacks."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_event.set()

    def run_callbacks(self) -> None:
        """Execute all registered cleanup callbacks in LIFO order."""
        for callback in reversed(self._callbacks):
            try:
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}", exc_info=True)

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Block until shutdown signal is received or timeout expires.

        Returns True if shutdown was triggered, False on timeout.
        """
        return self._shutdown_event.wait(timeout)

    def reset(self) -> None:
        """Reset the shutdown state (for testing)."""
        self._shutdown_event.clear()
