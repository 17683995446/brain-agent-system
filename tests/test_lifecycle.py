"""Tests for graceful shutdown management."""

import sys
import os
import signal
import threading
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from brain_agent.lifecycle import GracefulShutdown


class TestGracefulShutdown:
    """Tests for the GracefulShutdown manager."""

    def test_initial_state_not_shutting_down(self):
        """New instance should not be in shutdown state."""
        gs = GracefulShutdown()
        assert not gs.is_shutting_down

    def test_register_and_run_callbacks(self):
        """Callbacks should run in LIFO order."""
        gs = GracefulShutdown()
        order = []
        gs.register_callback(lambda: order.append("first"))
        gs.register_callback(lambda: order.append("second"))
        gs.register_callback(lambda: order.append("third"))
        gs.run_callbacks()
        assert order == ["third", "second", "first"]

    def test_callback_exception_does_not_stop_others(self):
        """A failing callback should not prevent subsequent callbacks."""
        gs = GracefulShutdown()
        results = []
        gs.register_callback(lambda: results.append("before"))
        gs.register_callback(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        gs.register_callback(lambda: results.append("after"))
        gs.run_callbacks()
        # "after" runs before the failing callback (LIFO), "before" runs after
        assert "after" in results
        assert "before" in results

    def test_wait_for_shutdown_with_timeout(self):
        """wait_for_shutdown should return False on timeout."""
        gs = GracefulShutdown()
        result = gs.wait_for_shutdown(timeout=0.1)
        assert result is False

    def test_wait_for_shutdown_triggered(self):
        """wait_for_shutdown should return True when shutdown is set."""
        gs = GracefulShutdown()

        def trigger():
            time.sleep(0.05)
            gs._shutdown_event.set()

        t = threading.Thread(target=trigger)
        t.start()
        result = gs.wait_for_shutdown(timeout=2.0)
        t.join()
        assert result is True

    def test_reset(self):
        """reset should clear the shutdown state."""
        gs = GracefulShutdown()
        gs._shutdown_event.set()
        assert gs.is_shutting_down
        gs.reset()
        assert not gs.is_shutting_down

    def test_install_signal_handlers(self):
        """install_signal_handlers should register handlers."""
        gs = GracefulShutdown()
        gs.install_signal_handlers()
        # Verify handlers are installed (not default)
        sigterm_handler = signal.getsignal(signal.SIGTERM)
        sigint_handler = signal.getsignal(signal.SIGINT)
        assert sigterm_handler is not signal.SIG_DFL
        assert sigint_handler is not signal.SIG_DFL

    def test_signal_sets_shutdown_event(self):
        """Sending SIGINT should set the shutdown event."""
        gs = GracefulShutdown()
        gs.install_signal_handlers()

        # Send SIGINT to self
        # Note: This is safe in test context
        import os
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.1)

        assert gs.is_shutting_down
        gs.reset()

    def test_shutdown_checker_with_scheduler(self):
        """shutdown_checker should integrate with scheduler.run()."""
        from brain_agent.scheduler import FrequencyScheduler, SchedulerConfig
        from brain_agent.tree import BrainTreeBuilder, PassThroughBrainNode
        from brain_agent.types import Signal

        # Build a small tree
        builder = BrainTreeBuilder()
        json_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'brain_tree_v5.json'
        )
        if os.path.exists(json_path):
            tree = builder.build_from_json(json_path)
            scheduler = FrequencyScheduler(tree, SchedulerConfig(max_steps=0))

            gs = GracefulShutdown()
            gs._shutdown_event.set()  # Simulate shutdown already triggered

            results = scheduler.run(
                steps=100,
                shutdown_checker=lambda: gs.is_shutting_down
            )
            # Should stop immediately (0 or 1 steps)
            assert len(results) <= 1
