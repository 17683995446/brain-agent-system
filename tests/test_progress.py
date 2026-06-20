"""Tests for progress feedback features (Stage 1 usability)."""

import pytest
from brain_agent import Simulation, Signal


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


class TestOnStepCallback:
    """Test the on_step callback parameter of run()."""

    def test_on_step_called_for_each_step(self):
        sim = Simulation.from_json(_get_json_path())
        calls = []

        def callback(index, result):
            calls.append((index, result.step))

        sim.run(steps=5, on_step=callback)

        assert len(calls) == 5
        for i, (idx, step) in enumerate(calls):
            assert idx == i

    def test_on_step_receives_step_result(self):
        sim = Simulation.from_json(_get_json_path())
        results = []

        def callback(index, result):
            results.append(result)

        sim.run(steps=3, on_step=callback)

        assert len(results) == 3
        for r in results:
            assert hasattr(r, "nodes_processed")
            assert hasattr(r, "total_latency_ms")

    def test_on_step_none_works(self):
        sim = Simulation.from_json(_get_json_path())
        results = sim.run(steps=3, on_step=None)
        assert len(results) == 3

    def test_on_step_with_input_callback(self):
        sim = Simulation.from_json(_get_json_path())
        step_inputs = []

        def input_cb(step):
            return Signal(value=float(step), timestamp=step * 0.005)

        def on_step(index, result):
            step_inputs.append(index)

        sim.run(steps=5, input_callback=input_cb, on_step=on_step)
        assert len(step_inputs) == 5


class TestIterSteps:
    """Test the iter_steps() generator method."""

    def test_iter_steps_yields_results(self):
        sim = Simulation.from_json(_get_json_path())
        results = list(sim.iter_steps(steps=5))
        assert len(results) == 5

    def test_iter_steps_is_generator(self):
        import types
        sim = Simulation.from_json(_get_json_path())
        gen = sim.iter_steps(steps=3)
        assert isinstance(gen, types.GeneratorType)

    def test_iter_steps_early_termination(self):
        sim = Simulation.from_json(_get_json_path())
        count = 0
        for result in sim.iter_steps(steps=100):
            count += 1
            if count >= 3:
                break
        assert count == 3

    def test_iter_steps_with_input_callback(self):
        sim = Simulation.from_json(_get_json_path())

        def input_cb(step):
            return Signal(value=float(step), timestamp=step * 0.005)

        results = list(sim.iter_steps(steps=5, input_callback=input_cb))
        assert len(results) == 5

    def test_iter_steps_updates_results(self):
        sim = Simulation.from_json(_get_json_path())
        assert len(sim.results) == 0

        for result in sim.iter_steps(steps=3):
            pass

        assert len(sim.results) == 3

    def test_iter_steps_with_shutdown_checker(self):
        sim = Simulation.from_json(_get_json_path())
        call_count = [0]

        def shutdown_checker():
            call_count[0] += 1
            return call_count[0] > 2  # Stop after 2 calls

        results = list(sim.iter_steps(steps=100, shutdown_checker=shutdown_checker))
        assert len(results) <= 3  # Should stop early

    def test_iter_steps_zero_steps(self):
        sim = Simulation.from_json(_get_json_path())
        results = list(sim.iter_steps(steps=0))
        assert len(results) == 0


class TestProgressBar:
    """Test the progress bar parameter."""

    def test_progress_without_tqdm_does_not_crash(self):
        """progress=True should not crash even if tqdm is not installed."""
        sim = Simulation.from_json(_get_json_path())
        results = sim.run(steps=3, progress=True)
        assert len(results) == 3

    def test_iter_steps_progress_without_tqdm(self):
        sim = Simulation.from_json(_get_json_path())
        results = list(sim.iter_steps(steps=3, progress=True))
        assert len(results) == 3


class TestRunWithProgress:
    """Test that run() properly delegates to iter_steps()."""

    def test_run_returns_same_count_as_iter_steps(self):
        sim1 = Simulation.from_json(_get_json_path())
        sim2 = Simulation.from_json(_get_json_path())

        run_results = sim1.run(steps=5)
        iter_results = list(sim2.iter_steps(steps=5))

        assert len(run_results) == len(iter_results) == 5

    def test_run_with_on_step_and_progress(self):
        """on_step takes precedence over progress bar."""
        sim = Simulation.from_json(_get_json_path())
        calls = []

        def callback(index, result):
            calls.append(index)

        results = sim.run(steps=3, on_step=callback, progress=True)
        assert len(results) == 3
        assert len(calls) == 3
