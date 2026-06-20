"""Tests for I/O input and output adapters (Phase 2 & 3)."""

import pytest
import json
import csv
import tempfile
import os
import numpy as np

from brain_agent import (
    Signal,
    Tensor,
    StateVector,
    Simulation,
    FrequencyScheduler,
)
from brain_agent.io import (
    InputAdapter,
    ListInput,
    CallbackInput,
    ArrayInput,
    FileInput,
    create_input,
    ResultExporter,
    create_exporter,
    MemoryOutput,
)


def _get_json_path():
    paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "brain_tree_v5.json"),
        "/workspace/BrainAgentSystem/data/brain_tree_v5.json",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    pytest.skip("brain_tree_v5.json not found")


# ============================================================================
# Input Adapter Tests
# ============================================================================

class TestListInput:
    def test_basic_usage(self):
        data = [Signal(value=float(i), timestamp=i * 0.005) for i in range(5)]
        inp = ListInput(data)
        assert inp(0).value == 0.0
        assert inp(4).value == 4.0

    def test_repeats_last_when_exhausted(self):
        data = [Signal(value=1.0, timestamp=0.0), Signal(value=2.0, timestamp=0.005)]
        inp = ListInput(data)
        assert inp(0).value == 1.0
        assert inp(1).value == 2.0
        assert inp(100).value == 2.0  # Repeats last

    def test_length(self):
        data = [Signal(value=1.0, timestamp=0.0) for _ in range(10)]
        inp = ListInput(data)
        assert inp.length == 10

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ListInput([])

    def test_with_tensors(self):
        data = [
            Tensor(data=np.ones((2, 2), dtype=np.float32), shape=(2, 2)),
            Tensor(data=np.zeros((2, 2), dtype=np.float32), shape=(2, 2)),
        ]
        inp = ListInput(data)
        assert inp(0).data[0, 0] == 1.0
        assert inp(1).data[0, 0] == 0.0


class TestCallbackInput:
    def test_basic_usage(self):
        def gen(step):
            return Signal(value=float(step), timestamp=step * 0.005)

        inp = CallbackInput(gen)
        assert inp(0).value == 0.0
        assert inp(5).value == 5.0

    def test_sine_wave(self):
        def gen(step):
            return Signal(value=np.sin(step * 0.1), timestamp=step * 0.005)

        inp = CallbackInput(gen)
        assert abs(inp(0).value - 0.0) < 1e-6
        assert abs(inp(15).value - np.sin(1.5)) < 1e-6

    def test_unlimited_length(self):
        inp = CallbackInput(lambda s: Signal(value=0.0, timestamp=0.0))
        assert inp.length == -1


class TestArrayInput:
    def test_tensor_arrays(self):
        arrays = [np.random.randn(2, 2).astype(np.float32) for _ in range(3)]
        inp = ArrayInput(arrays)
        result = inp(0)
        assert isinstance(result, Tensor)
        assert result.shape == (2, 2)

    def test_scalar_arrays_become_signals(self):
        arrays = [np.array(1.0), np.array(2.0)]
        inp = ArrayInput(arrays)
        result = inp(0)
        assert isinstance(result, Signal)
        assert result.value == 1.0

    def test_repeats_last(self):
        arrays = [np.ones((2,), dtype=np.float32)]
        inp = ArrayInput(arrays)
        result = inp(100)
        assert isinstance(result, Tensor)

    def test_length(self):
        arrays = [np.ones((2,), dtype=np.float32) for _ in range(5)]
        inp = ArrayInput(arrays)
        assert inp.length == 5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ArrayInput([])


class TestFileInput:
    def test_json_signals(self, tmp_path):
        filepath = tmp_path / "input.json"
        data = [
            {"type": "signal", "value": 1.0, "timestamp": 0.0},
            {"type": "signal", "value": 0.5, "timestamp": 0.005},
            {"type": "signal", "value": 0.25, "timestamp": 0.01},
        ]
        with open(filepath, "w") as f:
            json.dump(data, f)

        inp = FileInput(str(filepath))
        assert inp.length == 3
        assert inp(0).value == 1.0
        assert inp(2).value == 0.25

    def test_json_tensors(self, tmp_path):
        filepath = tmp_path / "tensors.json"
        data = [
            {"type": "tensor", "data": [[1, 2], [3, 4]], "shape": [2, 2]},
            {"type": "tensor", "data": [[5, 6], [7, 8]], "shape": [2, 2]},
        ]
        with open(filepath, "w") as f:
            json.dump(data, f)

        inp = FileInput(str(filepath))
        assert isinstance(inp(0), Tensor)
        assert inp(0).data[0, 0] == 1.0
        assert inp(1).data[1, 1] == 8.0

    def test_npz_input(self, tmp_path):
        filepath = tmp_path / "data.npz"
        arrays = {
            "step_0": np.ones((2, 2), dtype=np.float32),
            "step_1": np.zeros((2, 2), dtype=np.float32),
        }
        np.savez(str(filepath), **arrays)

        inp = FileInput(str(filepath))
        assert inp.length == 2
        assert isinstance(inp(0), Tensor)
        assert inp(0).data[0, 0] == 1.0
        assert inp(1).data[0, 0] == 0.0

    def test_csv_with_value_column(self, tmp_path):
        filepath = tmp_path / "signals.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["value", "timestamp"])
            writer.writeheader()
            writer.writerow({"value": "1.0", "timestamp": "0.0"})
            writer.writerow({"value": "0.5", "timestamp": "0.005"})

        inp = FileInput(str(filepath))
        assert inp.length == 2
        assert isinstance(inp(0), Signal)
        assert inp(0).value == 1.0

    def test_csv_with_numeric_columns(self, tmp_path):
        filepath = tmp_path / "data.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["a", "b", "c"])
            writer.writeheader()
            writer.writerow({"a": "1.0", "b": "2.0", "c": "3.0"})
            writer.writerow({"a": "4.0", "b": "5.0", "c": "6.0"})

        inp = FileInput(str(filepath))
        assert inp.length == 2
        result = inp(0)
        assert isinstance(result, Tensor)
        assert result.data[0] == 1.0

    def test_unsupported_format(self, tmp_path):
        filepath = tmp_path / "data.xyz"
        filepath.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            FileInput(str(filepath))


class TestCreateInput:
    def test_from_file(self, tmp_path):
        filepath = tmp_path / "input.json"
        with open(filepath, "w") as f:
            json.dump([{"type": "signal", "value": 1.0, "timestamp": 0.0}], f)
        inp = create_input(str(filepath))
        assert isinstance(inp, FileInput)

    def test_from_braindata_list(self):
        data = [Signal(value=1.0, timestamp=0.0)]
        inp = create_input(data)
        assert isinstance(inp, ListInput)

    def test_from_callback(self):
        inp = create_input(lambda s: Signal(value=float(s), timestamp=0.0))
        assert isinstance(inp, CallbackInput)

    def test_from_array_list(self):
        arrays = [np.ones((2, 2), dtype=np.float32)]
        inp = create_input(arrays)
        assert isinstance(inp, ArrayInput)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            create_input([])

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            create_input(42)


# ============================================================================
# Integration: Input + Simulation
# ============================================================================

class TestInputWithSimulation:
    def test_run_with_list_input(self):
        json_path = _get_json_path()
        sim = Simulation.from_json(json_path)
        data = [Signal(value=float(i), timestamp=i * 0.005) for i in range(5)]
        inp = ListInput(data)
        results = sim.run(steps=5, input_callback=inp)
        assert len(results) == 5

    def test_run_with_callback_input(self):
        json_path = _get_json_path()
        sim = Simulation.from_json(json_path)

        def gen(step):
            return Signal(value=np.sin(step * 0.1), timestamp=step * 0.005)

        inp = CallbackInput(gen)
        results = sim.run(steps=5, input_callback=inp)
        assert len(results) == 5

    def test_run_with_file_input(self, tmp_path):
        json_path = _get_json_path()
        filepath = tmp_path / "input.json"
        data = [{"type": "signal", "value": float(i), "timestamp": i * 0.005}
                for i in range(5)]
        with open(filepath, "w") as f:
            json.dump(data, f)

        sim = Simulation.from_json(json_path)
        inp = FileInput(str(filepath))
        results = sim.run(steps=5, input_callback=inp)
        assert len(results) == 5


# ============================================================================
# Output Adapter Tests
# ============================================================================

class TestResultExporter:
    def test_export_to_memory(self):
        exporter = create_exporter()
        assert isinstance(exporter._output, MemoryOutput)

    def test_export_results_to_json(self, tmp_path):
        json_path = _get_json_path()
        sim = Simulation.from_json(json_path)
        results = sim.run(steps=3)

        filepath = tmp_path / "results.json"
        exporter = create_exporter(str(filepath))
        exporter.export_results(results)
        exporter.close()

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 3
        assert "step" in data[0]
        assert "nodes_processed" in data[0]

    def test_export_results_to_csv(self, tmp_path):
        json_path = _get_json_path()
        sim = Simulation.from_json(json_path)
        results = sim.run(steps=3)

        filepath = tmp_path / "results.csv"
        exporter = create_exporter(str(filepath))
        exporter.export_results(results)
        exporter.close()

        assert filepath.exists()
        with open(filepath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert "step" in rows[0]

    def test_export_results_to_npz(self, tmp_path):
        json_path = _get_json_path()
        sim = Simulation.from_json(json_path)
        results = sim.run(steps=3)

        filepath = tmp_path / "results.npz"
        exporter = create_exporter(str(filepath))
        exporter.export_results(results)
        exporter.close()

        assert filepath.exists()
        data = np.load(str(filepath))
        assert "step_0_step" in data

    def test_export_metrics(self):
        json_path = _get_json_path()
        sim = Simulation.from_json(json_path)
        sim.run(steps=3)

        exporter = create_exporter()
        exporter.export_metrics(sim.metrics)
        assert exporter._metrics_str is not None
        assert len(exporter._metrics_str) > 0

    def test_export_node_states(self):
        json_path = _get_json_path()
        sim = Simulation.from_json(json_path)
        sim.run(steps=3)

        exporter = create_exporter()
        states = exporter.export_node_states(sim.tree)
        assert len(states) > 0
        node_0 = states[0]
        assert "node_id" in node_0
        assert "process_count" in node_0
        assert "queue_length" in node_0

    def test_summary(self):
        exporter = create_exporter()
        summary = exporter.summary()
        assert summary["result_count"] == 0
        assert summary["has_metrics"] is False
