"""Tests for the experiment framework (Phase 8)."""

import pytest
import json
import os
import tempfile

from brain_agent import (
    Experiment,
    ExperimentConfig,
    ParameterSweep,
    Signal,
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


class TestExperimentConfig:
    def test_default_config(self):
        config = ExperimentConfig()
        assert config.name == "experiment"
        assert config.steps == 100
        assert config.seed is None

    def test_custom_config(self):
        config = ExperimentConfig(
            name="test",
            steps=50,
            seed=42,
            dt_fast=0.002,
        )
        assert config.name == "test"
        assert config.steps == 50
        assert config.seed == 42
        assert config.dt_fast == 0.002

    def test_tags(self):
        config = ExperimentConfig(tags=["baseline", "v1"])
        assert "baseline" in config.tags


class TestExperiment:
    def test_run_basic(self):
        exp = Experiment("test_basic", steps=5, json_path=_get_json_path())
        results = exp.run()
        assert len(results) == 5
        assert exp.metadata["duration_seconds"] > 0

    def test_run_with_seed(self):
        exp = Experiment("test_seed", steps=5, seed=42, json_path=_get_json_path())
        results = exp.run()
        assert len(results) == 5

    def test_summary(self):
        exp = Experiment("test_summary", steps=5, json_path=_get_json_path())
        exp.run()
        s = exp.summary()
        assert "test_summary" in s
        assert "Steps:" in s
        assert "Avg latency:" in s

    def test_export_json(self, tmp_path):
        exp = Experiment("test_export", steps=3, json_path=_get_json_path())
        exp.run()
        filepath = str(tmp_path / "results.json")
        exp.export(filepath)
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 3

    def test_checkpoint(self, tmp_path):
        exp = Experiment("test_ckpt", steps=3, json_path=_get_json_path())
        exp.run()
        filepath = str(tmp_path / "checkpoint.json")
        exp.checkpoint(filepath)
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "results" in data
        assert len(data["results"]) == 3

    def test_results_property(self):
        exp = Experiment("test_results", steps=3, json_path=_get_json_path())
        assert len(exp.results) == 0
        exp.run()
        assert len(exp.results) == 3

    def test_metadata_property(self):
        exp = Experiment("test_meta", steps=3, json_path=_get_json_path())
        exp.run()
        meta = exp.metadata
        assert meta["name"] == "test_meta"
        assert meta["start_time"] is not None
        assert meta["end_time"] is not None
        assert meta["duration_seconds"] > 0

    def test_run_with_input_callback(self):
        exp = Experiment("test_input", steps=5, json_path=_get_json_path())

        def input_cb(step):
            return Signal(value=float(step), timestamp=step * 0.005)

        results = exp.run(input_callback=input_cb)
        assert len(results) == 5

    def test_custom_dt_values(self):
        exp = Experiment(
            "test_dt",
            steps=5,
            dt_fast=0.002,
            dt_medium=0.01,
            dt_slow=0.05,
            json_path=_get_json_path(),
        )
        results = exp.run()
        assert len(results) == 5


class TestParameterSweep:
    def test_add_param(self):
        sweep = ParameterSweep("test_sweep", steps=5)
        sweep.add_param("dt_fast", [0.001, 0.005])
        assert "dt_fast" in sweep._params
        assert len(sweep._params["dt_fast"]) == 2

    def test_run_sweep(self):
        sweep = ParameterSweep(
            "test_sweep",
            steps=3,
            json_path=_get_json_path(),
            seed=42,
        )
        sweep.add_param("dt_fast", [0.005, 0.01])
        results = sweep.run()
        assert len(results) == 2
        assert "params" in results[0]
        assert "avg_latency_ms" in results[0]

    def test_best_metric(self):
        sweep = ParameterSweep(
            "test_best",
            steps=3,
            json_path=_get_json_path(),
        )
        sweep.add_param("dt_fast", [0.005, 0.01])
        sweep.run()

        best = sweep.best(metric="avg_latency_ms", minimize=True)
        assert best is not None
        assert "params" in best

        worst = sweep.best(metric="avg_latency_ms", minimize=False)
        assert worst is not None

    def test_export(self, tmp_path):
        sweep = ParameterSweep(
            "test_export",
            steps=3,
            json_path=_get_json_path(),
        )
        sweep.add_param("dt_fast", [0.005])
        sweep.run()

        filepath = str(tmp_path / "sweep.json")
        sweep.export(filepath)
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert data["sweep_name"] == "test_export"
        assert data["total_experiments"] == 1

    def test_summary(self):
        sweep = ParameterSweep(
            "test_summary",
            steps=3,
            json_path=_get_json_path(),
        )
        sweep.add_param("dt_fast", [0.005, 0.01])
        sweep.run()
        s = sweep.summary()
        assert "test_summary" in s
        assert "Total experiments: 2" in s
        assert "Best" in s

    def test_results_property(self):
        sweep = ParameterSweep(
            "test_results",
            steps=3,
            json_path=_get_json_path(),
        )
        assert len(sweep.results) == 0
        sweep.add_param("dt_fast", [0.005])
        sweep.run()
        assert len(sweep.results) == 1

    def test_chaining(self):
        sweep = ParameterSweep("test_chain", steps=3)
        result = sweep.add_param("dt_fast", [0.005]).add_param("dt_medium", [0.02])
        assert result is sweep
        assert "dt_fast" in sweep._params
        assert "dt_medium" in sweep._params
