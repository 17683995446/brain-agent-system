"""Batch experiment framework for brain simulation.

Provides a high-level API for running multiple simulation experiments
with parameter sweeps, checkpointing, and deterministic reproduction.

Usage:
    from brain_agent import Experiment, ExperimentConfig

    # Single experiment
    exp = Experiment("my_experiment")
    exp.run(steps=100)

    # Parameter sweep
    sweep = ParameterSweep("dt_sweep")
    sweep.add_param("dt_fast", [0.001, 0.002, 0.005])
    sweep.add_param("dt_medium", [0.01, 0.02])
    results = sweep.run(steps=50)
    sweep.export("sweep_results.json")
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Literal

import numpy as np

from .api import Simulation
from .scheduler import SchedulerConfig, StepResult
from .types import BrainData

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        name: Experiment name (used for output files).
        steps: Number of simulation steps.
        seed: Random seed for reproducibility (None = non-deterministic).
        dt_fast: Fast band time step in seconds.
        dt_medium: Medium band time step in seconds.
        dt_slow: Slow band time step in seconds.
        json_path: Path to brain tree JSON (None = auto-discover).
        output_dir: Directory for output files.
        tags: Optional tags for categorizing experiments.
    """
    name: str = "experiment"
    steps: int = 100
    seed: Optional[int] = None
    dt_fast: float = 0.005
    dt_medium: float = 0.02
    dt_slow: float = 0.1
    json_path: Optional[str] = None
    output_dir: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class Experiment:
    """A single simulation experiment with reproducibility support.

    Wraps a Simulation instance with experiment metadata, checkpointing,
    and deterministic reproduction via random seed control.

    Example:
        exp = Experiment("baseline", steps=100, seed=42)
        results = exp.run()
        exp.export("results.json")
        print(exp.summary())

        # Reproduce
        exp2 = Experiment("baseline", steps=100, seed=42)
        results2 = exp2.run()
        assert results == results2  # Deterministic
    """

    def __init__(self, config: Union[ExperimentConfig, str, None] = None, **kwargs: Any) -> None:
        """Initialize experiment.

        Args:
            config: ExperimentConfig or name string. If string, creates
                ExperimentConfig with that name.
            **kwargs: Override ExperimentConfig fields if config is a string or None.
        """
        if isinstance(config, ExperimentConfig):
            self.config = config
        elif isinstance(config, str):
            self.config = ExperimentConfig(name=config, **kwargs)
        else:
            self.config = ExperimentConfig(**kwargs)

        self._sim: Optional[Simulation] = None
        self._results: List[StepResult] = []
        self._metadata: Dict[str, Any] = {
            "name": self.config.name,
            "config": asdict(self.config),
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
        }

    def _setup_seed(self) -> None:
        """Set random seeds for reproducibility."""
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            logger.info(f"Random seed set to {self.config.seed}")

    def _create_simulation(self) -> Simulation:
        """Create a Simulation instance from experiment config."""
        sched_config = SchedulerConfig(
            dt_fast=self.config.dt_fast,
            dt_medium=self.config.dt_medium,
            dt_slow=self.config.dt_slow,
        )
        return Simulation.from_json(
            json_path=self.config.json_path,
            config=sched_config,
        )

    def run(
        self,
        steps: Optional[int] = None,
        input_callback: Optional[Callable[[int], BrainData]] = None,
    ) -> List[StepResult]:
        """Run the experiment.

        Args:
            steps: Override config.steps if provided.
            input_callback: Optional input data generator.

        Returns:
            List of StepResult for each step.
        """
        self._setup_seed()
        self._sim = self._create_simulation()

        actual_steps = steps or self.config.steps
        self._metadata["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        start = time.perf_counter()

        self._results = self._sim.run(
            steps=actual_steps,
            input_callback=input_callback,
        )

        elapsed = time.perf_counter() - start
        self._metadata["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._metadata["duration_seconds"] = elapsed

        logger.info(
            f"Experiment '{self.config.name}' completed: "
            f"{actual_steps} steps in {elapsed:.2f}s"
        )
        return self._results

    def checkpoint(self, path: Optional[str] = None) -> str:
        """Save experiment checkpoint for later resumption.

        Args:
            path: Checkpoint file path. If None, uses output_dir/name_ckpt.json.

        Returns:
            Path to the checkpoint file.
        """
        if path is None:
            path = self._default_path("_ckpt.json")

        data = {
            "metadata": self._metadata,
            "results": [
                {
                    "step": r.step,
                    "sim_time": r.sim_time,
                    "band": r.band.value,
                    "nodes_processed": r.nodes_processed,
                    "total_latency_ms": r.total_latency_ms,
                    "max_node_latency_ms": r.max_node_latency_ms,
                    "errors": r.errors,
                }
                for r in self._results
            ],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Checkpoint saved to: {path}")
        return path

    def export(self, path: Optional[str] = None, format: Literal["json", "csv", "npz"] = "json") -> str:
        """Export experiment results.

        Args:
            path: Output file path. If None, uses output_dir/name_results.{format}.
            format: Output format ("json", "csv", "npz").

        Returns:
            Path to the exported file.
        """
        if path is None:
            path = self._default_path(f"_results.{format}")

        if self._sim is None:
            raise RuntimeError("Must run() before export()")

        self._sim.export(path=path, format=format)
        return path

    def _default_path(self, suffix: str) -> str:
        """Generate default output path."""
        if self.config.output_dir:
            return os.path.join(self.config.output_dir, f"{self.config.name}{suffix}")
        return f"{self.config.name}{suffix}"

    @property
    def results(self) -> List[StepResult]:
        """Get experiment results."""
        return self._results

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get experiment metadata."""
        return self._metadata

    def summary(self) -> str:
        """Get a human-readable summary.

        Returns:
            Formatted summary string.
        """
        lines = [
            f"Experiment: {self.config.name}",
            "=" * 50,
            f"  Steps:          {len(self._results)}",
            f"  Duration:       {self._metadata.get('duration_seconds', 0):.2f}s",
            f"  Seed:           {self.config.seed}",
            f"  dt_fast:        {self.config.dt_fast}s",
            f"  dt_medium:      {self.config.dt_medium}s",
            f"  dt_slow:        {self.config.dt_slow}s",
        ]
        if self._results:
            avg_latency = sum(r.total_latency_ms for r in self._results) / len(self._results)
            max_latency = max(r.total_latency_ms for r in self._results)
            total_errors = sum(len(r.errors) for r in self._results)
            avg_nodes = sum(r.nodes_processed for r in self._results) / len(self._results)
            lines.extend([
                f"  Avg latency:    {avg_latency:.2f}ms",
                f"  Max latency:    {max_latency:.2f}ms",
                f"  Avg nodes/step: {avg_nodes:.0f}",
                f"  Total errors:   {total_errors}",
            ])
        if self.config.tags:
            lines.append(f"  Tags:           {', '.join(self.config.tags)}")
        return "\n".join(lines)


class ParameterSweep:
    """Run experiments across a grid of parameters.

    Sweeps one or more parameters, running a full experiment for each
    combination. Results are aggregated for comparison.

    Example:
        sweep = ParameterSweep("dt_sweep", steps=50)
        sweep.add_param("dt_fast", [0.001, 0.002, 0.005])
        sweep.add_param("dt_medium", [0.01, 0.02])
        results = sweep.run()

        # Get best configuration
        best = sweep.best(metric="avg_latency", minimize=True)
        print(f"Best: {best}")

        # Export all results
        sweep.export("sweep_results.json")
    """

    def __init__(
        self,
        name: str = "sweep",
        steps: int = 100,
        json_path: Optional[str] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.name = name
        self.steps = steps
        self.json_path = json_path
        self.seed = seed
        self._params: Dict[str, List[Any]] = {}
        self._results: List[Dict[str, Any]] = []
        self._experiments: List[Experiment] = []

    def add_param(self, name: str, values: List[Any]) -> "ParameterSweep":
        """Add a parameter to sweep.

        Args:
            name: Parameter name (must match ExperimentConfig field).
            values: List of values to try.

        Returns:
            self (for chaining).
        """
        self._params[name] = values
        return self

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools

        keys = list(self._params.keys())
        value_lists = [self._params[k] for k in keys]

        combinations = []
        for combo in itertools.product(*value_lists):
            combinations.append(dict(zip(keys, combo)))
        return combinations

    def run(self) -> List[Dict[str, Any]]:
        """Run experiments for all parameter combinations.

        Returns:
            List of result dicts, each with parameters and metrics.
        """
        combinations = self._generate_combinations()
        total = len(combinations)
        logger.info(f"Starting parameter sweep: {total} experiments")

        self._results = []
        self._experiments = []

        for i, params in enumerate(combinations):
            exp_name = f"{self.name}_{i:04d}"
            config = ExperimentConfig(
                name=exp_name,
                steps=self.steps,
                seed=self.seed,
                json_path=self.json_path,
                **params,
            )
            exp = Experiment(config)
            results = exp.run()

            # Aggregate metrics
            avg_latency = sum(r.total_latency_ms for r in results) / len(results) if results else 0
            max_latency = max(r.total_latency_ms for r in results) if results else 0
            total_errors = sum(len(r.errors) for r in results)
            avg_nodes = sum(r.nodes_processed for r in results) / len(results) if results else 0

            result_entry = {
                "experiment": exp_name,
                "params": params,
                "steps": len(results),
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "avg_nodes_per_step": avg_nodes,
                "total_errors": total_errors,
                "duration_seconds": exp.metadata.get("duration_seconds", 0),
            }
            self._results.append(result_entry)
            self._experiments.append(exp)

            logger.info(
                f"  [{i+1}/{total}] {params} -> "
                f"avg_latency={avg_latency:.1f}ms, errors={total_errors}"
            )

        logger.info(f"Sweep complete: {total} experiments")
        return self._results

    def best(
        self,
        metric: str = "avg_latency_ms",
        minimize: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Find the best experiment by a metric.

        Args:
            metric: Metric name to compare.
            minimize: If True, lower is better. If False, higher is better.

        Returns:
            Best result entry, or None if no results.
        """
        if not self._results:
            return None

        sorted_results = sorted(
            self._results,
            key=lambda r: r.get(metric, 0),
            reverse=not minimize,
        )
        return sorted_results[0]

    def export(self, path: Optional[str] = None) -> str:
        """Export sweep results to JSON.

        Args:
            path: Output file path. If None, uses {name}_sweep.json.

        Returns:
            Path to the exported file.
        """
        if path is None:
            path = f"{self.name}_sweep.json"

        data = {
            "sweep_name": self.name,
            "parameters": self._params,
            "total_experiments": len(self._results),
            "results": self._results,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Sweep results exported to: {path}")
        return path

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Get all sweep results."""
        return self._results

    def summary(self) -> str:
        """Get a summary of the sweep.

        Returns:
            Formatted summary string.
        """
        lines = [
            f"Parameter Sweep: {self.name}",
            "=" * 50,
            f"  Total experiments: {len(self._results)}",
            f"  Parameters:        {list(self._params.keys())}",
        ]
        if self._results:
            best = self.best(metric="avg_latency_ms", minimize=True)
            worst = self.best(metric="avg_latency_ms", minimize=False)
            if best is not None and worst is not None:
                lines.extend([
                    "\n  Best (lowest latency):",
                    f"    Params: {best['params']}",
                    f"    Avg latency: {best['avg_latency_ms']:.2f}ms",
                    "\n  Worst (highest latency):",
                    f"    Params: {worst['params']}",
                    f"    Avg latency: {worst['avg_latency_ms']:.2f}ms",
                ])
        return "\n".join(lines)
