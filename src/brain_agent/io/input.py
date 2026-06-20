"""Input data adapters for brain simulation.

Provides a unified interface for feeding data into the simulation from
various sources: files (JSON, NPZ, CSV), numpy arrays, Python callbacks,
and pre-built BrainData lists.

Usage:
    from brain_agent.io import create_input
    from brain_agent import Simulation

    # From NPZ file
    inp = create_input("data.npz")
    sim = Simulation.from_json("brain_tree_v5.json")
    results = sim.run(steps=100, input_callback=inp)
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np

from ..types import BrainData, Signal, Tensor, StateVector, Message

logger = logging.getLogger(__name__)


class InputAdapter:
    """Base class for input data adapters.

    An InputAdapter is callable: calling it with a step number returns
    the BrainData for that step. This makes it directly usable as the
    `input_callback` parameter of `Simulation.run()`.
    """

    def __call__(self, step: int) -> BrainData:
        """Return input data for the given step.

        Args:
            step: Zero-based step number.

        Returns:
            BrainData for this step.
        """
        raise NotImplementedError

    @property
    def length(self) -> int:
        """Return the number of available steps, or -1 if unlimited."""
        return -1

    def close(self) -> None:
        """Release any resources held by this adapter."""
        pass


class ListInput(InputAdapter):
    """Input adapter backed by a pre-built list of BrainData.

    Simplest adapter: provide a list of BrainData objects, one per step.
    When the list is exhausted, the last item is repeated.

    Example:
        from brain_agent import Signal
        from brain_agent.io import ListInput

        data = [Signal(value=float(i), timestamp=i*0.005) for i in range(100)]
        inp = ListInput(data)
    """

    def __init__(self, data: List[BrainData]) -> None:
        if not data:
            raise ValueError("ListInput requires at least one BrainData item")
        self._data = data
        self._index = 0

    def __call__(self, step: int) -> BrainData:
        if step < len(self._data):
            return self._data[step]
        return self._data[-1]

    @property
    def length(self) -> int:
        return len(self._data)


class CallbackInput(InputAdapter):
    """Input adapter backed by a Python callable.

    The callback receives the step number and returns BrainData.
    Useful for procedural or real-time input generation.

    Example:
        from brain_agent import Signal
        from brain_agent.io import CallbackInput

        def generate(step):
            return Signal(value=np.sin(step * 0.1), timestamp=step * 0.005)

        inp = CallbackInput(generate)
    """

    def __init__(self, callback: Callable[[int], BrainData]) -> None:
        self._callback = callback

    def __call__(self, step: int) -> BrainData:
        return self._callback(step)


class ArrayInput(InputAdapter):
    """Input adapter backed by numpy arrays.

    Converts numpy arrays to Tensor or Signal on each step.
    For multi-dimensional arrays, creates Tensor objects.
    For scalar values, creates Signal objects.

    Args:
        arrays: List of numpy arrays, one per step.
        dt: Time step in seconds for timestamp calculation.

    Example:
        import numpy as np
        from brain_agent.io import ArrayInput

        arrays = [np.random.randn(1, 8, 8).astype(np.float32) for _ in range(100)]
        inp = ArrayInput(arrays, dt=0.005)
    """

    def __init__(self, arrays: List[np.ndarray], dt: float = 0.005) -> None:
        if not arrays:
            raise ValueError("ArrayInput requires at least one array")
        self._arrays = arrays
        self._dt = dt

    def __call__(self, step: int) -> BrainData:
        idx = min(step, len(self._arrays) - 1)
        arr = self._arrays[idx]
        if arr.ndim == 0:
            return Signal(value=float(arr), timestamp=step * self._dt)
        return Tensor(data=arr, shape=arr.shape)

    @property
    def length(self) -> int:
        return len(self._arrays)


class FileInput(InputAdapter):
    """Input adapter that reads data from a file.

    Supports formats:
    - JSON: Array of objects with "type" and data fields
    - NPZ: NumPy compressed archive with arrays per step
    - CSV: Tabular data, one row per step

    Args:
        path: Path to the input file.
        dt: Time step in seconds for timestamp calculation.
        data_type: Force conversion to a specific type ("tensor", "signal",
                   "state_vector"). If None, auto-detect from data.

    JSON format example:
        [
            {"type": "signal", "value": 1.0, "timestamp": 0.0},
            {"type": "signal", "value": 0.5, "timestamp": 0.005}
        ]

    NPZ format: Each key "step_0", "step_1", ... is an array for that step.

    CSV format: Each row is a step. Numeric columns are packed into a Tensor.
        A "value" column becomes a Signal. A "timestamp" column sets the timestamp.

    Example:
        from brain_agent.io import FileInput

        inp = FileInput("input_data.json")
        # or
        inp = FileInput("signals.npz", data_type="signal")
    """

    def __init__(
        self,
        path: Union[str, Path],
        dt: float = 0.005,
        data_type: Optional[str] = None,
    ) -> None:
        self._path = Path(path)
        self._dt = dt
        self._data_type = data_type
        self._data: List[BrainData] = []
        self._index = 0
        self._load()

    def _load(self) -> None:
        """Load all data from the file into memory."""
        suffix = self._path.suffix.lower()

        if suffix == ".json":
            self._load_json()
        elif suffix == ".npz":
            self._load_npz()
        elif suffix == ".csv":
            self._load_csv()
        else:
            raise ValueError(f"Unsupported input file format: '{suffix}'. Supported formats: .json, .npz, .csv.")

        logger.info(f"Loaded {len(self._data)} input samples from {self._path}")

    def _load_json(self) -> None:
        """Load from JSON file."""
        with open(self._path, "r", encoding="utf-8") as f:
            items = json.load(f)

        if not isinstance(items, list):
            items = [items]

        for i, item in enumerate(items):
            dtype = item.get("type", self._data_type or "signal")

            if dtype == "signal":
                self._data.append(Signal(
                    value=float(item.get("value", 0.0)),
                    timestamp=float(item.get("timestamp", i * self._dt)),
                ))
            elif dtype == "tensor":
                arr = np.array(item.get("data", []), dtype=np.float32)
                shape = tuple(item.get("shape", arr.shape))
                self._data.append(Tensor(data=arr, shape=shape))
            elif dtype == "state_vector":
                arr = np.array(item.get("data", []), dtype=np.float32)
                dim = int(item.get("dimension", len(arr)))
                self._data.append(StateVector(data=arr, dimension=dim))
            elif dtype == "message":
                self._data.append(Message(
                    topic=item.get("topic", ""),
                    payload=item.get("payload", {}),
                    timestamp=float(item.get("timestamp", i * self._dt)),
                ))
            else:
                raise ValueError(
                    f"Unknown data type in JSON: '{dtype}'. "
                    f"Valid types: 'signal', 'tensor', 'state_vector', 'spike_train', 'message'."
                )

    def _load_npz(self) -> None:
        """Load from NPZ file."""
        with np.load(self._path) as npz:
            keys = sorted(
                [k for k in npz.files if k.startswith("step_")],
                key=lambda k: int(k.split("_")[1])
            )

            if not keys:
                keys = sorted(npz.files)

            for i, key in enumerate(keys):
                arr = npz[key]

                if self._data_type == "signal" or arr.ndim == 0:
                    self._data.append(Signal(
                        value=float(arr.flat[0]),
                        timestamp=i * self._dt,
                    ))
                elif self._data_type == "state_vector" or arr.ndim == 1:
                    self._data.append(StateVector(
                        data=arr.astype(np.float32),
                        dimension=arr.shape[0],
                    ))
                else:
                    self._data.append(Tensor(
                        data=arr.astype(np.float32),
                        shape=arr.shape,
                    ))

    def _load_csv(self) -> None:
        """Load from CSV file."""
        with open(self._path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for i, row in enumerate(rows):
            timestamp = float(row.get("timestamp", i * self._dt))

            if "value" in row:
                self._data.append(Signal(
                    value=float(row["value"]),
                    timestamp=timestamp,
                ))
            else:
                numeric_cols = {
                    k: float(v) for k, v in row.items()
                    if k != "timestamp" and v.strip()
                }
                if numeric_cols:
                    arr = np.array(list(numeric_cols.values()), dtype=np.float32)
                    self._data.append(Tensor(data=arr, shape=arr.shape))
                else:
                    self._data.append(Signal(value=0.0, timestamp=timestamp))

    def __call__(self, step: int) -> BrainData:
        if step < len(self._data):
            return self._data[step]
        return self._data[-1]

    @property
    def length(self) -> int:
        return len(self._data)


def create_input(
    source: Union[str, Path, List[BrainData], Callable[[int], BrainData], List[np.ndarray]],
    **kwargs: Any,
) -> InputAdapter:
    """Create an input adapter from various source types.

    Auto-detects the appropriate adapter based on the source type:
    - str/Path: FileInput (reads from file)
    - list of BrainData: ListInput
    - callable: CallbackInput
    - list of np.ndarray: ArrayInput

    Args:
        source: The input source.
        **kwargs: Additional arguments passed to the adapter constructor.

    Returns:
        An InputAdapter instance.

    Example:
        from brain_agent.io import create_input
        from brain_agent import Signal

        # From file
        inp = create_input("data.json")

        # From list
        inp = create_input([Signal(value=1.0, timestamp=0.0)])

        # From callback
        inp = create_input(lambda step: Signal(value=step, timestamp=step*0.005))
    """
    if isinstance(source, (str, Path)):
        return FileInput(source, **kwargs)
    elif isinstance(source, list):
        if not source:
            raise ValueError("Cannot create input from empty list")
        if isinstance(source[0], BrainData):
            return ListInput(source)
        elif isinstance(source[0], np.ndarray):
            return ArrayInput(source, **kwargs)
        else:
            raise ValueError(
                f"List items must be BrainData or np.ndarray, got {type(source[0])}"
            )
    elif callable(source):
        return CallbackInput(source)
    else:
        raise ValueError(
            f"Unsupported input source type: {type(source).__name__}. "
            f"Supported types: list, callable, str (file path), numpy ndarray, BrainData."
        )
