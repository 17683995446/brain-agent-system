"""I/O module for brain simulation data input and output.

Provides adapters for reading simulation input data from various formats
and exporting simulation results.
"""

from .input import (
    InputAdapter,
    FileInput,
    ArrayInput,
    CallbackInput,
    ListInput,
    create_input,
)
from .output import (
    OutputAdapter,
    FileOutput,
    MemoryOutput,
    ResultExporter,
    create_exporter,
)

__all__ = [
    # Input
    "InputAdapter",
    "FileInput",
    "ArrayInput",
    "CallbackInput",
    "ListInput",
    "create_input",
    # Output
    "OutputAdapter",
    "FileOutput",
    "MemoryOutput",
    "ResultExporter",
    "create_exporter",
]
