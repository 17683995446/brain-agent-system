"""Path discovery utilities for brain simulation data files.

Centralizes the logic for finding brain_tree_v5.json and other data files,
eliminating hardcoded absolute paths and duplicate implementations.
"""

from pathlib import Path
from typing import Optional


def find_brain_tree_json(explicit_path: Optional[str] = None) -> str:
    """Find the brain tree JSON file.

    Search order:
    1. Explicit path provided as argument
    2. Path from BRAIN_JSON_PATH environment variable
    3. Relative to package installation (data/brain_tree_v5.json)
    4. Current working directory

    Args:
        explicit_path: Explicit path to the JSON file, or None to search.

    Returns:
        Absolute path to the brain tree JSON file.

    Raises:
        FileNotFoundError: If the file cannot be found in any location.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return str(p.resolve())
        raise FileNotFoundError(f"Specified JSON path does not exist: {explicit_path}")

    # Check environment variable
    import os
    env_path = os.environ.get("BRAIN_JSON_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p.resolve())

    # Search relative to package installation
    candidates = [
        Path(__file__).parent.parent / "data" / "brain_tree_v5.json",
        Path.cwd() / "data" / "brain_tree_v5.json",
        Path.cwd() / "brain_tree_v5.json",
    ]

    for p in candidates:
        if p.exists():
            return str(p.resolve())

    raise FileNotFoundError(
        "brain_tree_v5.json not found. Set BRAIN_JSON_PATH environment "
        "variable or pass json_path explicitly."
    )
