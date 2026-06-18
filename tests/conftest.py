import os
import pytest
import numpy as np

# Find JSON file - try multiple locations
_JSON_SEARCH_PATHS = [
    "/workspace/BrainAgentSystem/data/brain_tree_v5.json",
    "/data/user/work/brain_tree_v5.json",
]


def find_json_path():
    for p in _JSON_SEARCH_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("brain_tree_v5.json not found in any search path")


@pytest.fixture(scope="session")
def brain_json_path():
    return find_json_path()
