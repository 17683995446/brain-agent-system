"""Shared pytest fixtures for BrainAgentSystem tests."""

import os
import pytest

from brain_agent.paths import find_brain_tree_json


@pytest.fixture(scope="session")
def brain_json_path():
    """Path to the brain tree JSON file, discovered via standard locations."""
    return find_brain_tree_json()
