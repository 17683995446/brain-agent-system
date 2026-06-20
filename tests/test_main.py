"""Test the main entry point."""
import subprocess
import sys


def test_main_help():
    result = subprocess.run(
        [sys.executable, "-m", "brain_agent.main", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "brain-agent" in result.stdout
    assert "run" in result.stdout
    assert "info" in result.stdout
    assert "validate" in result.stdout


def test_main_version():
    result = subprocess.run(
        [sys.executable, "-m", "brain_agent.main", "--version"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "brain-agent" in result.stdout


def test_main_runs():
    result = subprocess.run(
        [
            sys.executable, "-m", "brain_agent.main",
            "run", "--steps", "5", "--log-level", "WARNING", "--no-server",
        ],
        capture_output=True, text=True, cwd="/workspace/BrainAgentSystem",
    )
    assert result.returncode == 0


def test_main_info():
    result = subprocess.run(
        [
            sys.executable, "-m", "brain_agent.main",
            "info",
        ],
        capture_output=True, text=True, cwd="/workspace/BrainAgentSystem",
    )
    assert result.returncode == 0
    assert "Brain Agent System" in result.stdout


def test_main_validate():
    result = subprocess.run(
        [
            sys.executable, "-m", "brain_agent.main",
            "validate",
        ],
        capture_output=True, text=True, cwd="/workspace/BrainAgentSystem",
    )
    assert result.returncode == 0
    assert "VALID" in result.stdout


def test_main_no_command():
    result = subprocess.run(
        [sys.executable, "-m", "brain_agent.main"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
