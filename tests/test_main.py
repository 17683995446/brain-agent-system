"""Test the main entry point."""
import subprocess
import sys

def test_main_help():
    result = subprocess.run([sys.executable, "-m", "brain_agent.main", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "大脑仿真Agent系统" in result.stdout

def test_main_runs():
    result = subprocess.run(
        [sys.executable, "-m", "brain_agent.main", "--steps", "5", "--log-level", "WARNING", "--no-server"],
        capture_output=True, text=True, cwd="/workspace/BrainAgentSystem"
    )
    assert result.returncode == 0
