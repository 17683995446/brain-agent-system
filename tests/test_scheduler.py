"""Tests for multi-rate frequency scheduler (Phase 9).

Tests FrequencyScheduler node classification, step execution,
multi-rate band scheduling, timing, error handling, reset, stats,
and full 2047-node tree integration.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain_agent.node import BrainNode, NodeConfig, RegionType, Protocol
from brain_agent.types import BrainData, Signal, Tensor, Message
from brain_agent.tree import (
    BrainTreeBuilder,
    BrainTree,
    PassThroughBrainNode,
)
from brain_agent.scheduler import (
    FrequencyBand,
    SchedulerConfig,
    StepResult,
    FrequencyScheduler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DESIGN_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "L00_BrainSystem"
)


def _make_small_tree_json(tmp_path, num_layers=4):
    """Create a small JSON file with a complete binary tree of num_layers layers.

    Layer 0: 1 node, Layer 1: 2 nodes, ..., Layer num_layers-1: 2^(num_layers-1) nodes
    Total nodes = 2^num_layers - 1
    """
    nodes = []
    node_id = 0
    for layer in range(num_layers):
        nodes_in_layer = 2 ** layer
        for i in range(nodes_in_layer):
            parent_id = None
            children = []
            if layer > 0:
                parent_id = (node_id - 1) // 2
            if layer < num_layers - 1:
                children = [2 * node_id + 1, 2 * node_id + 2]

            nodes.append({
                "id": node_id,
                "name_cn": f"节点{node_id}",
                "short_name": f"Node{node_id}",
                "full_name": f"Node{node_id}",
                "folder_name": f"L{layer:02d}_Node{node_id}",
                "layer": layer,
                "parent_id": parent_id,
                "responsibility": f"节点{node_id}的职责",
                "tensor_shape": "[1,64,64]" if layer < 3 else "[1,32,32]",
                "frequency": f"{100 - layer * 10}Hz",
                "protocol": "ASYNC_PUSH",
                "algorithm": "pass_through",
                "region_type": "sensory" if layer < 2 else "relay",
                "children": children,
            })
            node_id += 1

    json_path = tmp_path / "test_tree.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"nodes": nodes}, f, ensure_ascii=False, indent=2)
    return json_path


def _make_mixed_protocol_json(tmp_path, num_layers=4):
    """Create a JSON tree with mixed protocols for testing band classification.

    Layer 0: PUB_SUB (slow)
    Layer 1: ASYNC_PUSH (fast) and SYNC_CALL (medium)
    Layer 2+: mix of all three
    """
    nodes = []
    node_id = 0
    protocols = ["PUB_SUB", "ASYNC_PUSH", "SYNC_CALL"]
    for layer in range(num_layers):
        nodes_in_layer = 2 ** layer
        for i in range(nodes_in_layer):
            parent_id = None
            children = []
            if layer > 0:
                parent_id = (node_id - 1) // 2
            if layer < num_layers - 1:
                children = [2 * node_id + 1, 2 * node_id + 2]

            # Distribute protocols across nodes
            protocol = protocols[node_id % 3]

            nodes.append({
                "id": node_id,
                "name_cn": f"节点{node_id}",
                "short_name": f"Node{node_id}",
                "full_name": f"Node{node_id}",
                "folder_name": f"L{layer:02d}_Node{node_id}",
                "layer": layer,
                "parent_id": parent_id,
                "responsibility": f"节点{node_id}的职责",
                "tensor_shape": "[1,64,64]",
                "frequency": f"{100 - layer * 10}Hz",
                "protocol": protocol,
                "algorithm": "pass_through",
                "region_type": "relay",
                "children": children,
            })
            node_id += 1

    json_path = tmp_path / "test_mixed_tree.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"nodes": nodes}, f, ensure_ascii=False, indent=2)
    return json_path


def _make_full_2047_json(tmp_path):
    """Create a full 2047-node JSON file for testing.

    11 layers (0-10), total = 2^11 - 1 = 2047 nodes.
    """
    nodes = []
    num_layers = 11
    for node_id in range(2047):
        layer = 0
        start = 0
        for l in range(num_layers):
            if node_id < start + 2 ** l:
                layer = l
                break
            start += 2 ** l

        parent_id = None
        children = []
        if layer > 0:
            parent_id = (node_id - 1) // 2
        if layer < num_layers - 1:
            children = [2 * node_id + 1, 2 * node_id + 2]

        if layer == 0:
            rt = "unknown"
        elif layer <= 2:
            rt = "sensory" if node_id % 2 == 0 else "motor"
        elif layer <= 5:
            rt = "memory"
        else:
            rt = "relay"

        nodes.append({
            "id": node_id,
            "name_cn": f"节点{node_id}",
            "short_name": f"Node{node_id}",
            "full_name": f"Node{node_id}",
            "folder_name": f"L{layer:02d}_Node{node_id}",
            "layer": layer,
            "parent_id": parent_id,
            "responsibility": f"节点{node_id}的职责",
            "tensor_shape": "[1,64,64]",
            "frequency": f"{100 - layer * 10}Hz",
            "protocol": "ASYNC_PUSH",
            "algorithm": "pass_through",
            "region_type": rt,
            "children": children,
        })

    json_path = tmp_path / "test_tree_2047.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"nodes": nodes}, f, ensure_ascii=False, indent=2)
    return json_path


# ---------------------------------------------------------------------------
# 1. Test node classification into frequency bands
# ---------------------------------------------------------------------------

class TestNodeClassification:
    """Test that nodes are correctly classified into frequency bands."""

    def test_async_push_nodes_go_to_fast_band(self, tmp_path):
        """Nodes with ASYNC_PUSH protocol should be in the fast band."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        fast_nodes = scheduler._band_nodes[FrequencyBand.FAST]

        # All nodes in the small tree have ASYNC_PUSH protocol
        assert len(fast_nodes) == tree.total_nodes
        for node in fast_nodes:
            assert node.protocol == Protocol.ASYNC_PUSH

    def test_mixed_protocols_classified_correctly(self, tmp_path):
        """Nodes with different protocols should be in the correct bands."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)

        # Verify each node is in the correct band
        for node in tree.bfs_traverse():
            if node.protocol == Protocol.ASYNC_PUSH:
                assert node in scheduler._band_nodes[FrequencyBand.FAST]
            elif node.protocol == Protocol.SYNC_CALL:
                assert node in scheduler._band_nodes[FrequencyBand.MEDIUM]
            elif node.protocol == Protocol.PUB_SUB:
                assert node in scheduler._band_nodes[FrequencyBand.SLOW]

    def test_all_nodes_classified(self, tmp_path):
        """Every node in the tree should be classified into exactly one band."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)

        total_classified = (
            len(scheduler._band_nodes[FrequencyBand.FAST])
            + len(scheduler._band_nodes[FrequencyBand.MEDIUM])
            + len(scheduler._band_nodes[FrequencyBand.SLOW])
        )
        assert total_classified == tree.total_nodes

    def test_band_counts_in_stats(self, tmp_path):
        """get_stats should report correct band counts."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        stats = scheduler.get_stats()

        assert stats["band_counts"]["fast"] == len(
            scheduler._band_nodes[FrequencyBand.FAST]
        )
        assert stats["band_counts"]["medium"] == len(
            scheduler._band_nodes[FrequencyBand.MEDIUM]
        )
        assert stats["band_counts"]["slow"] == len(
            scheduler._band_nodes[FrequencyBand.SLOW]
        )


# ---------------------------------------------------------------------------
# 2. Test single step execution
# ---------------------------------------------------------------------------

class TestSingleStepExecution:
    """Test that a single simulation step works correctly."""

    def test_step_returns_step_result(self, tmp_path):
        """step() should return a StepResult instance."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        result = scheduler.step()

        assert isinstance(result, StepResult)
        assert result.step == 1
        assert result.band == FrequencyBand.FAST
        assert result.errors == []

    def test_step_processes_fast_band_nodes(self, tmp_path):
        """A single step should process all fast band nodes."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        result = scheduler.step()

        fast_count = len(scheduler._band_nodes[FrequencyBand.FAST])
        assert result.nodes_processed >= fast_count

    def test_step_increments_sim_time(self, tmp_path):
        """Each step should increment sim_time by dt_fast."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        config = SchedulerConfig(dt_fast=0.005)
        scheduler = FrequencyScheduler(tree, config)

        assert scheduler._sim_time == 0.0
        scheduler.step()
        assert scheduler._sim_time == 0.005
        scheduler.step()
        assert scheduler._sim_time == 0.010

    def test_step_with_input_data(self, tmp_path):
        """step() with input_data should pass it to node 0."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        input_signal = Signal(value=42.0, timestamp=0.0)
        result = scheduler.step(input_data=input_signal)

        assert result.step == 1
        assert result.errors == []

    def test_step_without_input_data(self, tmp_path):
        """step() without input_data should use default Signal."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        result = scheduler.step()

        assert result.step == 1
        assert result.errors == []


# ---------------------------------------------------------------------------
# 3. Test multi-step run
# ---------------------------------------------------------------------------

class TestMultiStepRun:
    """Test running multiple simulation steps."""

    def test_run_returns_list_of_results(self, tmp_path):
        """run() should return a list of StepResult instances."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        results = scheduler.run(steps=10)

        assert len(results) == 10
        for i, result in enumerate(results):
            assert isinstance(result, StepResult)
            assert result.step == i + 1

    def test_run_increments_step_counter(self, tmp_path):
        """run() should increment the internal step counter."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        assert scheduler._step == 0

        scheduler.run(steps=5)
        assert scheduler._step == 5

    def test_run_accumulates_sim_time(self, tmp_path):
        """run() should accumulate sim_time correctly."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        config = SchedulerConfig(dt_fast=0.005)
        scheduler = FrequencyScheduler(tree, config)

        scheduler.run(steps=100)
        assert scheduler._sim_time == pytest.approx(0.5, abs=1e-9)

    def test_run_results_stored_internally(self, tmp_path):
        """run() should store results in the internal results list."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        scheduler.run(steps=10)

        assert len(scheduler._results) == 10


# ---------------------------------------------------------------------------
# 4. Test medium band executes every 4 steps
# ---------------------------------------------------------------------------

class TestMediumBandFrequency:
    """Test that medium band nodes execute every 4 steps."""

    @staticmethod
    def _track_process_calls(nodes):
        """Wrap each node's process() to count invocations via a shared dict."""
        call_counts = {}
        for node in nodes:
            call_counts[node.node_id] = 0
            original = node.process

            def tracked(data, _orig=original, _id=node.node_id):
                call_counts[_id] += 1
                return _orig(data)

            node.process = tracked
        return call_counts

    def test_medium_band_on_step_4(self, tmp_path):
        """Medium band should execute on step 4."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        medium_nodes = scheduler._band_nodes[FrequencyBand.MEDIUM]

        if not medium_nodes:
            pytest.skip("No medium band nodes in this tree configuration")

        # Track process calls on medium nodes
        call_counts = self._track_process_calls(medium_nodes)

        # Run 4 steps
        scheduler.run(steps=4)

        # Medium band nodes should have been processed once
        for node in medium_nodes:
            assert call_counts[node.node_id] == 1, (
                f"Medium node {node.node_id} was not processed on step 4"
            )

    def test_medium_band_not_on_step_3(self, tmp_path):
        """Medium band should NOT execute on step 3."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        medium_nodes = scheduler._band_nodes[FrequencyBand.MEDIUM]

        if not medium_nodes:
            pytest.skip("No medium band nodes in this tree configuration")

        # Track process calls on medium nodes
        call_counts = self._track_process_calls(medium_nodes)

        # Run 3 steps
        scheduler.run(steps=3)

        # Medium band nodes should NOT have been processed yet
        for node in medium_nodes:
            assert call_counts[node.node_id] == 0, (
                f"Medium node {node.node_id} was processed before step 4"
            )

    def test_medium_band_executes_twice_in_8_steps(self, tmp_path):
        """Medium band should execute twice in 8 steps (steps 4 and 8)."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        medium_nodes = scheduler._band_nodes[FrequencyBand.MEDIUM]

        if not medium_nodes:
            pytest.skip("No medium band nodes in this tree configuration")

        # Track process calls on medium nodes
        call_counts = self._track_process_calls(medium_nodes)

        scheduler.run(steps=8)

        for node in medium_nodes:
            assert call_counts[node.node_id] == 2, (
                f"Medium node {node.node_id} call_count={call_counts[node.node_id]}, expected 2"
            )

    def test_medium_band_counter_resets(self, tmp_path):
        """Medium band counter should reset after firing."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)

        # After step 4, counter should be reset to 0
        scheduler.run(steps=4)
        assert scheduler._band_counters[FrequencyBand.MEDIUM] == 0

        # After step 5, counter should be 1
        scheduler.step()
        assert scheduler._band_counters[FrequencyBand.MEDIUM] == 1


# ---------------------------------------------------------------------------
# 5. Test slow band executes every 20 steps
# ---------------------------------------------------------------------------

class TestSlowBandFrequency:
    """Test that slow band nodes execute every 20 steps."""

    @staticmethod
    def _track_process_calls(nodes):
        """Wrap each node's process() to count invocations via a shared dict."""
        call_counts = {}
        for node in nodes:
            call_counts[node.node_id] = 0
            original = node.process

            def tracked(data, _orig=original, _id=node.node_id):
                call_counts[_id] += 1
                return _orig(data)

            node.process = tracked
        return call_counts

    def test_slow_band_on_step_20(self, tmp_path):
        """Slow band should execute on step 20."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        slow_nodes = scheduler._band_nodes[FrequencyBand.SLOW]

        if not slow_nodes:
            pytest.skip("No slow band nodes in this tree configuration")

        # Track process calls on slow nodes
        call_counts = self._track_process_calls(slow_nodes)

        scheduler.run(steps=20)

        for node in slow_nodes:
            assert call_counts[node.node_id] == 1, (
                f"Slow node {node.node_id} was not processed on step 20"
            )

    def test_slow_band_not_on_step_19(self, tmp_path):
        """Slow band should NOT execute before step 20."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        slow_nodes = scheduler._band_nodes[FrequencyBand.SLOW]

        if not slow_nodes:
            pytest.skip("No slow band nodes in this tree configuration")

        # Track process calls on slow nodes
        call_counts = self._track_process_calls(slow_nodes)

        scheduler.run(steps=19)

        for node in slow_nodes:
            assert call_counts[node.node_id] == 0, (
                f"Slow node {node.node_id} was processed before step 20"
            )

    def test_slow_band_counter_resets(self, tmp_path):
        """Slow band counter should reset after firing."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)

        scheduler.run(steps=20)
        assert scheduler._band_counters[FrequencyBand.SLOW] == 0

    def test_slow_band_executes_twice_in_40_steps(self, tmp_path):
        """Slow band should execute twice in 40 steps (steps 20 and 40)."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        slow_nodes = scheduler._band_nodes[FrequencyBand.SLOW]

        if not slow_nodes:
            pytest.skip("No slow band nodes in this tree configuration")

        # Track process calls on slow nodes
        call_counts = self._track_process_calls(slow_nodes)

        scheduler.run(steps=40)

        for node in slow_nodes:
            assert call_counts[node.node_id] == 2, (
                f"Slow node {node.node_id} call_count={call_counts[node.node_id]}, expected 2"
            )


# ---------------------------------------------------------------------------
# 6. Test step timing
# ---------------------------------------------------------------------------

class TestStepTiming:
    """Test timing measurements in step results."""

    def test_step_latency_is_non_negative(self, tmp_path):
        """Step latency measurements should be non-negative."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        result = scheduler.step()

        assert result.total_latency_ms >= 0.0
        assert result.max_node_latency_ms >= 0.0

    def test_max_latency_not_greater_than_total(self, tmp_path):
        """Max node latency should not exceed total latency."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        result = scheduler.step()

        assert result.max_node_latency_ms <= result.total_latency_ms + 1e-9

    def test_sim_time_increases_by_dt_fast(self, tmp_path):
        """sim_time in StepResult should increase by dt_fast each step."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        config = SchedulerConfig(dt_fast=0.005)
        scheduler = FrequencyScheduler(tree, config)

        results = scheduler.run(steps=5)
        for i, result in enumerate(results):
            expected_time = (i + 1) * 0.005
            assert result.sim_time == pytest.approx(expected_time, abs=1e-9)

    def test_timing_with_custom_dt(self, tmp_path):
        """Scheduler should respect custom dt_fast value."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        config = SchedulerConfig(dt_fast=0.01)
        scheduler = FrequencyScheduler(tree, config)

        scheduler.run(steps=10)
        assert scheduler._sim_time == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# 7. Test error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test that errors in node processing are captured."""

    def test_error_in_node_captured_in_result(self, tmp_path):
        """Errors during node.process() should be captured in StepResult.errors."""
        json_path = _make_small_tree_json(tmp_path, num_layers=2)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        # Make root node's process raise an error
        original_process = tree.root.process

        def failing_process(data):
            raise RuntimeError("Intentional test error")

        tree.root.process = failing_process

        scheduler = FrequencyScheduler(tree)
        result = scheduler.step()

        assert len(result.errors) > 0
        assert any("Node 0" in e for e in result.errors)

        # Restore
        tree.root.process = original_process

    def test_error_does_not_stop_other_nodes(self, tmp_path):
        """An error in one node should not prevent other nodes from processing."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        fast_nodes = scheduler._band_nodes[FrequencyBand.FAST]

        if len(fast_nodes) < 2:
            pytest.skip("Need at least 2 fast nodes")

        # Make one node fail
        failing_node = fast_nodes[0]
        original_process = failing_node.process

        def failing_process(data):
            raise ValueError("test error")

        failing_node.process = failing_process

        result = scheduler.step()

        # Should have at least one error
        assert len(result.errors) > 0

        # Other nodes should still have been processed
        # (nodes_processed should be >= fast_count - 1)
        assert result.nodes_processed >= len(fast_nodes) - 1

        # Restore
        failing_node.process = original_process

    def test_multiple_errors_captured(self, tmp_path):
        """Multiple errors should all be captured."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        # Make multiple nodes fail
        originals = {}
        for node in list(tree.bfs_traverse())[:3]:
            originals[node.node_id] = node.process

            def failing_process(data):
                raise RuntimeError("multi error")

            node.process = failing_process

        scheduler = FrequencyScheduler(tree)
        result = scheduler.step()

        assert len(result.errors) >= 3

        # Restore
        for node_id, orig in originals.items():
            tree.get_node(node_id).process = orig

    def test_run_continues_despite_errors(self, tmp_path):
        """run() should continue even if some steps have errors."""
        json_path = _make_small_tree_json(tmp_path, num_layers=2)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        # Make root fail
        original_process = tree.root.process

        def failing_process(data):
            raise RuntimeError("persistent error")

        tree.root.process = failing_process

        scheduler = FrequencyScheduler(tree)
        results = scheduler.run(steps=5)

        assert len(results) == 5
        # Every step should have errors
        for result in results:
            assert len(result.errors) > 0

        # Restore
        tree.root.process = original_process


# ---------------------------------------------------------------------------
# 8. Test reset
# ---------------------------------------------------------------------------

class TestReset:
    """Test scheduler and node reset functionality."""

    def test_reset_clears_step_counter(self, tmp_path):
        """reset() should set step counter back to 0."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        scheduler.run(steps=10)
        assert scheduler._step == 10

        scheduler.reset()
        assert scheduler._step == 0

    def test_reset_clears_sim_time(self, tmp_path):
        """reset() should set sim_time back to 0.0."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        scheduler.run(steps=10)

        scheduler.reset()
        assert scheduler._sim_time == 0.0

    def test_reset_clears_results(self, tmp_path):
        """reset() should clear the internal results list."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        scheduler.run(steps=10)
        assert len(scheduler._results) == 10

        scheduler.reset()
        assert len(scheduler._results) == 0

    def test_reset_clears_band_counters(self, tmp_path):
        """reset() should reset all band counters to 0."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        scheduler.run(steps=10)

        scheduler.reset()
        assert scheduler._band_counters[FrequencyBand.FAST] == 0
        assert scheduler._band_counters[FrequencyBand.MEDIUM] == 0
        assert scheduler._band_counters[FrequencyBand.SLOW] == 0

    def test_reset_clears_node_metrics(self, tmp_path):
        """reset() should call reset() on all nodes."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)

        # Track process calls to verify nodes were processed
        call_counts = {}
        for node in tree.bfs_traverse():
            call_counts[node.node_id] = 0
            original = node.process

            def tracked(data, _orig=original, _id=node.node_id):
                call_counts[_id] += 1
                return _orig(data)

            node.process = tracked

        scheduler.run(steps=5)

        # Verify nodes have been processed
        for node in tree.bfs_traverse():
            assert call_counts[node.node_id] > 0

        scheduler.reset()

        # Verify node metrics are cleared (input_queue, state, etc.)
        for node in tree.bfs_traverse():
            assert len(node._input_queue) == 0
            assert len(node._state) == 0

    def test_scheduler_works_after_reset(self, tmp_path):
        """Scheduler should work correctly after a reset."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        scheduler.run(steps=10)
        scheduler.reset()

        # Should work as if starting fresh
        result = scheduler.step()
        assert result.step == 1
        assert result.sim_time == pytest.approx(0.005, abs=1e-9)
        assert result.errors == []


# ---------------------------------------------------------------------------
# 9. Test stats
# ---------------------------------------------------------------------------

class TestStats:
    """Test scheduler statistics."""

    def test_stats_returns_dict(self, tmp_path):
        """get_stats() should return a dict."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        stats = scheduler.get_stats()

        assert isinstance(stats, dict)

    def test_stats_has_required_keys(self, tmp_path):
        """get_stats() should contain all required keys."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        stats = scheduler.get_stats()

        assert "step" in stats
        assert "sim_time" in stats
        assert "total_results" in stats
        assert "band_counts" in stats
        assert "total_nodes" in stats

    def test_stats_total_nodes(self, tmp_path):
        """stats total_nodes should match tree total_nodes."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        stats = scheduler.get_stats()

        assert stats["total_nodes"] == tree.total_nodes

    def test_stats_after_steps(self, tmp_path):
        """stats should reflect the current state after running steps."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        config = SchedulerConfig(dt_fast=0.005)
        scheduler = FrequencyScheduler(tree, config)
        scheduler.run(steps=50)

        stats = scheduler.get_stats()
        assert stats["step"] == 50
        assert stats["sim_time"] == pytest.approx(0.25, abs=1e-9)
        assert stats["total_results"] == 50

    def test_stats_band_counts_sum_to_total(self, tmp_path):
        """Sum of band_counts should equal total_nodes."""
        json_path = _make_mixed_protocol_json(tmp_path, num_layers=4)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        stats = scheduler.get_stats()

        band_sum = sum(stats["band_counts"].values())
        assert band_sum == stats["total_nodes"]

    def test_stats_initial_state(self, tmp_path):
        """Initial stats should show step=0, sim_time=0.0."""
        json_path = _make_small_tree_json(tmp_path, num_layers=3)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        stats = scheduler.get_stats()

        assert stats["step"] == 0
        assert stats["sim_time"] == 0.0
        assert stats["total_results"] == 0


# ---------------------------------------------------------------------------
# 10. Test with full 2047-node tree (run 10 steps)
# ---------------------------------------------------------------------------

class TestFullTreeScheduler:
    """Test scheduler with the full 2047-node tree."""

    def test_full_tree_classification(self, tmp_path):
        """All 2047 nodes should be classified into bands."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        stats = scheduler.get_stats()

        total_classified = sum(stats["band_counts"].values())
        assert total_classified == 2047

    def test_full_tree_run_10_steps(self, tmp_path):
        """Running 10 steps on the full tree should complete without errors."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        results = scheduler.run(steps=10)

        assert len(results) == 10
        for result in results:
            assert result.errors == [], f"Step {result.step} had errors: {result.errors}"

    def test_full_tree_all_fast_nodes_processed_each_step(self, tmp_path):
        """All fast band nodes should be processed on every step."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        fast_count = len(scheduler._band_nodes[FrequencyBand.FAST])

        results = scheduler.run(steps=10)

        # All 2047 nodes have ASYNC_PUSH, so all are in fast band
        # nodes_processed should be at least fast_count on every step
        for result in results:
            assert result.nodes_processed >= fast_count

    def test_full_tree_sim_time_after_10_steps(self, tmp_path):
        """Sim time after 10 steps should be 10 * dt_fast."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        config = SchedulerConfig(dt_fast=0.005)
        scheduler = FrequencyScheduler(tree, config)
        scheduler.run(steps=10)

        assert scheduler._sim_time == pytest.approx(0.05, abs=1e-9)

    def test_full_tree_step_counter(self, tmp_path):
        """Step counter should be 10 after running 10 steps."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        scheduler.run(steps=10)

        assert scheduler._step == 10

    def test_full_tree_stats(self, tmp_path):
        """Stats should be correct after running on full tree."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        scheduler.run(steps=10)

        stats = scheduler.get_stats()
        assert stats["step"] == 10
        assert stats["total_nodes"] == 2047
        assert stats["total_results"] == 10

    def test_full_tree_reset_and_rerun(self, tmp_path):
        """After reset, running again should work from scratch."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        scheduler.run(steps=10)
        scheduler.reset()

        results = scheduler.run(steps=10)
        assert len(results) == 10
        assert scheduler._step == 10
        assert results[0].step == 1

    def test_full_tree_latency_within_bounds(self, tmp_path):
        """Step latency should be reasonable for 2047 nodes."""
        json_path = _make_full_2047_json(tmp_path)
        builder = BrainTreeBuilder()
        tree = builder.build_from_json(str(json_path))

        scheduler = FrequencyScheduler(tree)
        results = scheduler.run(steps=10)

        for result in results:
            # Each step should complete in a reasonable time
            # (allow generous margin for CI environments)
            assert result.total_latency_ms < 5000.0, (
                f"Step {result.step} took too long: {result.total_latency_ms}ms"
            )
