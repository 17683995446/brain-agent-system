"""Tests for monitoring and observability (Phase 10).

Tests cover MetricsCollector, Tracer, and integration with scheduler.
"""

import time

import pytest

from brain_agent.node import BrainNode, NodeConfig, RegionType, Protocol
from brain_agent.tree import BrainTree, PassThroughBrainNode
from brain_agent.scheduler import FrequencyScheduler, SchedulerConfig, StepResult, FrequencyBand
from brain_agent.types import Signal
from brain_agent.monitoring.metrics import MetricsCollector, MetricPoint
from brain_agent.monitoring.tracer import Tracer, TraceSpan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(node_id: int = 0, layer: int = 0, region_type=RegionType.RELAY,
                 protocol=Protocol.ASYNC_PUSH, parent_id=None):
    return NodeConfig(
        node_id=node_id,
        name_cn="TestNode",
        short_name=f"Node{node_id}",
        folder_name=f"L{layer:02d}_Node{node_id}",
        layer=layer,
        parent_id=parent_id,
        responsibility="test",
        region_type=region_type,
        tensor_shape="",
        frequency="100Hz",
        protocol=protocol,
        algorithm="passthrough",
        children_ids=[],
    )


def _build_small_tree():
    """Build a 3-node tree: root (id=0) -> child1 (id=1), child2 (id=2)."""
    root_cfg = _make_config(node_id=0, layer=0)
    child1_cfg = _make_config(node_id=1, layer=1, parent_id=0)
    child2_cfg = _make_config(node_id=2, layer=1, parent_id=0)

    root = PassThroughBrainNode(root_cfg)
    child1 = PassThroughBrainNode(child1_cfg)
    child2 = PassThroughBrainNode(child2_cfg)

    root.connect_child(child1)
    root.connect_child(child2)

    nodes = {0: root, 1: child1, 2: child2}
    return BrainTree(nodes)


# ---------------------------------------------------------------------------
# 1. Test MetricsCollector creation
# ---------------------------------------------------------------------------

class TestMetricsCollectorCreation:
    def test_create_without_tree(self):
        collector = MetricsCollector()
        assert collector._tree is None
        assert collector.get_all_latest() == {}

    def test_create_with_tree(self):
        tree = _build_small_tree()
        collector = MetricsCollector(tree=tree)
        assert collector._tree is tree
        assert collector.get_all_latest() == {}


# ---------------------------------------------------------------------------
# 2. Test collect_node_metrics
# ---------------------------------------------------------------------------

class TestCollectNodeMetrics:
    def test_collect_from_empty_collector(self):
        collector = MetricsCollector()
        collector.collect_node_metrics()  # should not raise
        assert collector.get_all_latest() == {}

    def test_collect_from_tree(self):
        tree = _build_small_tree()
        collector = MetricsCollector(tree=tree)
        collector.collect_node_metrics()

        # Should have collected metrics for 3 nodes
        latency_points = collector.get_metric("brain_node_processing_latency_ms")
        assert len(latency_points) == 3

        count_points = collector.get_metric("brain_node_process_count")
        assert len(count_points) == 3

        queue_points = collector.get_metric("brain_node_queue_depth")
        assert len(queue_points) == 3

    def test_collect_node_metrics_labels(self):
        tree = _build_small_tree()
        collector = MetricsCollector(tree=tree)
        collector.collect_node_metrics()

        latency_points = collector.get_metric("brain_node_processing_latency_ms")
        for pt in latency_points:
            assert "node_id" in pt.labels
            assert "layer" in pt.labels
            assert "type" in pt.labels


# ---------------------------------------------------------------------------
# 3. Test collect_step_result
# ---------------------------------------------------------------------------

class TestCollectStepResult:
    def test_collect_step_result_basic(self):
        collector = MetricsCollector()
        result = StepResult(
            step=1,
            sim_time=0.005,
            band=FrequencyBand.FAST,
            nodes_processed=3,
            total_latency_ms=1.5,
            max_node_latency_ms=0.8,
            errors=[],
        )
        collector.collect_step_result(result)

        assert collector.get_latest("brain_system_step_latency_ms").value == 1.5
        assert collector.get_latest("brain_system_nodes_processed").value == 3
        assert collector.get_latest("brain_system_errors_total") is None

    def test_collect_step_result_with_errors(self):
        collector = MetricsCollector()
        result = StepResult(
            step=2,
            sim_time=0.01,
            band=FrequencyBand.FAST,
            nodes_processed=2,
            total_latency_ms=2.0,
            max_node_latency_ms=1.5,
            errors=["Node 5: timeout", "Node 8: overflow"],
        )
        collector.collect_step_result(result)

        assert collector.get_latest("brain_system_errors_total").value == 2


# ---------------------------------------------------------------------------
# 4. Test get_latest
# ---------------------------------------------------------------------------

class TestGetLatest:
    def test_get_latest_empty(self):
        collector = MetricsCollector()
        assert collector.get_latest("nonexistent") is None

    def test_get_latest_returns_last(self):
        collector = MetricsCollector()
        collector._add_metric("test_metric", 10.0, time.time())
        collector._add_metric("test_metric", 20.0, time.time())
        collector._add_metric("test_metric", 30.0, time.time())

        latest = collector.get_latest("test_metric")
        assert latest is not None
        assert latest.value == 30.0


# ---------------------------------------------------------------------------
# 5. Test get_all_latest
# ---------------------------------------------------------------------------

class TestGetAllLatest:
    def test_get_all_latest_empty(self):
        collector = MetricsCollector()
        assert collector.get_all_latest() == {}

    def test_get_all_latest_multiple_metrics(self):
        collector = MetricsCollector()
        collector._add_metric("metric_a", 1.0, time.time())
        collector._add_metric("metric_a", 2.0, time.time())
        collector._add_metric("metric_b", 5.0, time.time())

        all_latest = collector.get_all_latest()
        assert all_latest["metric_a"] == 2.0
        assert all_latest["metric_b"] == 5.0


# ---------------------------------------------------------------------------
# 6. Test export_prometheus format
# ---------------------------------------------------------------------------

class TestExportPrometheus:
    def test_export_empty(self):
        collector = MetricsCollector()
        assert collector.export_prometheus() == ""

    def test_export_with_labels(self):
        collector = MetricsCollector()
        collector._add_metric(
            "brain_node_processing_latency_ms",
            0.5,
            time.time(),
            {"node_id": "1", "layer": "0", "type": "relay"}
        )

        output = collector.export_prometheus()
        assert 'brain_node_processing_latency_ms' in output
        assert 'node_id="1"' in output
        assert 'layer="0"' in output
        assert 'type="relay"' in output
        assert '0.5' in output

    def test_export_without_labels(self):
        collector = MetricsCollector()
        collector._add_metric("brain_system_step_latency_ms", 1.5, time.time())

        output = collector.export_prometheus()
        assert output == "brain_system_step_latency_ms 1.5"

    def test_export_multiple_latest(self):
        collector = MetricsCollector()
        collector._add_metric("metric_a", 10.0, time.time(), {"k": "v"})
        collector._add_metric("metric_a", 20.0, time.time(), {"k": "v2"})
        collector._add_metric("metric_b", 5.0, time.time())

        output = collector.export_prometheus()
        lines = output.strip().split("\n")
        assert len(lines) == 2
        # metric_a should show latest (20.0) with labels k="v2"
        assert 'metric_a{k="v2"} 20.0' in output
        assert 'metric_b 5.0' in output


# ---------------------------------------------------------------------------
# 7. Test Tracer start_span / finish_span
# ---------------------------------------------------------------------------

class TestTracerBasic:
    def test_start_span(self):
        tracer = Tracer()
        span = tracer.start_span(node_id=0, node_name="Root", layer=0)
        assert span.node_id == 0
        assert span.node_name == "Root"
        assert span.layer == 0
        assert span.start_time > 0

    def test_finish_span(self):
        tracer = Tracer()
        span = tracer.start_span(node_id=0, node_name="Root", layer=0)
        time.sleep(0.01)  # small delay
        tracer.finish_span(span)

        assert span.end_time > 0
        assert span.duration_ms > 0
        assert span.error is None

    def test_finish_span_with_error(self):
        tracer = Tracer()
        span = tracer.start_span(node_id=1, node_name="Child", layer=1)
        tracer.finish_span(span, error="timeout")

        assert span.error == "timeout"

    def test_disabled_tracer(self):
        tracer = Tracer()
        tracer._enabled = False
        span = tracer.start_span(node_id=0, node_name="Root", layer=0)
        tracer.finish_span(span)

        assert span.start_time == 0
        assert span.duration_ms == 0.0
        assert len(tracer.get_traces()) == 0


# ---------------------------------------------------------------------------
# 8. Test Tracer format_trace
# ---------------------------------------------------------------------------

class TestTracerFormat:
    def test_format_empty(self):
        tracer = Tracer()
        assert tracer.format_trace() == "No traces"

    def test_format_single_span(self):
        tracer = Tracer()
        span = tracer.start_span(node_id=0, node_name="Root", layer=0)
        tracer.finish_span(span)

        output = tracer.format_trace()
        assert "Trace: Root (L00)" in output
        assert "Root (L00)" in output
        assert "ms" in output

    def test_format_with_error(self):
        tracer = Tracer()
        span = tracer.start_span(node_id=0, node_name="Root", layer=0)
        tracer.finish_span(span, error="crash")

        output = tracer.format_trace()
        assert "ERROR: crash" in output


# ---------------------------------------------------------------------------
# 9. Test Tracer with nested spans
# ---------------------------------------------------------------------------

class TestTracerNested:
    def test_nested_spans(self):
        tracer = Tracer()
        root = tracer.start_span(node_id=0, node_name="Root", layer=0)
        child1 = tracer.start_span(node_id=1, node_name="Child1", layer=1)
        tracer.finish_span(child1)
        child2 = tracer.start_span(node_id=2, node_name="Child2", layer=1)
        tracer.finish_span(child2)
        tracer.finish_span(root)

        traces = tracer.get_traces()
        assert len(traces) == 1
        assert len(traces[0]["root"].children) == 2

    def test_format_nested(self):
        tracer = Tracer()
        root = tracer.start_span(node_id=0, node_name="Root", layer=0)
        child = tracer.start_span(node_id=1, node_name="Child", layer=1)
        tracer.finish_span(child)
        tracer.finish_span(root)

        output = tracer.format_trace()
        lines = output.strip().split("\n")
        # Root header line + root span detail + child span detail
        assert len(lines) == 3
        assert "Trace: Root (L00)" in lines[0]
        assert "Child (L01)" in lines[2]

    def test_deeply_nested(self):
        tracer = Tracer()
        s0 = tracer.start_span(node_id=0, node_name="L0", layer=0)
        s1 = tracer.start_span(node_id=1, node_name="L1", layer=1)
        s2 = tracer.start_span(node_id=2, node_name="L2", layer=2)
        tracer.finish_span(s2)
        tracer.finish_span(s1)
        tracer.finish_span(s0)

        traces = tracer.get_traces()
        assert len(traces) == 1
        root = traces[0]["root"]
        assert len(root.children) == 1
        assert len(root.children[0].children) == 1

    def test_clear_traces(self):
        tracer = Tracer()
        span = tracer.start_span(node_id=0, node_name="Root", layer=0)
        tracer.finish_span(span)
        assert len(tracer.get_traces()) == 1

        tracer.clear()
        assert len(tracer.get_traces()) == 0
        assert tracer.format_trace() == "No traces"


# ---------------------------------------------------------------------------
# 10. Test integration: run scheduler with metrics collection
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_scheduler_with_metrics_collector(self):
        tree = _build_small_tree()
        scheduler = FrequencyScheduler(tree, config=SchedulerConfig())
        collector = MetricsCollector(tree=tree)

        # Run a few steps and collect metrics
        for _ in range(5):
            result = scheduler.step()
            collector.collect_step_result(result)

        collector.collect_node_metrics()

        # Verify step metrics were collected
        step_latency = collector.get_metric("brain_system_step_latency_ms")
        assert len(step_latency) == 5

        nodes_processed = collector.get_metric("brain_system_nodes_processed")
        assert len(nodes_processed) == 5

        # Verify node metrics were collected
        node_latency = collector.get_metric("brain_node_processing_latency_ms")
        assert len(node_latency) == 3  # 3 nodes

    def test_scheduler_with_tracer(self):
        tree = _build_small_tree()
        scheduler = FrequencyScheduler(tree, config=SchedulerConfig())
        tracer = Tracer()

        # Trace a single step through the root node
        root_span = tracer.start_span(
            node_id=0, node_name="Root", layer=0, data_type="Signal"
        )
        scheduler.step()
        tracer.finish_span(root_span)

        traces = tracer.get_traces()
        assert len(traces) == 1
        assert traces[0]["root"].duration_ms > 0

        output = tracer.format_trace()
        assert "Root (L00)" in output

    def test_full_integration(self):
        tree = _build_small_tree()
        scheduler = FrequencyScheduler(tree, config=SchedulerConfig())
        collector = MetricsCollector(tree=tree)
        tracer = Tracer()

        # Run simulation with monitoring
        for i in range(3):
            root_span = tracer.start_span(
                node_id=0, node_name="Root", layer=0, data_type="Signal"
            )
            result = scheduler.step()
            collector.collect_step_result(result)
            tracer.finish_span(root_span)

        collector.collect_node_metrics()

        # Verify all components collected data
        assert len(tracer.get_traces()) == 3
        assert len(collector.get_metric("brain_system_step_latency_ms")) == 3
        assert len(collector.get_metric("brain_node_processing_latency_ms")) == 3

        # Prometheus export should have content
        prom_output = collector.export_prometheus()
        assert len(prom_output) > 0
        assert "brain_system_step_latency_ms" in prom_output

    def test_metrics_collector_clear(self):
        collector = MetricsCollector()
        collector._add_metric("test", 1.0, time.time())
        assert len(collector.get_all_latest()) == 1

        collector.clear()
        assert collector.get_all_latest() == {}
