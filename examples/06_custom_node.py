"""Example 6: Custom Node Implementation

Learn how to write a custom BrainNode subclass with specialized
processing logic. This example shows the template method pattern:
override _process_tensor, _process_signal, etc. to customize behavior.

Usage:
    python examples/06_custom_node.py
"""

import numpy as np
from brain_agent import BrainNode, NodeConfig, RegionType, Protocol
from brain_agent.types import BrainData, Tensor, StateVector, Signal
from brain_agent.testing import make_node_config, build_small_tree, make_test_signal
from brain_agent.scheduler import FrequencyScheduler


class ThresholdNode(BrainNode):
    """A custom node that applies a threshold to input signals.

    Signals below the threshold are suppressed (set to 0).
    Signals above the threshold are passed through with a gain factor.
    """

    def _setup_ports(self):
        """Define input/output ports for this node.

        Ports are set directly on _input_ports and _output_ports dicts.
        The base class _setup_ports() already sets up default ports
        based on region_type, so we only override if we need custom ports.
        """
        super()._setup_ports()

    def _process_signal(self, data: BrainData) -> BrainData:
        """Apply threshold to Signal input.

        This overrides the base class hook. The base class process()
        method handles timing, metrics, and dispatch automatically.
        """
        threshold = self.get_state("threshold", 0.5)
        gain = self.get_state("gain", 2.0)

        if data.value < threshold:
            return Signal(value=0.0, timestamp=data.timestamp)
        else:
            return Signal(value=data.value * gain, timestamp=data.timestamp)

    def _process_tensor(self, data: BrainData) -> BrainData:
        """Apply threshold to Tensor input (element-wise)."""
        threshold = self.get_state("threshold", 0.5)
        gain = self.get_state("gain", 2.0)
        result = np.where(data.data > threshold, data.data * gain, 0.0)
        return Tensor(data=result.astype(np.float32), shape=result.shape)


class AccumulatorNode(BrainNode):
    """A custom node that accumulates signal values over time.

    Maintains a running sum of all received signal values.
    Useful for integrating signals across simulation steps.
    """

    def _process_signal(self, data: BrainData) -> BrainData:
        """Accumulate signal value and return running total."""
        current_sum = self.get_state("running_sum", 0.0)
        current_sum += data.value
        self.set_state("running_sum", current_sum)
        return Signal(value=current_sum, timestamp=data.timestamp)


def main():
    print("=" * 60)
    print("Custom Node Implementation Example")
    print("=" * 60)

    # --- Example 1: ThresholdNode ---
    print("\n1. ThresholdNode: Suppress signals below threshold")
    print("-" * 50)

    config = make_node_config(
        node_id=0,
        short_name="Threshold",
        region_type="relay",
    )
    threshold_node = ThresholdNode(config)
    threshold_node.set_state("threshold", 0.5)
    threshold_node.set_state("gain", 2.0)

    # Test with various signal values
    test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for val in test_values:
        input_signal = Signal(value=val, timestamp=0.0)
        output_signal = threshold_node.process(input_signal)
        status = "PASSED" if output_signal.value > 0 else "SUPPRESSED"
        print(f"  Input: {val:.1f} -> Output: {output_signal.value:.2f} [{status}]")

    # --- Example 2: AccumulatorNode ---
    print("\n2. AccumulatorNode: Running sum over time")
    print("-" * 50)

    config2 = make_node_config(
        node_id=1,
        short_name="Accumulator",
        region_type="memory",
    )
    acc_node = AccumulatorNode(config2)

    for step in range(5):
        input_signal = Signal(value=1.0, timestamp=step * 0.005)
        output_signal = acc_node.process(input_signal)
        print(f"  Step {step}: input=1.0 -> running_sum={output_signal.value:.1f}")

    # --- Example 3: Using custom nodes in a simulation ---
    print("\n3. Custom nodes in a small simulation tree")
    print("-" * 50)

    # Build a small tree (15 nodes, 4 layers)
    tree = build_small_tree(n_layers=4)
    print(f"  Built tree with {tree.total_nodes} nodes")

    # Replace root node with our ThresholdNode
    root_config = make_node_config(
        node_id=0,
        short_name="ThresholdRoot",
        region_type="cognitive",
    )
    custom_root = ThresholdNode(root_config)
    custom_root.set_state("threshold", 0.3)
    custom_root.set_state("gain", 1.5)

    # Run a few steps
    scheduler = FrequencyScheduler(tree)
    for step in range(3):
        input_signal = Signal(value=0.5 + step * 0.1, timestamp=step * 0.005)
        result = scheduler.step(input_data=input_signal)
        print(f"  Step {step}: {result.nodes_processed} nodes processed, "
              f"{result.total_latency_ms:.1f}ms")

    print("\n" + "=" * 60)
    print("Key takeaways:")
    print("  - Override _process_signal, _process_tensor, _process_state_vector")
    print("  - Use get_state()/set_state() for node-internal state")
    print("  - The base class process() handles timing and metrics")
    print("  - Use brain_agent.testing for test utilities")
    print("=" * 60)


if __name__ == "__main__":
    main()
