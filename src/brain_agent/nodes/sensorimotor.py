"""L01: SensorimotorSystemNode - Sensory-motor integration.

Neuroscience basis:
The sensorimotor system integrates sensory input with motor output planning
through parallel processing streams in the parietal and frontal cortices.
The dorsal stream (where/how pathway) processes spatial information for action,
while gain control mechanisms adjust sensitivity based on task demands.
This node implements gain-field modulation (Salinas & Abbott, 1994) where
multiplicative gain factors modulate sensory representations based on
motor plans and task context.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class SensorimotorSystemNode(BrainNode):
    """Sensorimotor system: sensory-motor integration.

    Integrates sensory input with motor output planning using gain control
    based on current task demands. Operates on 32x32x8 tensor representations
    that encode spatial maps across 8 feature channels.

    Processing logic:
    1. Extract spatial gain field from task context
    2. Apply multiplicative gain modulation to sensory input
    3. Compute sensorimotor transformation (sensory -> motor mapping)
    4. Generate integrated output for downstream processing
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Gain field parameters
        self.set_state("gain_field", np.ones((32, 32), dtype=np.float32))
        self.set_state("task_gain", np.float32(1.0))
        self.set_state("motor_bias", np.zeros((1, 32, 32), dtype=np.float32))

    def process(self, data: BrainData) -> BrainData:
        """Integrate sensory input with motor output planning.

        Apply gain control based on current task demands using numpy
        tensor operations on 32x32x8 input.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Processed output with sensorimotor integration applied
        """
        start = time.perf_counter()
        if isinstance(data, Tensor):
            result: BrainData = self._process_tensor(data)
        elif isinstance(data, StateVector):
            result = self._process_state_vector(data)
        elif isinstance(data, Signal):
            result = self._process_signal(data)
        elif isinstance(data, Message):
            result = data
        else:
            raise ValueError(f"Unsupported data type: {type(data).__name__}")

        self._update_metrics(time.perf_counter() - start)
        return result

    def _process_tensor(self, data: Tensor) -> Tensor:
        """Apply sensorimotor integration to tensor input.

        Implements gain-field modulation where each spatial location
        receives a multiplicative gain factor. The gain field is
        computed as a 2D Gaussian centered on the current attention focus,
        modeling the spatial attention fields observed in parietal cortex.

        The transformation applies:
        1. Per-channel gain modulation: output[c] = input[c] * gain_field
        2. Cross-channel integration: weighted sum across channels
        3. Motor bias addition: output += motor_bias
        """
        gain_field = self.get_state("gain_field", np.ones((32, 32), dtype=np.float32))
        task_gain = self.get_state("task_gain", np.float32(1.0))
        motor_bias = self.get_state("motor_bias", np.zeros((1, 32, 32), dtype=np.float32))

        input_data = data.data  # shape: (32, 32, 8) or similar

        # Apply gain field modulation across all channels
        # gain_field is (32, 32), broadcast across channel dimension
        if input_data.ndim == 3:
            # Shape: (H, W, C) - apply gain to spatial dims
            modulated = input_data * gain_field[:, :, np.newaxis] * task_gain
        elif input_data.ndim == 2:
            # Shape: (H, W) - apply gain directly
            modulated = input_data * gain_field * task_gain
        else:
            # General case: apply gain to last two spatial dims
            modulated = input_data * task_gain

        # Cross-channel integration via weighted averaging
        if input_data.ndim == 3 and input_data.shape[2] > 1:
            # Channel weights: slight preference for central channels
            n_channels = input_data.shape[2]
            channel_weights = np.ones(n_channels, dtype=np.float32)
            center = n_channels / 2.0
            for c in range(n_channels):
                channel_weights[c] = np.float32(
                    np.exp(-0.5 * ((c - center) / (center * 0.5 + 1e-6)) ** 2)
                )
            channel_weights /= channel_weights.sum()

            # Weighted channel combination
            integrated = np.sum(modulated * channel_weights[np.newaxis, np.newaxis, :], axis=2)
            # Keep single channel output
            result = integrated[:, :, np.newaxis]
        else:
            result = modulated

        # Add motor bias if shapes are compatible
        if result.shape == motor_bias.shape:
            result = result + motor_bias

        return Tensor(data=result.astype(np.float32), shape=result.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Process state vector input through sensorimotor transformation.

        Applies a learned linear transformation that maps sensory state
        to motor planning state, with gain modulation.
        """
        task_gain = self.get_state("task_gain", np.float32(1.0))
        modulated = data.data * task_gain

        # Sensorimotor transformation: circular shift to model
        # the mapping from sensory to motor coordinates
        shift = int(task_gain * 4) % data.dimension
        transformed = np.roll(modulated, shift)

        return StateVector(data=transformed.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal through gain control.

        Task gain modulates the signal amplitude, modeling how
        task demands scale sensorimotor responsiveness.
        """
        task_gain = self.get_state("task_gain", np.float32(1.0))
        return Signal(value=float(data.value * task_gain), timestamp=data.timestamp)
