"""L02: MotorPathwayNode - Movement planning and execution.

Neuroscience basis:
The motor pathway implements the transformation from motor plans to muscle
activation patterns, following the cortical hierarchy: supplementary motor
area (SMA) -> premotor cortex -> primary motor cortex (M1) -> spinal cord.
M1 uses a population vector code (Georgopoulos et al., 1986) where the
direction of movement is encoded by the weighted vector sum of neuronal
preferred directions. Force and velocity are governed by the size principle
(Henneman, 1957) where motor units are recruited in order of increasing size.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class MotorPathwayNode(BrainNode):
    """Motor pathway: movement planning and execution.

    Converts motor plans to muscle activation patterns with force/velocity
    scaling. Operates on Tensor[float32, 1x32x32] representations encoding
    the motor map (2D spatial layout of muscle groups).

    Processing logic:
    1. Population vector decoding from motor plan
    2. Force/velocity scaling via the size principle
    3. Muscle activation pattern generation
    4. Temporal smoothing for coordinated movement
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Motor unit properties
        self.set_state("force_gain", np.float32(1.0))
        self.set_state("velocity_gain", np.float32(1.0))
        self.set_state("muscle_tone", np.float32(0.1))
        # Previous activation for temporal smoothing
        self.set_state("prev_activation", np.zeros((1, 32, 32), dtype=np.float32))
        # Population vector directions (preferred directions for motor units)
        self.set_state("preferred_dirs", self._init_preferred_directions())

    def _init_preferred_directions(self) -> np.ndarray:
        """Initialize preferred directions for population coding.

        Each spatial location in the motor map has a preferred movement
        direction, uniformly distributed across 360 degrees.
        """
        dirs = np.zeros((32, 32, 2), dtype=np.float32)
        for i in range(32):
            for j in range(32):
                angle = np.float32(2.0 * np.pi * (i * 32 + j) / (32 * 32))
                dirs[i, j, 0] = np.cos(angle)
                dirs[i, j, 1] = np.sin(angle)
        return dirs

    def process(self, data: BrainData) -> BrainData:
        """Convert motor plans to muscle activation patterns.

        Apply force/velocity scaling to Tensor[float32, 1x32x32] operations.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Motor activation output
        """
        start = time.perf_counter()
        if isinstance(data, Tensor):
            result = self._process_tensor(data)
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
        """Convert motor plan tensor to muscle activation pattern.

        Steps:
        1. Population vector decoding: weighted sum of preferred directions
        2. Force scaling: apply gain based on signal magnitude
        3. Velocity scaling: temporal derivative component
        4. Temporal smoothing: exponential moving average
        """
        input_data = data.data
        force_gain = self.get_state("force_gain", np.float32(1.0))
        velocity_gain = self.get_state("velocity_gain", np.float32(1.0))
        muscle_tone = self.get_state("muscle_tone", np.float32(0.1))
        prev_activation = self.get_state("prev_activation", np.zeros((1, 32, 32), dtype=np.float32))

        # Ensure 3D shape (1, 32, 32)
        if input_data.ndim == 2:
            activation_map = input_data[np.newaxis, :, :]
        else:
            activation_map = input_data

        # Population vector decoding: compute movement direction
        if activation_map.shape[1] == 32 and activation_map.shape[2] == 32:
            preferred_dirs = self.get_state("preferred_dirs")
            activity = activation_map[0]  # (32, 32)

            # Weighted sum of preferred directions
            pv_x = np.sum(activity * preferred_dirs[:, :, 0])
            pv_y = np.sum(activity * preferred_dirs[:, :, 1])
            movement_direction = np.arctan2(pv_y, pv_x)

            # Directional gain: boost activation in movement direction
            dir_gain = np.float32(1.0 + 0.2 * np.cos(
                movement_direction -
                np.arctan2(preferred_dirs[:, :, 1], preferred_dirs[:, :, 0])
            ))
            activation_map[0] = activation_map[0] * dir_gain

        # Force scaling (size principle: recruit larger units for more force)
        force_scaled = activation_map * force_gain

        # Velocity component: difference from previous activation
        if prev_activation.shape == force_scaled.shape:
            velocity_component = (force_scaled - prev_activation) * velocity_gain
        else:
            velocity_component = force_scaled * velocity_gain * 0.1

        # Combined activation with muscle tone baseline
        combined = force_scaled + velocity_component + muscle_tone

        # Temporal smoothing (exponential moving average)
        alpha = np.float32(0.7)
        smoothed = prev_activation * alpha + combined * (1.0 - alpha) \
            if prev_activation.shape == combined.shape else combined

        # Rectify: muscle activation cannot be negative
        output = np.maximum(smoothed, 0.0)

        self.set_state("prev_activation", output.copy())

        return Tensor(data=output.astype(np.float32), shape=output.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Process state vector as motor plan.

        Applies force scaling and rectification to generate
        muscle activation state vector.
        """
        force_gain = self.get_state("force_gain", np.float32(1.0))
        modulated = np.maximum(data.data * force_gain, 0.0)
        return StateVector(data=modulated.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal as motor command amplitude.

        Force gain scales the command, and muscle tone provides baseline.
        """
        force_gain = self.get_state("force_gain", np.float32(1.0))
        muscle_tone = self.get_state("muscle_tone", np.float32(0.1))
        output = max(0.0, data.value * force_gain) + muscle_tone
        return Signal(value=output, timestamp=data.timestamp)
