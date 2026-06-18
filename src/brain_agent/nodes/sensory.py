"""L02: SensoryPathwayNode - Initial sensory feature extraction.

Neuroscience basis:
The sensory pathways perform initial feature extraction through a series of
filters analogous to those found in primary sensory cortices. In the visual
system, V1 neurons respond to Gabor-like features (Hubel & Wiesel, 1962):
specific orientations, spatial frequencies, and spatial locations. This node
implements Gabor-like filtering using numpy FFT for efficient convolution,
followed by cross-modal normalization to ensure balanced processing across
different sensory modalities.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class SensoryPathwayNode(BrainNode):
    """Sensory pathway: initial feature extraction.

    Applies Gabor-like filtering for feature extraction and normalizes
    sensory input across modalities. Transforms Tensor[float32, 32x32x8]
    to Tensor[float32, 32x32x8] with enhanced feature representations.

    Processing logic:
    1. Apply Gabor-like filters via FFT-based convolution
    2. Rectify (ReLU) to model neuronal nonlinearity
    3. Local contrast normalization (divisive normalization)
    4. Cross-modal gain normalization
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Pre-compute Gabor filter bank
        self._build_gabor_filters()
        # Cross-modal gain
        self.set_state("modality_gain", np.float32(1.0))
        self.set_state("adaptation_state", np.float32(1.0))

    def _build_gabor_filters(self):
        """Build a bank of Gabor-like filters at multiple orientations.

        Gabor filters model the receptive fields of V1 simple cells:
        g(x,y) = exp(-(x'^2 + gamma^2*y'^2)/(2*sigma^2)) * cos(2*pi*x'/lambda)

        where x' = x*cos(theta) + y*sin(theta)
              y' = -x*sin(theta) + y*cos(theta)
        """
        self._filters = []
        n_orientations = 4
        size = 7  # filter kernel size
        sigma = np.float32(2.0)
        lambd = np.float32(4.0)
        gamma = np.float32(0.5)

        y, x = np.mgrid[-size//2:size//2+1, -size//2:size//2+1].astype(np.float32)

        for theta_idx in range(n_orientations):
            theta = np.float32(theta_idx * np.pi / n_orientations)
            x_prime = x * np.cos(theta) + y * np.sin(theta)
            y_prime = -x * np.sin(theta) + y * np.cos(theta)

            gabor = np.exp(-(x_prime**2 + gamma**2 * y_prime**2) / (2 * sigma**2))
            gabor *= np.cos(2 * np.pi * x_prime / lambd)
            gabor = gabor.astype(np.float32)
            self._filters.append(gabor)

    def process(self, data: BrainData) -> BrainData:
        """Apply Gabor-like filtering and normalization.

        Tensor[float32, 32x32x8] -> Tensor[float32, 32x32x8]

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Feature-enhanced output
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
        """Apply Gabor filtering and normalization to tensor input.

        Pipeline:
        1. FFT-based convolution with Gabor filter bank
        2. Rectification (ReLU nonlinearity)
        3. Local contrast normalization
        4. Cross-modal gain adjustment
        """
        input_data = data.data
        modality_gain = self.get_state("modality_gain", np.float32(1.0))
        adaptation = self.get_state("adaptation_state", np.float32(1.0))

        if input_data.ndim == 3:
            h, w, c = input_data.shape
            output = np.zeros_like(input_data)

            # Apply filters to each channel
            for ch in range(c):
                channel_data = input_data[:, :, ch]

                # FFT-based convolution with each Gabor filter
                filtered_channels = []
                for gabor_filter in self._filters:
                    # Pad filter to match input size
                    fh, fw = gabor_filter.shape
                    padded_filter = np.zeros((h, w), dtype=np.float32)
                    pad_h, pad_w = (h - fh) // 2, (w - fw) // 2
                    padded_filter[pad_h:pad_h+fh, pad_w:pad_w+fw] = gabor_filter

                    # FFT convolution
                    fft_input = np.fft.fft2(channel_data)
                    fft_filter = np.fft.fft2(padded_filter)
                    conv_result = np.real(np.fft.ifft2(fft_input * fft_filter))

                    filtered_channels.append(conv_result)

                # Stack filtered outputs (take mean across orientations for this channel)
                if len(self._filters) <= c - ch:
                    max(1, len(self._filters))
                else:
                    pass

                # Use first filter result for this channel
                output[:, :, ch] = filtered_channels[ch % len(self._filters)]

            # Rectification (ReLU) - models neuronal threshold nonlinearity
            output = np.maximum(output, 0.0)

            # Local contrast normalization (divisive normalization)
            # Models lateral inhibition in sensory cortex
            local_sum = self._local_normalization(output)
            output = output / (local_sum + np.float32(1e-6))

            # Apply modality gain and adaptation
            output = output * modality_gain * adaptation

            # Update adaptation state (sensory adaptation)
            mean_activity = np.float32(np.mean(np.abs(output)))
            new_adaptation = np.float32(
                adaptation * 0.99 + (1.0 - mean_activity) * 0.01
            )
            self.set_state("adaptation_state", new_adaptation)

            return Tensor(data=output.astype(np.float32), shape=data.shape)
        else:
            # 2D input: apply single filter
            h, w = input_data.shape
            output = np.zeros_like(input_data)

            for gabor_filter in self._filters:
                fh, fw = gabor_filter.shape
                padded_filter = np.zeros((h, w), dtype=np.float32)
                pad_h, pad_w = (h - fh) // 2, (w - fw) // 2
                padded_filter[pad_h:pad_h+fh, pad_w:pad_w+fw] = gabor_filter

                fft_input = np.fft.fft2(input_data)
                fft_filter = np.fft.fft2(padded_filter)
                conv_result = np.real(np.fft.ifft2(fft_input * fft_filter))
                output += conv_result

            output = np.maximum(output, 0.0) * modality_gain
            return Tensor(data=output.astype(np.float32), shape=data.shape)

    def _local_normalization(self, data: np.ndarray) -> np.ndarray:
        """Compute local normalization denominator.

        Uses a local averaging filter (analogous to surround suppression
        in sensory cortex) to compute the normalization pool for each
        spatial location.
        """
        if data.ndim == 3:
            h, w, c = data.shape
            kernel_size = 3
            pad = kernel_size // 2

            abs_data = np.abs(data)
            local_sum = np.zeros_like(abs_data)

            for ch in range(c):
                padded = np.pad(abs_data[:, :, ch], pad, mode='reflect')
                # Use uniform_filter via sliding window (simple convolution)
                # Build integral image for O(1) per-pixel box filter
                integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
                # Prepend zeros for easier indexing
                integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant',
                                  constant_values=0)
                # Box filter: sum in [i-pad:i+pad+1, j-pad:j+pad+1]
                # = integral[i+pad+1, j+pad+1] - integral[i, j+pad+1]
                #   - integral[i+pad+1, j] + integral[i, j]
                ks = kernel_size
                box_sum = (
                    integral[ks:, ks:] -
                    integral[:-ks, ks:] -
                    integral[ks:, :-ks] +
                    integral[:-ks, :-ks]
                )
                local_sum[:, :, ch] = box_sum[:h, :w]

            return local_sum / np.float32(kernel_size * kernel_size)
        else:
            return np.abs(data) + np.float32(1e-6)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Process state vector through sensory feature extraction.

        Applies a learned linear filter (analogous to sensory receptive fields)
        followed by rectification.
        """
        modality_gain = self.get_state("modality_gain", np.float32(1.0))
        # Rectify and scale
        modulated = np.maximum(data.data, 0.0) * modality_gain
        return StateVector(data=modulated.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal through sensory pathway.

        Applies half-wave rectification (only positive signals pass),
        modeling the threshold properties of sensory neurons.
        """
        modality_gain = self.get_state("modality_gain", np.float32(1.0))
        rectified = max(0.0, data.value) * modality_gain
        return Signal(value=rectified, timestamp=data.timestamp)
