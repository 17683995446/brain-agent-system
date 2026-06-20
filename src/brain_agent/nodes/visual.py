"""L03: VisualPathwayNode - Retinal to cortical visual processing.

Neuroscience basis:
The visual pathway processes information through multiple stages:
1. Retina: Center-surround receptive fields via ON/OFF bipolar cells
   (Kuffler, 1953). ON cells respond to light increments, OFF cells to decrements.
2. LGN: Thalamic relay with temporal filtering
3. V1: Orientation-selective neurons via Gabor-like receptive fields
   (Hubel & Wiesel, 1962), organized in cortical columns
4. Spatial frequency analysis via FFT-like decomposition in cortical circuits
This node implements the early visual processing pipeline using numpy FFT
for efficient spatial frequency filtering.
"""

from __future__ import annotations
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import Tensor, StateVector, Signal


class VisualPathwayNode(BrainNode):
    """Visual pathway: retinal to cortical processing.

    Implements center-surround receptive field computation, ON/OFF channel
    separation, and spatial frequency filtering using numpy FFT.

    Processing logic:
    1. Center-surround receptive field (DoG - Difference of Gaussians)
    2. ON/OFF channel separation (positive/negative rectification)
    3. Spatial frequency filtering (band-pass via FFT)
    4. Contrast normalization
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Receptive field parameters
        self.set_state("center_sigma", np.float32(1.5))
        self.set_state("surround_sigma", np.float32(3.0))
        self.set_state("contrast_sensitivity", np.float32(1.0))
        # Spatial frequency band parameters
        self.set_state("low_freq_cutoff", np.float32(0.05))
        self.set_state("high_freq_cutoff", np.float32(0.4))
        # Adaptation state
        self.set_state("light_adaptation", np.float32(1.0))

    def _process_tensor(self, data: Tensor) -> Tensor:
        """Full visual processing pipeline for tensor input.

        Pipeline:
        1. Center-surround receptive field (Difference of Gaussians)
        2. ON/OFF channel separation
        3. Spatial frequency band-pass filtering via FFT
        4. Contrast normalization
        """
        input_data = data.data
        center_sigma = self.get_state("center_sigma", np.float32(1.5))
        surround_sigma = self.get_state("surround_sigma", np.float32(3.0))
        contrast_sensitivity = self.get_state("contrast_sensitivity", np.float32(1.0))
        light_adapt = self.get_state("light_adaptation", np.float32(1.0))

        if input_data.ndim < 2:
            return Tensor(data=input_data, shape=data.shape)

        h, w = input_data.shape[:2]

        # Step 1: Center-surround receptive field (DoG)
        # Create Difference of Gaussians kernel
        y, x = np.mgrid[-h//2:h//2, -w//2:w//2].astype(np.float32)
        r_sq = x**2 + y**2

        center_gaussian = np.exp(-r_sq / (2 * center_sigma**2))
        surround_gaussian = np.exp(-r_sq / (2 * surround_sigma**2))

        # Normalize kernels
        center_gaussian /= (center_gaussian.sum() + 1e-8)
        surround_gaussian /= (surround_gaussian.sum() + 1e-8)

        # DoG: center - surround (excitatory center, inhibitory surround)
        dog_kernel = center_gaussian - 0.8 * surround_gaussian

        # Apply via FFT convolution
        if input_data.ndim == 3:
            output = np.zeros_like(input_data)
            for ch in range(input_data.shape[2]):
                fft_input = np.fft.fft2(input_data[:, :, ch])
                fft_kernel = np.fft.fft2(np.fft.ifftshift(dog_kernel))
                conv = np.real(np.fft.ifft2(fft_input * fft_kernel))
                output[:, :, ch] = conv
        else:
            fft_input = np.fft.fft2(input_data)
            fft_kernel = np.fft.fft2(np.fft.ifftshift(dog_kernel))
            output = np.real(np.fft.ifft2(fft_input * fft_kernel))

        # Step 2: ON/OFF channel separation
        on_channel = np.maximum(output, 0.0)  # ON: responds to increments
        off_channel = np.maximum(-output, 0.0)  # OFF: responds to decrements

        # Combine ON and OFF channels
        if input_data.ndim == 3:
            n_channels = input_data.shape[2]
            if n_channels >= 2:
                output[:, :, 0::2] = on_channel[:, :, 0::2]  # Even channels: ON
                if n_channels > 1:
                    output[:, :, 1::2] = off_channel[:, :, 1::2]  # Odd channels: OFF
            else:
                output = on_channel
        else:
            output = on_channel

        # Step 3: Spatial frequency band-pass filtering via FFT
        low_cutoff = self.get_state("low_freq_cutoff", np.float32(0.05))
        high_cutoff = self.get_state("high_freq_cutoff", np.float32(0.4))

        if output.ndim == 3:
            for ch in range(output.shape[2]):
                output[:, :, ch] = self._bandpass_filter(
                    output[:, :, ch], h, w, low_cutoff, high_cutoff
                )
        else:
            output = self._bandpass_filter(output, h, w, low_cutoff, high_cutoff)

        # Step 4: Contrast normalization
        mean_intensity = np.mean(np.abs(output))
        if mean_intensity > 1e-6:
            output = output / (mean_intensity + np.float32(1e-6)) * contrast_sensitivity

        # Apply light adaptation
        output = output * light_adapt

        # Update adaptation state
        new_adapt = np.float32(light_adapt * 0.99 + (1.0 / (mean_intensity + 1e-6)) * 0.01)
        new_adapt = np.clip(new_adapt, 0.1, 10.0)
        self.set_state("light_adaptation", new_adapt)

        return Tensor(data=output.astype(np.float32), shape=data.shape)

    def _bandpass_filter(self, data: np.ndarray, h: int, w: int,
                         low_cutoff: float, high_cutoff: float) -> np.ndarray:
        """Apply band-pass spatial frequency filter using FFT.

        Creates a radial frequency mask that passes frequencies between
        low_cutoff and high_cutoff (as fraction of Nyquist frequency).
        This models the spatial frequency tuning of V1 neurons.
        """
        # Compute frequency grid
        fy = np.fft.fftfreq(h).astype(np.float32)
        fx = np.fft.fftfreq(w).astype(np.float32)
        FX, FY = np.meshgrid(fx, fy)
        freq_radius = np.sqrt(FX**2 + FY**2)

        # Band-pass mask
        low_mask = (freq_radius >= low_cutoff).astype(np.float32)
        high_mask = (freq_radius <= high_cutoff).astype(np.float32)
        band_mask = low_mask * high_mask

        # Apply filter
        fft_data = np.fft.fft2(data)
        filtered = np.real(np.fft.ifft2(fft_data * band_mask))

        return filtered

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Process state vector through visual pathway.

        Applies center-surround-like operation: subtract local mean
        from each element to enhance contrast.
        """
        local_mean = np.mean(data.data)
        contrast_enhanced = data.data - local_mean
        return StateVector(data=contrast_enhanced.astype(np.float32),
                           dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal as luminance input.

        Applies light adaptation to the signal.
        """
        light_adapt = self.get_state("light_adaptation", np.float32(1.0))
        return Signal(value=float(data.value * light_adapt), timestamp=data.timestamp)
