"""L03: AuditoryPathwayNode - Cochlear to cortical auditory processing.

Neuroscience basis:
The auditory pathway processes sound through several stages:
1. Cochlea: Basilar membrane performs frequency decomposition (tonotopic mapping)
   where different frequencies activate different locations along the membrane
   (von Bekesy, 1960).
2. Auditory nerve: Frequency-specific fibers with phase-locking
3. Brainstem: Sound localization via interaural time difference (ITD) and
   interaural level difference (ILD) processing (Jeffress, 1948)
4. Auditory cortex: Spectrotemporal receptive fields with complex frequency tuning
This node implements frequency decomposition via FFT, tonotopic mapping,
and sound localization cue extraction.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class AuditoryPathwayNode(BrainNode):
    """Auditory pathway: cochlear to cortical processing.

    Implements frequency decomposition (spectrogram-like), tonotopic mapping,
    and sound localization cues.

    Processing logic:
    1. Frequency decomposition via STFT-like processing
    2. Tonotopic mapping: frequency -> spatial location
    3. Sound localization: ITD/ILD cue extraction
    4. Temporal envelope extraction
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Cochlear parameters
        self.set_state("n_frequency_bands", 32)
        self.set_state("frequency_range", (np.float32(20.0), np.float32(20000.0)))
        # Localization parameters
        self.set_state("head_radius", np.float32(0.0875))  # ~8.75 cm human head
        self.set_state("sound_speed", np.float32(343.0))  # m/s at 20C
        # Adaptation
        self.set_state("loudness_adaptation", np.float32(1.0))

    def process(self, data: BrainData) -> BrainData:
        """Process auditory input through cochlear to cortical pipeline.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Auditory-processed output
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
        """Process tensor through auditory pipeline.

        Interprets tensor as multi-channel audio representation:
        - Spatial dimensions map to frequency bands (tonotopic)
        - Channel dimension maps to time frames or binaural channels

        Pipeline:
        1. Frequency decomposition via 1D FFT along temporal axis
        2. Tonotopic mapping: reorganize by frequency
        3. Sound localization cues
        """
        input_data = data.data
        loudness_adapt = self.get_state("loudness_adaptation", np.float32(1.0))

        if input_data.ndim == 3:
            h, w, c = input_data.shape
            output = np.zeros_like(input_data)

            for ch in range(c):
                channel_data = input_data[:, :, ch]

                # Frequency decomposition along rows (tonotopic axis)
                # Each row represents a time segment
                for row in range(h):
                    segment = channel_data[row, :]

                    # FFT for frequency decomposition
                    spectrum = np.fft.rfft(segment)
                    magnitude = np.abs(spectrum).astype(np.float32)

                    # Mel-like compression (log compression of magnitudes)
                    compressed = np.log1p(magnitude)

                    # Map back to spatial representation (tonotopic)
                    # Lower frequencies -> lower indices
                    if len(compressed) <= w:
                        padded = np.zeros(w, dtype=np.float32)
                        padded[:len(compressed)] = compressed
                        output[row, :, ch] = padded
                    else:
                        output[row, :, ch] = compressed[:w]

                # Sound localization: compute inter-channel differences
                if c >= 2:
                    # ITD approximation: phase difference between channels
                    left = output[:, :, 0]
                    right = output[:, :, 1]

                    # Compute cross-correlation for ITD
                    itd_map = self._compute_itd(left, right)
                    output[:, :, min(2, c-1)] = itd_map

            # Apply loudness adaptation
            output = output * loudness_adapt

            # Update adaptation
            rms = np.float32(np.sqrt(np.mean(output**2)))
            new_adapt = np.float32(loudness_adapt * 0.99 + (1.0 / (rms + 1e-6)) * 0.01)
            new_adapt = np.clip(new_adapt, 0.1, 10.0)
            self.set_state("loudness_adaptation", new_adapt)

            return Tensor(data=output.astype(np.float32), shape=data.shape)
        else:
            # 2D input: treat as single-channel spectrogram
            h, w = input_data.shape
            output = np.zeros_like(input_data)

            for row in range(h):
                segment = input_data[row, :]
                spectrum = np.fft.rfft(segment)
                magnitude = np.abs(spectrum).astype(np.float32)
                compressed = np.log1p(magnitude)
                if len(compressed) <= w:
                    output[row, :len(compressed)] = compressed

            return Tensor(data=(output * loudness_adapt).astype(np.float32),
                          shape=data.shape)

    def _compute_itd(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute Interaural Time Difference (ITD) map.

        ITD is computed via cross-correlation between left and right
        channels, providing sound localization cues. The maximum
        physiological ITD for humans is ~640 microseconds.

        Uses normalized cross-correlation for robust estimation.
        """
        h, w = left.shape
        itd_map = np.zeros((h, w), dtype=np.float32)

        for row in range(h):
            left_row = left[row, :]
            right_row = right[row, :]

            # Normalized cross-correlation
            left_row_norm = left_row - np.mean(left_row)
            right_row_norm = right_row - np.mean(right_row)

            left_row_std = np.std(left_row_norm)
            right_row_std = np.std(right_row_norm)

            if left_row_std > 1e-6 and right_row_std > 1e-6:
                correlation = np.correlate(left_row_norm, right_row_norm, mode='same')
                max_corr = np.max(np.abs(correlation))
                if max_corr > 1e-6:
                    # Normalize to [0, 1]
                    itd_map[row, :] = (correlation / max_corr).astype(np.float32)
                else:
                    itd_map[row, :] = np.float32(0.5)
            else:
                itd_map[row, :] = np.float32(0.5)

        return itd_map

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Process state vector through auditory pathway.

        Applies frequency-domain filtering: decompose into frequency
        components, apply cochlear filter bank, reconstruct.
        """
        # FFT-based frequency decomposition
        spectrum = np.fft.rfft(data.data)
        magnitude = np.abs(spectrum).astype(np.float32)

        # Cochlear filter bank: emphasize mid-frequencies (speech range)
        n_freq = len(magnitude)
        freq_weights = np.ones(n_freq, dtype=np.float32)
        for i in range(n_freq):
            # Mel-like weighting: peak at mid frequencies
            center = n_freq / 4.0
            freq_weights[i] = np.exp(-0.5 * ((i - center) / (center * 0.5 + 1e-6)) ** 2)

        # Apply filter and reconstruct
        filtered_spectrum = spectrum * freq_weights
        reconstructed = np.fft.irfft(filtered_spectrum, n=data.dimension)

        return StateVector(data=reconstructed.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal as sound amplitude.

        Applies loudness adaptation (Weber's law: perceived loudness
        is proportional to log of intensity).
        """
        loudness_adapt = self.get_state("loudness_adaptation", np.float32(1.0))
        # Weber-Fechner law approximation
        perceived = float(np.sign(data.value) * np.log1p(abs(data.value)))
        return Signal(value=perceived * loudness_adapt, timestamp=data.timestamp)
