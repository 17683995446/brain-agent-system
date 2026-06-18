"""L03: SomatosensoryPathwayNode - Touch and proprioception processing.

Neuroscience basis:
The somatosensory system processes touch, proprioception, temperature, and
nociception (pain) through distinct pathways:
1. Dorsal column-medial lemniscal pathway: Fine touch and proprioception
   with precise topographic organization (Penfield's sensory homunculus)
2. Spinothalamic pathway: Temperature and nociception
3. Primary somatosensory cortex (S1): Contains distinct areas for different
   submodalities (3a: proprioception, 3b: touch, 1/2: integration)
4. Somatotopic mapping: Adjacent body parts are represented in adjacent
   cortical regions (Penfield & Rasmussen, 1950)
This node implements tactile feature extraction, proprioceptive integration,
and somatotopic mapping.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class SomatosensoryPathwayNode(BrainNode):
    """Somatosensory pathway: touch and proprioception.

    Implements tactile feature extraction, proprioceptive signal integration,
    and somatotopic mapping.

    Processing logic:
    1. Tactile feature extraction (pressure, vibration, texture)
    2. Proprioceptive signal integration (position, velocity)
    3. Somatotopic mapping (body-part specific processing)
    4. Lateral inhibition for edge enhancement
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Receptive field parameters
        self.set_state("mechanoreceptor_gain", np.float32(1.0))
        self.set_state("proprioceptive_gain", np.float32(1.0))
        self.set_state("nociceptor_threshold", np.float32(0.8))
        # Somatotopic map: body part regions
        self.set_state("body_map", self._create_body_map())
        # Adaptation
        self.set_state("tactile_adaptation", np.float32(1.0))

    def _create_body_map(self) -> np.ndarray:
        """Create a somatotopic body map.

        The map assigns different spatial regions to different body parts,
        following Penfield's sensory homunculus where hand and face areas
        are disproportionately large relative to their physical size.
        """
        body_map = np.zeros((32, 32), dtype=np.float32)

        # Hand region (large representation - high tactile acuity)
        body_map[8:20, 8:20] = 1.0
        # Face region (large representation)
        body_map[2:8, 12:24] = 0.9
        # Arm region
        body_map[12:24, 20:28] = 0.6
        # Trunk region
        body_map[20:28, 4:16] = 0.4
        # Leg region
        body_map[24:32, 16:28] = 0.3

        # Smooth the map
        from scipy.ndimage import gaussian_filter
        try:
            body_map = gaussian_filter(body_map, sigma=1.0).astype(np.float32)
        except ImportError:
            # Simple box filter if scipy not available
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            for _ in range(2):
                padded = np.pad(body_map, 1, mode='reflect')
                body_map = np.zeros_like(body_map)
                for i in range(32):
                    for j in range(32):
                        body_map[i, j] = np.sum(
                            padded[i:i+3, j:j+3] * kernel
                        )

        return body_map

    def process(self, data: BrainData) -> BrainData:
        """Process somatosensory input.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Somatosensory-processed output
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
        """Process tensor through somatosensory pipeline.

        Pipeline:
        1. Somatotopic mapping: weight by body region
        2. Tactile feature extraction: edge detection (Meissner corpuscles)
        3. Vibration detection: temporal derivative (Pacinian corpuscles)
        4. Lateral inhibition for spatial sharpening
        """
        input_data = data.data
        mech_gain = self.get_state("mechanoreceptor_gain", np.float32(1.0))
        self.get_state("proprioceptive_gain", np.float32(1.0))
        body_map = self.get_state("body_map", np.ones((32, 32), dtype=np.float32))
        tactile_adapt = self.get_state("tactile_adaptation", np.float32(1.0))

        if input_data.ndim < 2:
            return Tensor(data=input_data * mech_gain, shape=data.shape)

        h, w = input_data.shape[:2]

        # Step 1: Somatotopic mapping
        if h == 32 and w == 32:
            if input_data.ndim == 3:
                mapped = input_data * body_map[:, :, np.newaxis]
            else:
                mapped = input_data * body_map
        else:
            mapped = input_data

        # Step 2: Tactile feature extraction - edge detection
        # Laplacian filter for edge detection (models Meissner corpuscle response)
        if mapped.ndim == 3:
            edges = np.zeros_like(mapped)
            for ch in range(mapped.shape[2]):
                edges[:, :, ch] = self._laplacian(mapped[:, :, ch])
        else:
            edges = self._laplacian(mapped)

        # Step 3: Vibration detection (temporal change)
        # Use spatial gradient magnitude as proxy for vibration
        if mapped.ndim == 3:
            vibration = np.zeros_like(mapped)
            for ch in range(mapped.shape[2]):
                vibration[:, :, ch] = self._gradient_magnitude(mapped[:, :, ch])
        else:
            vibration = self._gradient_magnitude(mapped)

        # Combine: static pressure + edges + vibration
        if mapped.ndim == 3:
            output = mapped * 0.5 + edges * 0.3 + vibration * 0.2
        else:
            output = mapped * 0.5 + edges * 0.3 + vibration * 0.2

        # Step 4: Lateral inhibition
        if output.ndim == 3:
            for ch in range(output.shape[2]):
                output[:, :, ch] = self._lateral_inhibition(output[:, :, ch])
        else:
            output = self._lateral_inhibition(output)

        # Apply gains and adaptation
        output = output * mech_gain * tactile_adapt

        # Update tactile adaptation (rapid adaptation like Pacinian corpuscles)
        activity = np.float32(np.mean(np.abs(output)))
        new_adapt = np.float32(tactile_adapt * 0.95 + (1.0 / (activity + 1e-6)) * 0.05)
        new_adapt = np.clip(new_adapt, 0.1, 10.0)
        self.set_state("tactile_adaptation", new_adapt)

        return Tensor(data=output.astype(np.float32), shape=data.shape)

    def _laplacian(self, data: np.ndarray) -> np.ndarray:
        """Compute Laplacian (edge detection) for tactile feature extraction.

        The Laplacian is a discrete second derivative that detects edges
        and fine details, modeling the response of Meissner corpuscles
        to changes in skin deformation.
        """
        padded = np.pad(data, 1, mode='reflect')
        laplacian = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, :-2] + padded[1:-1, 2:] -
            4 * padded[1:-1, 1:-1]
        )
        return np.maximum(laplacian, 0.0).astype(np.float32)

    def _gradient_magnitude(self, data: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude for vibration detection.

        Pacinian corpuscles respond to rapid changes in pressure (vibration).
        The spatial gradient magnitude serves as a proxy for vibration detection.
        """
        gy, gx = np.gradient(data.astype(np.float64))
        magnitude = np.sqrt(gx**2 + gy**2)
        return magnitude.astype(np.float32)

    def _lateral_inhibition(self, data: np.ndarray) -> np.ndarray:
        """Apply lateral inhibition for spatial sharpening.

        Lateral inhibition enhances contrast between adjacent stimuli,
        improving spatial resolution of tactile perception. This is
        implemented as subtractive surround (Mexican hat - DoG).
        """
        h, w = data.shape
        # Center: 3x3 averaging
        padded = np.pad(data, 1, mode='reflect')
        center = np.zeros_like(data)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                center += padded[1+di:h+1+di, 1+dj:w+1+dj]
        center /= 9.0

        # Surround: 5x5 averaging (excluding center 3x3)
        padded5 = np.pad(data, 2, mode='reflect')
        surround = np.zeros_like(data)
        count = 0
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if abs(di) > 1 or abs(dj) > 1:
                    surround += padded5[2+di:h+2+di, 2+dj:w+2+dj]
                    count += 1
        surround /= count

        # Lateral inhibition: center - surround
        result = data + (center - surround) * 0.5
        return np.maximum(result, 0.0).astype(np.float32)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Process state vector as proprioceptive input.

        Applies gain modulation and extracts velocity component
        (derivative approximation).
        """
        proprio_gain = self.get_state("proprioceptive_gain", np.float32(1.0))
        # Proprioceptive processing: position + velocity
        modulated = data.data * proprio_gain
        return StateVector(data=modulated.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal as pressure/force input.

        Applies mechanoreceptor gain and threshold detection.
        """
        mech_gain = self.get_state("mechanoreceptor_gain", np.float32(1.0))
        nociceptor_threshold = self.get_state("nociceptor_threshold", np.float32(0.8))

        # Mechanoreceptor response
        output = data.value * mech_gain

        # Nociceptor: only activate above threshold
        if abs(output) > nociceptor_threshold:
            output = np.sign(output) * (abs(output) - nociceptor_threshold)

        return Signal(value=float(output), timestamp=data.timestamp)
