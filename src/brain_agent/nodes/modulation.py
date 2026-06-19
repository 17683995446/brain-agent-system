"""L02: ModulationSystemNode - Neuromodulator release and gain modulation.

Neuroscience basis:
Neuromodulatory systems (dopamine, serotonin, acetylcholine, norepinephrine)
broadcast their signals widely across the brain, modulating the gain of
neural processing rather than carrying specific information (Aston-Jones &
Cohen, 2005). This node computes neuromodulator release levels based on
input salience and applies multiplicative gain modulation to target regions.
The gain modulation follows an inverted-U relationship (Arnsten, 2009)
where optimal performance occurs at intermediate levels of neuromodulation.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class ModulationSystemNode(BrainNode):
    """Modulation system: neuromodulator release and gain modulation.

    Computes neuromodulator release levels and applies gain modulation
    to target regions. Operates on Signal[float32] for scalar modulation
    signals.

    Processing logic:
    1. Compute release level based on input salience
    2. Apply inverted-U gain function
    3. Modulate target region activity
    4. Maintain baseline tonic release
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Neuromodulator state
        self.set_state("baseline_level", np.float32(0.3))
        self.set_state("current_level", np.float32(0.3))
        self.set_state("peak_gain", np.float32(1.0))
        self.set_state("decay_rate", np.float32(0.1))
        self.set_state("optimal_level", np.float32(0.5))
        self.set_state("gain_width", np.float32(0.3))

    def process(self, data: BrainData) -> BrainData:
        """Compute neuromodulator release and apply gain modulation.

        Signal[float32] operations.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Modulated output
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

    def _compute_gain(self) -> np.float32:
        """Compute inverted-U gain function.

        The inverted-U relationship (Arnsten, 2009) models how
        neuromodulator level affects neural gain:
        gain(x) = peak * exp(-(x - optimal)^2 / (2 * width^2))

        At optimal levels, gain is maximal. Too little or too much
        modulation reduces processing efficiency.
        """
        current = self.get_state("current_level", np.float32(0.3))
        peak = self.get_state("peak_gain", np.float32(1.0))
        optimal = self.get_state("optimal_level", np.float32(0.5))
        width = self.get_state("gain_width", np.float32(0.3))

        gain = peak * np.exp(-((current - optimal) ** 2) / (2 * width ** 2))
        return np.float32(gain)

    def _update_level(self, salience: float) -> None:
        """Update neuromodulator release level.

        Combines tonic (baseline) and phasic (salience-driven) components:
        level(t+1) = baseline + phasic_response * decay
        """
        baseline = self.get_state("baseline_level", np.float32(0.3))
        current = self.get_state("current_level", np.float32(0.3))
        decay = self.get_state("decay_rate", np.float32(0.1))

        # Phasic response: rapid increase driven by salience
        phasic = np.float32(salience * 0.5)

        # Exponential decay toward baseline
        new_level = current * (1.0 - decay) + (baseline + phasic) * decay
        new_level = np.float32(np.clip(new_level, 0.0, 1.0))
        self.set_state("current_level", new_level)

    def _process_tensor(self, data: Tensor) -> Tensor:
        """Apply neuromodulatory gain to tensor input.

        The gain is applied uniformly across all elements, modeling
        the diffuse projection patterns of neuromodulatory nuclei
        (e.g., locus coeruleus norepinephrine, ventral tegmental area dopamine).
        """
        # Compute salience from input
        salience = float(np.tanh(np.mean(np.abs(data.data))))
        self._update_level(salience)
        gain = self._compute_gain()

        modulated = data.data * gain
        return Tensor(data=modulated.astype(np.float32), shape=data.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Apply neuromodulatory gain to state vector.

        Gain modulation scales the entire state vector uniformly,
        affecting the signal-to-noise ratio of neural representations.
        """
        salience = float(np.tanh(np.mean(np.abs(data.data))))
        self._update_level(salience)
        gain = self._compute_gain()

        modulated = data.data * gain
        return StateVector(data=modulated.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal through neuromodulatory system.

        The signal modulates the release level, and the output
        reflects the current gain state.
        """
        salience = float(np.tanh(abs(data.value)))
        self._update_level(salience)
        gain = self._compute_gain()

        return Signal(value=float(data.value * gain), timestamp=data.timestamp)
