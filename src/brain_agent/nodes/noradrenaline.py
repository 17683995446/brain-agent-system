"""L03: NoradrenalineSystemNode - Arousal and attention modulation.

Neuroscience basis:
The noradrenaline (norepinephrine) system originates from the locus coeruleus (LC)
and modulates arousal, attention, and signal-to-noise ratio across the brain:
1. Arousal regulation: LC tonic firing maintains baseline arousal and
   sleep-wake transitions (Aston-Jones & Bloom, 1981)
2. Attention gain modulation: Phasic LC responses to task-relevant stimuli
   enhance signal-to-noise ratio in target cortical areas (Aston-Jones & Cohen, 2005)
3. Phasic vs tonic mode: Tonic mode (high baseline, low phasic) occurs during
   disengaged exploration; phasic mode (low baseline, high phasic responses)
   occurs during focused task performance
4. Gain modulation: NE increases gain of sensory responses, improving
   information processing but also increasing metabolic cost
   (Berridge & Waterhouse, 2003)
This node implements arousal computation, attention gain modulation, and
phasic/tonic release mode switching.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class NoradrenalineSystemNode(BrainNode):
    """Noradrenaline system: arousal and attention.

    Implements arousal level computation, attention gain modulation,
    and phasic/tonic release mode switching.

    Processing logic:
    1. Compute arousal level from input salience and history
    2. Determine phasic vs tonic release mode
    3. Apply gain modulation to target regions
    4. Adjust signal-to-noise ratio
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Arousal state
        self.set_state("arousal_level", np.float32(0.5))
        self.set_state("arousal_history", np.zeros(10, dtype=np.float32))
        # Release mode
        self.set_state("mode", "phasic")  # "phasic" or "tonic"
        self.set_state("tonic_baseline", np.float32(0.3))
        self.set_state("phasic_threshold", np.float32(0.6))
        # Gain parameters
        self.set_state("gain_multiplier", np.float32(1.0))
        self.set_state("noise_level", np.float32(0.1))
        # LC dynamics
        self.set_state("lc_firing_rate", np.float32(2.0))  # Hz baseline
        self.set_state("phasic_response", np.float32(0.0))
        self.set_state("decay_rate", np.float32(0.2))
        # Utility signals
        self.set_state("expected_utility", np.float32(0.5))
        self.set_state("outcome_history", [])

    def process(self, data: BrainData) -> BrainData:
        """Compute arousal and apply attention gain modulation.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Noradrenaline-modulated output
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

    def _compute_arousal(self, salience: float) -> np.float32:
        """Compute arousal level from input salience.

        Arousal follows LC dynamics:
        - Tonic firing: maintains baseline arousal
        - Phasic response: transient arousal boost for salient stimuli
        - Arousal integrates salience with exponential decay

        The LC-NE system implements an adaptive gain theory (Aston-Jones & Cohen, 2005)
        where the mode (phasic vs tonic) optimizes performance-exploitation tradeoff.
        """
        current_arousal = self.get_state("arousal_level", np.float32(0.5))
        decay = self.get_state("decay_rate", np.float32(0.2))
        tonic = self.get_state("tonic_baseline", np.float32(0.3))

        # Phasic response to salient input
        phasic = np.float32(salience * 0.5)
        self.set_state("phasic_response", phasic)

        # Arousal update: decay toward tonic + phasic boost
        new_arousal = np.float32(
            current_arousal * (1.0 - decay) + (tonic + phasic) * decay
        )
        new_arousal = np.clip(new_arousal, 0.0, 1.0)

        # Update arousal history
        history = self.get_state("arousal_history", np.zeros(10, dtype=np.float32))
        history = np.roll(history, -1)
        history[-1] = new_arousal
        self.set_state("arousal_history", history)

        self.set_state("arousal_level", new_arousal)

        # Update LC firing rate (arousal-dependent)
        lc_rate = np.float32(1.0 + 5.0 * new_arousal)  # 1-6 Hz range
        self.set_state("lc_firing_rate", lc_rate)

        return new_arousal

    def _determine_mode(self, arousal: np.float32, salience: float) -> None:
        """Determine phasic vs tonic release mode.

        Phasic mode: Focused task performance
        - Moderate arousal, high task utility
        - Phasic responses to task-relevant stimuli
        - Optimizes signal-to-noise for current task

        Tonic mode: Exploratory/disengaged
        - High or very low arousal
        - Reduced phasic responses, elevated tonic firing
        - Facilitates scanning for alternative opportunities

        Based on Aston-Jones & Cohen (2005) adaptive gain theory.
        """
        expected_utility = self.get_state("expected_utility", np.float32(0.5))

        # Mode determination based on arousal and utility
        if 0.3 < arousal < 0.7 and expected_utility > 0.4:
            self.set_state("mode", "phasic")
        else:
            self.set_state("mode", "tonic")

    def _compute_gain(self, arousal: np.float32) -> np.float32:
        """Compute attention gain from arousal level.

        NE gain follows an inverted-U relationship with arousal:
        - Low arousal: low gain (poor signal processing)
        - Moderate arousal: optimal gain (best performance)
        - High arousal: excessive gain (noisy, distractible)

        gain(arousal) = 1 + k * arousal * (1 - arousal)
        This parabolic function peaks at arousal = 0.5
        """
        k = np.float32(2.0)  # gain coefficient
        gain = np.float32(1.0 + k * arousal * (1.0 - arousal))
        self.set_state("gain_multiplier", gain)
        return gain

    def _process_tensor(self, data: Tensor) -> Tensor:
        """Apply noradrenaline gain modulation to tensor input.

        NE increases gain of sensory responses, improving information
        processing. The gain depends on arousal level and release mode.
        """
        salience = float(np.tanh(np.mean(np.abs(data.data))))
        arousal = self._compute_arousal(salience)
        self._determine_mode(arousal, salience)
        gain = self._compute_gain(arousal)

        mode = self.get_state("mode", "phasic")

        if mode == "phasic":
            # Phasic mode: focused gain enhancement
            output = data.data * gain
        else:
            # Tonic mode: diffuse, slightly lower gain
            tonic_gain = np.float32(gain * 0.8)
            output = data.data * tonic_gain

        return Tensor(data=output.astype(np.float32), shape=data.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Apply noradrenaline modulation to state vector.

        Enhances signal-to-noise ratio by amplifying the state vector
        proportionally to arousal-dependent gain.
        """
        salience = float(np.tanh(np.mean(np.abs(data.data))))
        arousal = self._compute_arousal(salience)
        self._determine_mode(arousal, salience)
        gain = self._compute_gain(arousal)

        # SNR enhancement: amplify signal, add minimal noise
        noise = self.get_state("noise_level", np.float32(0.1))
        noise_component = np.random.randn(data.dimension).astype(np.float32) * noise * (1.0 - arousal)

        output = data.data * gain + noise_component
        return StateVector(data=output.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal through noradrenaline system.

        Returns the current NE level (tonic + phasic) as the signal.
        """
        salience = float(np.tanh(abs(data.value)))
        arousal = self._compute_arousal(salience)
        self._determine_mode(arousal, salience)

        # NE signal = tonic baseline + phasic response
        tonic = self.get_state("tonic_baseline", np.float32(0.3))
        phasic = self.get_state("phasic_response", np.float32(0.0))
        ne_level = float(tonic + phasic)

        return Signal(value=ne_level, timestamp=data.timestamp)
