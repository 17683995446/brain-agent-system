"""L03: AmygdalaCircuitNode - Emotion processing.

Neuroscience basis:
The amygdala is the central hub for emotional processing, particularly
fear conditioning and valence evaluation:
1. Basolateral amygdala (BLA): Sensory input integration and valence
   assignment (positive/negative evaluation) (Paton et al., 2006)
2. Central nucleus (CeA): Output nucleus that generates emotional responses
   via projections to hypothalamus, brainstem, and PFC
3. Fear conditioning: CS-US association learning via Hebbian plasticity
   in amygdala circuits (LeDoux, 2000)
4. Emotional memory modulation: Amygdala modulates hippocampal memory
   consolidation via norepinephrine release (McGaugh, 2004)
This node implements valence evaluation, fear conditioning, and emotional
memory modulation.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class AmygdalaCircuitNode(BrainNode):
    """Amygdala circuit: emotion processing.

    Implements valence evaluation (positive/negative), fear conditioning
    (CS-US association), and emotional memory modulation.

    Processing logic:
    1. Valence evaluation: compute positive/negative valence from input
    2. Fear conditioning: associate conditioned stimulus with unconditioned
    3. Emotional memory modulation: modulate consolidation strength
    4. Generate emotional response output
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Valence state
        self.set_state("current_valence", np.float32(0.0))  # -1 (negative) to +1 (positive)
        self.set_state("arousal", np.float32(0.0))  # 0 (calm) to 1 (excited)
        # Fear conditioning
        self.set_state("cs_us_associations", {})  # CS pattern -> US valence
        self.set_state("conditioning_strength", np.float32(0.5))
        self.set_state("fear_threshold", np.float32(0.6))
        # Emotional memory modulation
        self.set_state("memory_modulation", np.float32(1.0))
        self.set_state("emotional_salience", np.float32(0.0))
        # Extinction
        self.set_state("extinction_rate", np.float32(0.01))

    def process(self, data: BrainData) -> BrainData:
        """Process input through amygdala emotion circuit.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Emotion-processed output
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

    def _evaluate_valence(self, input_data: np.ndarray) -> tuple:
        """Evaluate emotional valence from input.

        Valence is computed based on:
        1. Overall activation level (arousal)
        2. Distribution skewness (positive skew = positive valence)
        3. Pattern matching against stored CS-US associations

        Returns:
            (valence, arousal) tuple where valence in [-1, 1], arousal in [0, 1]
        """
        # Arousal: overall activation magnitude
        arousal = float(np.tanh(np.mean(np.abs(input_data))))

        # Valence from distribution statistics
        mean_val = float(np.mean(input_data))
        std_val = float(np.std(input_data)) + 1e-6

        # Positive skew -> positive valence, negative skew -> negative valence
        if len(input_data) > 2:
            skewness = float(np.mean(((input_data - mean_val) / std_val) ** 3))
            valence = np.tanh(skewness * 0.5 + mean_val * 0.5)
        else:
            valence = np.tanh(mean_val)

        # Check CS-US associations (fear conditioning)
        cs_us = self.get_state("cs_us_associations", {})
        if len(cs_us) > 0:
            for cs_pattern, us_valence in cs_us.items():
                # Simple similarity check
                cs_vec = np.frombuffer(bytes.fromhex(cs_pattern), dtype=np.float32)
                if len(cs_vec) == len(input_data):
                    sim = np.dot(input_data, cs_vec) / (
                        np.linalg.norm(input_data) * np.linalg.norm(cs_vec) + 1e-6
                    )
                    if sim > 0.7:
                        # Conditioned response: shift valence toward US valence
                        conditioning = self.get_state("conditioning_strength", np.float32(0.5))
                        valence = valence * (1 - conditioning) + us_valence * conditioning

        return float(valence), float(arousal)

    def _process_tensor(self, data: Tensor) -> Tensor:
        """Process tensor through amygdala emotion circuit.

        Evaluates valence, applies emotional modulation, and generates
        emotionally colored output.
        """
        input_data = data.data
        flat = input_data.reshape(-1)

        # Evaluate valence and arousal
        valence, arousal = self._evaluate_valence(flat)

        # Update emotional state
        self.set_state("current_valence", np.float32(valence))
        self.set_state("arousal", np.float32(arousal))

        # Emotional modulation: valence-dependent gain
        # Positive valence enhances positive features, negative enhances negative
        positive_mask = (input_data > 0).astype(np.float32)
        negative_mask = (input_data < 0).astype(np.float32)

        # Valence-weighted modulation
        if valence > 0:
            output = input_data * (1.0 + 0.3 * valence * positive_mask -
                                   0.2 * valence * negative_mask)
        else:
            output = input_data * (1.0 - 0.2 * abs(valence) * positive_mask +
                                   0.3 * abs(valence) * negative_mask)

        # Arousal modulation: high arousal amplifies everything
        output = output * (1.0 + 0.5 * arousal)

        # Update emotional memory modulation
        emotional_salience = np.float32(abs(valence) * arousal)
        self.set_state("emotional_salience", emotional_salience)
        memory_mod = np.float32(1.0 + 0.5 * emotional_salience)
        self.set_state("memory_modulation", memory_mod)

        return Tensor(data=output.astype(np.float32), shape=data.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Process state vector through amygdala circuit.

        Full emotional evaluation pipeline for state vectors.
        """
        valence, arousal = self._evaluate_valence(data.data)

        self.set_state("current_valence", np.float32(valence))
        self.set_state("arousal", np.float32(arousal))

        # Emotional coloring of the state vector
        # Positive valence: boost positive components
        # Negative valence: boost negative components
        positive = np.maximum(data.data, 0.0)
        negative = np.minimum(data.data, 0.0)

        if valence > 0:
            output = positive * (1.0 + 0.3 * valence) + negative * (1.0 - 0.2 * valence)
        else:
            output = positive * (1.0 + 0.2 * valence) + negative * (1.0 - 0.3 * abs(valence))

        output = output * (1.0 + 0.5 * arousal)

        # Update memory modulation
        emotional_salience = np.float32(abs(valence) * arousal)
        self.set_state("emotional_salience", emotional_salience)
        self.set_state("memory_modulation", np.float32(1.0 + 0.5 * emotional_salience))

        return StateVector(data=output.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal through amygdala valence system.

        If signal represents a US (unconditioned stimulus), update
        fear conditioning associations.
        """
        self.get_state("current_valence", np.float32(0.0))
        arousal = self.get_state("arousal", np.float32(0.0))

        # Emotional amplification
        emotional_gain = 1.0 + 0.5 * arousal
        output = data.value * emotional_gain

        # If strong negative signal, this could be a US for fear conditioning
        if data.value < -0.5:
            conditioning = self.get_state("conditioning_strength", np.float32(0.5))
            new_conditioning = np.float32(min(1.0, conditioning + 0.1))
            self.set_state("conditioning_strength", new_conditioning)

        return Signal(value=float(output), timestamp=data.timestamp)
