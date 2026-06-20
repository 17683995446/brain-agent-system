"""L00: BrainSystemNode - Root node for global integration and coordination.

Neuroscience basis:
The brain as a whole maintains global states such as arousal level and attentional
focus. This node simulates the global workspace theory (Baars, 1988) where information
from all subsystems is integrated into a unified conscious workspace. The global
integration uses weighted summation of inputs, with weights modulated by current
arousal and attention states. The node maintains persistent global state across
processing cycles, analogous to the brain's homeostatic mechanisms.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class BrainSystemNode(BrainNode):
    """Root node: Global integration and coordination.

    Implements the global workspace model where all subsystem outputs converge
    into a unified representation. Maintains global arousal and attention state
    that modulates processing across all subsystems.

    Processing logic:
    1. Update global arousal based on input magnitude (homeostatic regulation)
    2. Compute attention focus via softmax over input channels
    3. Apply weighted integration: output = arousal * attention * input
    4. Route integrated signal to appropriate subsystems
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Global brain state
        self.set_state("arousal", np.float32(0.5))
        self.set_state("attention_focus", np.zeros(128, dtype=np.float32))
        self.set_state("global_context", np.zeros(128, dtype=np.float32))
        # Arousal dynamics: time constant for exponential decay
        self.set_state("arousal_tau", np.float32(0.1))

    def process(self, data: BrainData) -> BrainData:
        """Global integration: weighted sum of all inputs with arousal modulation.

        Maintains global state (arousal level, attention focus) and routes
        data to appropriate subsystems.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Processed output preserving input type with global modulation applied
        """
        start = time.perf_counter()
        arousal = self._update_arousal(data)
        self.set_state("arousal", arousal)

        if isinstance(data, Tensor):
            result: BrainData = self._apply_tensor_modulation(data, arousal)
        elif isinstance(data, StateVector):
            result = self._apply_state_vector_modulation(data, arousal)
        elif isinstance(data, Signal):
            result = self._apply_signal_modulation(data, arousal)
        elif isinstance(data, Message):
            result = data  # Pass through control messages unchanged
        else:
            raise ValueError(
                f"Unsupported data type: {type(data).__name__}. "
                f"Supported types: Tensor, StateVector, Signal, Message."
            )

        self._update_metrics(time.perf_counter() - start)
        return result

    def _update_arousal(self, data: BrainData) -> np.float32:
        """Update global arousal level based on input salience.

        Arousal follows exponential dynamics:
        arousal(t+1) = arousal(t) * exp(-dt/tau) + salience * (1 - exp(-dt/tau))

        This models the LC-NE system's tonic firing that maintains baseline
        arousal with phasic responses to salient stimuli.
        """
        current_arousal = self.get_state("arousal", np.float32(0.5))
        tau = self.get_state("arousal_tau", np.float32(0.1))

        # Compute input salience
        if isinstance(data, Tensor):
            salience = np.float32(np.tanh(np.mean(np.abs(data.data))))
        elif isinstance(data, StateVector):
            salience = np.float32(np.tanh(np.mean(np.abs(data.data))))
        elif isinstance(data, Signal):
            salience = np.float32(np.tanh(abs(data.value)))
        else:
            salience = np.float32(0.0)

        # Exponential integration toward salience
        alpha = np.float32(1.0 - np.exp(-1.0 / tau))
        new_arousal = current_arousal * (1.0 - alpha) + salience * alpha
        return np.float32(np.clip(new_arousal, 0.0, 1.0))

    def _apply_tensor_modulation(self, data: Tensor, arousal: np.float32) -> Tensor:
        """Apply global modulation to tensor input.

        Uses arousal as a multiplicative gain factor applied uniformly
        across all spatial and feature dimensions, simulating the
        diffuse neuromodulatory effects of the LC-NE system.
        """
        # Arousal gain modulation
        modulated = data.data * arousal

        # Update global context via running average
        context = self.get_state("global_context", np.zeros(128, dtype=np.float32))
        flat = modulated.reshape(-1)
        # Pad or truncate to 128 dimensions
        if len(flat) >= 128:
            context_update = flat[:128]
        else:
            context_update = np.zeros(128, dtype=np.float32)
            context_update[:len(flat)] = flat
        new_context = context * 0.9 + context_update * 0.1
        self.set_state("global_context", new_context)

        return Tensor(data=modulated, shape=data.shape)

    def _apply_state_vector_modulation(self, data: StateVector, arousal: np.float32) -> StateVector:
        """Apply global modulation to state vector input.

        Computes attention-weighted integration between current global
        context and the input state vector.
        """
        context = self.get_state("global_context", np.zeros(128, dtype=np.float32))
        dim = min(data.dimension, 128)

        # Attention: softmax similarity between input and context
        similarity = np.dot(data.data[:dim], context[:dim])
        attention_weight = np.float32(1.0 / (1.0 + np.exp(-similarity)))

        # Integrate: blend input with context based on attention
        blended = (data.data * (1.0 - attention_weight * 0.3) +
                   context[:data.dimension] * (attention_weight * 0.3))
        modulated = blended * arousal

        return StateVector(data=modulated.astype(np.float32), dimension=data.dimension)

    def _apply_signal_modulation(self, data: Signal, arousal: np.float32) -> Signal:
        """Apply global modulation to scalar signal.

        Arousal gates the signal amplitude, simulating how global brain
        state modulates the impact of individual neuromodulator signals.
        """
        modulated_value = float(data.value * arousal)
        return Signal(value=modulated_value, timestamp=data.timestamp)
