"""L03: ThalamicRelayStationNode - Sensory/motor signal relay with gating.

Neuroscience basis:
The thalamus is the central relay station for nearly all sensory and motor
signals traveling between the cortex and subcortical structures. Key features:
1. Relay function: Sensory signals (except olfaction) pass through thalamic
   nuclei before reaching cortex (Sherman & Guillery, 2001)
2. Attention-based gating: Thalamic reticular nucleus (TRN) provides
   inhibitory gating based on attentional demands
3. Burst/tonic mode switching: Thalamic relay neurons switch between
   burst mode (low-threshold Ca2+ spikes during sleep/inattention) and
   tonic mode (single spikes during wakeful attention) (Steriade & Llinas, 1988)
4. Topographic mapping: Spatial organization is preserved through thalamic relay
This node implements these core thalamic processing principles.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class ThalamicRelayStationNode(BrainNode):
    """Thalamic relay: sensory/motor signal relay with gating.

    Implements signal relay with attention-based gating, topographic mapping
    preservation, and burst/tonic mode switching.

    Processing logic:
    1. Attention-based gating (TRN inhibitory control)
    2. Burst/tonic mode switching based on arousal
    3. Topographic mapping preservation
    4. Signal relay with gain modulation
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Thalamic state
        self.set_state("relay_gain", np.float32(1.0))
        self.set_state("attention_gate", np.float32(1.0))
        self.set_state("mode", "tonic")  # "tonic" or "burst"
        self.set_state("arousal_level", np.float32(0.7))
        self.set_state("burst_threshold", np.float32(0.3))
        self.set_state("trn_inhibition", np.float32(0.0))
        # Previous activity for burst detection
        self.set_state("prev_activity", np.float32(0.0))

    def process(self, data: BrainData) -> BrainData:
        """Relay and gate incoming signals.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Gated and relayed output
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

    def _update_mode(self, input_salience: float) -> None:
        """Update burst/tonic mode based on arousal and input.

        In tonic mode (high arousal): faithful relay of signals
        In burst mode (low arousal): only strong/salient signals pass through,
        with enhanced gain for novel stimuli (Sherman, 2001).
        """
        arousal = self.get_state("arousal_level", np.float32(0.7))
        burst_threshold = self.get_state("burst_threshold", np.float32(0.3))

        if arousal > burst_threshold:
            self.set_state("mode", "tonic")
        else:
            self.set_state("mode", "burst")

    def _compute_gate(self, input_salience: float) -> np.float32:
        """Compute attention-based gating via TRN inhibition.

        The thalamic reticular nucleus (TRN) provides inhibitory gating:
        - High attention: TRN inhibition focused on irrelevant pathways
        - Low attention: diffuse TRN inhibition reduces all signals
        - Gate = 1 - TRN_inhibition (higher = more signal passes)
        """
        trn = self.get_state("trn_inhibition", np.float32(0.0))

        # TRN inhibition is modulated by attention and input salience
        # High salience reduces TRN inhibition for that pathway
        gate = np.float32(1.0 - trn + 0.3 * input_salience)
        gate = np.clip(gate, 0.0, 1.0)

        self.set_state("attention_gate", gate)
        return gate

    def _process_tensor(self, data: Tensor) -> Tensor:
        """Relay tensor with attention gating and mode-dependent processing.

        Tonic mode: faithful relay with attention gating
        Burst mode: enhanced response to novel/salient inputs
        """
        input_data = data.data
        relay_gain = self.get_state("relay_gain", np.float32(1.0))

        # Compute input salience
        salience = float(np.tanh(np.mean(np.abs(input_data))))

        # Update mode and gate
        self._update_mode(salience)
        gate = self._compute_gate(salience)

        mode = self.get_state("mode", "tonic")

        if mode == "tonic":
            # Faithful relay with attention gating
            output = input_data * relay_gain * gate
        else:
            # Burst mode: enhance salient inputs, suppress background
            prev_activity = self.get_state("prev_activity", np.float32(0.0))
            novelty = np.float32(max(0.0, salience - prev_activity))

            # Burst response: high gain for novel stimuli
            burst_gain = np.float32(1.0 + 2.0 * novelty)
            output = input_data * relay_gain * gate * burst_gain

        # Preserve topographic mapping (identity mapping through relay)
        self.set_state("prev_activity", np.float32(salience))

        return Tensor(data=output.astype(np.float32), shape=data.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Relay state vector with gating.

        Applies attention gate and mode-dependent processing.
        """
        salience = float(np.tanh(np.mean(np.abs(data.data))))
        self._update_mode(salience)
        gate = self._compute_gate(salience)
        relay_gain = self.get_state("relay_gain", np.float32(1.0))

        mode = self.get_state("mode", "tonic")
        if mode == "burst":
            prev_activity = self.get_state("prev_activity", np.float32(0.0))
            novelty = np.float32(max(0.0, salience - prev_activity))
            relay_gain = np.float32(relay_gain * (1.0 + 2.0 * novelty))

        output = data.data * relay_gain * gate
        self.set_state("prev_activity", np.float32(salience))

        return StateVector(data=output.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Relay scalar signal with gating.

        In burst mode, only signals above a threshold are relayed.
        """
        salience = float(np.tanh(abs(data.value)))
        self._update_mode(salience)
        gate = self._compute_gate(salience)
        relay_gain = self.get_state("relay_gain", np.float32(1.0))

        mode = self.get_state("mode", "tonic")
        if mode == "burst" and abs(data.value) < 0.1:
            # Burst mode: suppress weak signals
            return Signal(value=0.0, timestamp=data.timestamp)

        output = data.value * relay_gain * gate
        self.set_state("prev_activity", np.float32(salience))

        return Signal(value=float(output), timestamp=data.timestamp)
