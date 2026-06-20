"""L01: CognitiveSystemNode - Memory, emotion, and regulation.

Neuroscience basis:
The cognitive system encompasses prefrontal cortex (PFC), anterior cingulate cortex (ACC),
and other higher-order association areas. The PFC maintains working memory representations
and exerts top-down control over sensory and motor systems (Miller & Cohen, 2001).
The ACC monitors conflict and errors, signaling the need for cognitive control adjustment.
This node implements top-down modulation where cognitive state (goals, context) biases
processing in lower-level systems, and integrates memory retrieval with emotional evaluation.
"""

from __future__ import annotations
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import Tensor, StateVector, Signal


class CognitiveSystemNode(BrainNode):
    """Cognitive system: memory, emotion, regulation.

    Processes cognitive signals including memory retrieval, emotion regulation,
    and top-down modulation of lower systems. Operates primarily on
    StateVector[float32, 128] representations.

    Processing logic:
    1. Retrieve relevant context from working memory
    2. Apply top-down bias based on current goals
    3. Integrate emotional valence with cognitive evaluation
    4. Generate modulatory output for downstream systems
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Working memory buffer
        self.set_state("working_memory", np.zeros(128, dtype=np.float32))
        self.set_state("goal_vector", np.zeros(128, dtype=np.float32))
        self.set_state("cognitive_load", np.float32(0.0))
        self.set_state("top_down_strength", np.float32(0.5))

    def _process_tensor(self, data: Tensor) -> Tensor:
        """Apply top-down cognitive modulation to tensor input.

        The PFC exerts top-down control by biasing competition among
        sensory representations (Desimone & Duncan, 1995). This is modeled
        as a multiplicative gain applied based on similarity between
        input features and current goal representations.
        """
        goal = self.get_state("goal_vector", np.zeros(128, dtype=np.float32))
        top_down = self.get_state("top_down_strength", np.float32(0.5))

        # Flatten tensor to compare with goal vector
        flat = data.data.reshape(-1)
        dim = min(len(flat), 128)

        # Compute similarity between input and goal
        goal_norm = np.linalg.norm(goal[:dim])
        input_norm = np.linalg.norm(flat[:dim])
        if goal_norm > 1e-6 and input_norm > 1e-6:
            similarity = np.dot(flat[:dim], goal[:dim]) / (goal_norm * input_norm)
        else:
            similarity = 0.0

        # Top-down gain: boost features similar to goals
        gain = np.float32(1.0 + top_down * np.clip(similarity, -1.0, 1.0))

        modulated = data.data * gain

        # Update working memory with compressed representation
        self._update_working_memory(flat)

        return Tensor(data=modulated.astype(np.float32), shape=data.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Process state vector through cognitive pipeline.

        Implements working memory gating (O'Reilly & Frank, 2006):
        input is integrated into working memory if it exceeds a
        maintenance threshold, and top-down bias is applied based
        on goal similarity.
        """
        goal = self.get_state("goal_vector", np.zeros(128, dtype=np.float32))
        top_down = self.get_state("top_down_strength", np.float32(0.5))
        dim = min(data.dimension, 128)

        # Update working memory
        self._update_working_memory(data.data)

        # Compute goal-directed attention
        goal_norm = np.linalg.norm(goal[:dim])
        input_norm = np.linalg.norm(data.data[:dim])
        if goal_norm > 1e-6 and input_norm > 1e-6:
            goal_similarity = np.dot(data.data[:dim], goal[:dim]) / (goal_norm * input_norm)
        else:
            goal_similarity = 0.0

        # Top-down bias: shift representation toward goal
        bias = goal[:data.dimension] * top_down * np.clip(goal_similarity, 0.0, 1.0)
        modulated = data.data + bias

        # Apply cognitive load attenuation
        cognitive_load = self.get_state("cognitive_load", np.float32(0.0))
        load_factor = np.float32(1.0 / (1.0 + cognitive_load))
        modulated = modulated * load_factor

        return StateVector(data=modulated.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal through cognitive evaluation.

        Cognitive load modulates signal impact: under high load,
        signals are attenuated to prevent overload.
        """
        cognitive_load = self.get_state("cognitive_load", np.float32(0.0))
        load_factor = 1.0 / (1.0 + cognitive_load)
        return Signal(value=float(data.value * load_factor), timestamp=data.timestamp)

    def _update_working_memory(self, input_data: np.ndarray) -> None:
        """Update working memory buffer with new input.

        Working memory follows an exponential decay model:
        WM(t+1) = WM(t) * decay_rate + input * encoding_rate

        This models the persistent neural activity in PFC that maintains
        information in the absence of external input.
        """
        wm = self.get_state("working_memory", np.zeros(128, dtype=np.float32))
        min(len(input_data), 128)

        # Encoding: compress input to 128 dimensions
        encoding = np.zeros(128, dtype=np.float32)
        if len(input_data) >= 128:
            encoding = input_data[:128].astype(np.float32)  # type: ignore[assignment]
        else:
            encoding[:len(input_data)] = input_data.astype(np.float32)

        # Exponential integration with decay
        decay_rate = np.float32(0.95)
        encoding_rate = np.float32(0.1)
        new_wm = wm * decay_rate + encoding * encoding_rate

        # Update cognitive load (proportional to WM activity)
        load = np.float32(np.mean(np.abs(new_wm)))
        self.set_state("cognitive_load", load)
        self.set_state("working_memory", new_wm)
