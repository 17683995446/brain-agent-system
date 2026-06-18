"""L02: MemorySystemNode - Encoding, storage, and retrieval.

Neuroscience basis:
The memory system implements key principles from computational neuroscience:
- Encoding: Synaptic plasticity via Hebbian learning (Bliss & Lemoine, 1973)
- Storage: Content-addressable memory via autoassociative networks (Hopfield, 1982)
- Retrieval: Pattern completion through recurrent attractor dynamics
The hippocampus serves as a temporary index for cortical memory storage
(McClelland et al., 1995), with slow learning in neocortex complementing
fast learning in hippocampus. This node implements a simplified version of
these principles using numpy operations on StateVector[float32, 128].
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class MemorySystemNode(BrainNode):
    """Memory system: encoding, storage, retrieval.

    Implements memory encoding with synaptic weight update, content-addressable
    retrieval, and consolidation. Operates on StateVector[float32, 128].

    Processing logic:
    1. Encoding: Hebbian weight update for new patterns
    2. Storage: Maintain weight matrix for content-addressable memory
    3. Retrieval: Pattern completion via attractor dynamics
    4. Consolidation: Gradual transfer to long-term storage
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Memory store: weight matrix for autoassociative memory
        self.set_state("weight_matrix", np.zeros((128, 128), dtype=np.float32))
        # Memory traces: list of stored patterns
        self.set_state("memory_traces", [])
        # Encoding parameters
        self.set_state("learning_rate", np.float32(0.1))
        self.set_state("forgetting_rate", np.float32(0.001))
        self.set_state("retrieval_threshold", np.float32(0.5))
        self.set_state("consolidation_level", np.float32(0.0))
        # Memory capacity
        self.set_state("memory_capacity", 100)

    def process(self, data: BrainData) -> BrainData:
        """Memory encoding, storage, and retrieval.

        StateVector[float32, 128] operations.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Memory-processed output
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
        """Process tensor through memory system.

        Flattens tensor to state vector, processes through memory,
        and reshapes back to original tensor shape.
        """
        flat = data.data.reshape(-1)
        dim = min(len(flat), 128)

        # Pad or truncate to 128 dimensions
        state = np.zeros(128, dtype=np.float32)
        state[:dim] = flat[:dim]

        # Process through memory
        result_state = self._memory_process(state)

        # Reshape back
        output_flat = np.zeros(len(flat), dtype=np.float32)
        output_flat[:dim] = result_state[:dim]
        output = output_flat.reshape(data.shape)

        return Tensor(data=output.astype(np.float32), shape=data.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Full memory processing pipeline for state vectors.

        1. Compute similarity to stored memories (retrieval cue)
        2. If similar memory found: pattern completion (retrieval)
        3. If novel: encode into memory (storage)
        4. Apply consolidation dynamics
        """
        dim = min(data.dimension, 128)
        input_vec = np.zeros(128, dtype=np.float32)
        input_vec[:dim] = data.data[:dim]

        result = self._memory_process(input_vec)

        return StateVector(data=result[:data.dimension].astype(np.float32),
                          dimension=data.dimension)

    def _memory_process(self, input_vec: np.ndarray) -> np.ndarray:
        """Core memory processing: encoding, retrieval, consolidation.

        Implements the Complementary Learning Systems (CLS) model:
        - Fast hippocampal learning via Hebbian plasticity
        - Slow neocortical consolidation via interleaved replay
        - Content-addressable retrieval via pattern completion
        """
        weight_matrix = self.get_state("weight_matrix", np.zeros((128, 128), dtype=np.float32))
        learning_rate = self.get_state("learning_rate", np.float32(0.1))
        forgetting_rate = self.get_state("forgetting_rate", np.float32(0.001))
        retrieval_threshold = self.get_state("retrieval_threshold", np.float32(0.5))
        memory_traces = self.get_state("memory_traces", [])
        memory_capacity = self.get_state("memory_capacity", 100)

        # Step 1: Content-addressable retrieval
        retrieved = np.zeros(128, dtype=np.float32)
        best_similarity = 0.0

        if len(memory_traces) > 0:
            for trace in memory_traces:
                # Cosine similarity
                trace_norm = np.linalg.norm(trace)
                input_norm = np.linalg.norm(input_vec)
                if trace_norm > 1e-6 and input_norm > 1e-6:
                    sim = np.dot(input_vec, trace) / (trace_norm * input_norm)
                else:
                    sim = 0.0

                if sim > best_similarity:
                    best_similarity = sim
                    retrieved = trace.copy()

            # Pattern completion: if good match found, blend with retrieved
            if best_similarity > retrieval_threshold:
                # Attractor dynamics: move toward stored pattern
                completion_strength = np.float32(best_similarity)
                output = input_vec * (1.0 - completion_strength) + \
                         retrieved * completion_strength
            else:
                output = input_vec.copy()
        else:
            output = input_vec.copy()

        # Step 2: Encoding (Hebbian weight update)
        # Delta rule: dw = lr * (input - output) * input^T
        error = input_vec - output
        weight_update = np.outer(error, input_vec) * learning_rate
        new_weights = weight_matrix + weight_update

        # Apply forgetting (weight decay)
        new_weights *= (1.0 - forgetting_rate)

        # Store updated weights
        self.set_state("weight_matrix", new_weights)

        # Step 3: Store new trace if novel
        if best_similarity < retrieval_threshold and len(memory_traces) < memory_capacity:
            new_trace = input_vec.copy()
            memory_traces.append(new_trace)
            self.set_state("memory_traces", memory_traces)

        # Step 4: Consolidation dynamics
        consolidation = self.get_state("consolidation_level", np.float32(0.0))
        new_consolidation = np.float32(min(1.0, consolidation + learning_rate * 0.01))
        self.set_state("consolidation_level", new_consolidation)

        return output

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal as memory relevance signal.

        Signal magnitude modulates learning rate: high relevance
        increases encoding strength.
        """
        learning_rate = self.get_state("learning_rate", np.float32(0.1))
        relevance = np.tanh(abs(data.value))
        effective_lr = learning_rate * (0.5 + 0.5 * relevance)
        self.set_state("learning_rate", effective_lr)
        return Signal(value=data.value, timestamp=data.timestamp)
