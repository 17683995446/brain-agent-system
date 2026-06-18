"""L03: HippocampalCircuitNode - Episodic memory processing.

Neuroscience basis:
The hippocampus is critical for episodic memory formation and spatial navigation.
Its internal circuitry implements distinct computational functions:
1. Dentate Gyrus (DG): Pattern separation - transforms similar inputs into
   dissimilar representations (Marr, 1971; Rolls et al., 2006)
2. CA3: Autoassociative network for pattern completion - can reconstruct
   complete memories from partial cues via recurrent collaterals (Hopfield, 1982)
3. CA1: Temporal sequence encoding and mismatch detection - compares input
   from EC with CA3 output to detect novelty (Lisman & Otmakhova, 2001)
This node implements these three core hippocampal computations.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class HippocampalCircuitNode(BrainNode):
    """Hippocampal circuit: episodic memory.

    Implements pattern separation (dentate gyrus), pattern completion (CA3),
    and temporal sequence encoding (CA1).

    Processing logic:
    1. DG: Sparse random projection for pattern separation
    2. CA3: Autoassociative retrieval via recurrent dynamics
    3. CA1: Temporal sequence encoding and novelty detection
    4. Output: Integrated hippocampal representation
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Dentate Gyrus: sparse random projection matrix
        self.set_state("dg_projection", self._init_dg_projection(128, 256))
        # CA3: recurrent weight matrix for autoassociation
        self.set_state("ca3_weights", np.zeros((256, 256), dtype=np.float32))
        # CA3: stored patterns for pattern completion
        self.set_state("ca3_patterns", [])
        # CA1: temporal context buffer
        self.set_state("ca1_context", np.zeros(256, dtype=np.float32))
        self.set_state("ca1_prev_input", np.zeros(256, dtype=np.float32))
        # Learning parameters
        self.set_state("dg_sparsity", np.float32(0.1))
        self.set_state("ca3_learning_rate", np.float32(0.05))
        self.set_state("ca1_learning_rate", np.float32(0.02))
        self.set_state("pattern_capacity", 50)

    def _init_dg_projection(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Initialize Dentate Gyrus sparse random projection.

        The DG uses sparse random connectivity to perform pattern separation:
        similar inputs are mapped to dissimilar, sparse representations.
        This is modeled as a sparse random matrix with low connection probability.
        """
        # Sparse random projection: ~10% connectivity
        projection = np.random.randn(input_dim, output_dim).astype(np.float32)
        mask = np.random.binomial(1, 0.1, (input_dim, output_dim)).astype(np.float32)
        projection *= mask
        # Normalize columns
        col_norms = np.linalg.norm(projection, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-6)
        projection /= col_norms
        return projection

    def process(self, data: BrainData) -> BrainData:
        """Process input through hippocampal circuit.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Hippocampal-processed output
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
        """Process tensor through hippocampal circuit.

        Flattens to vector, processes through DG->CA3->CA1 pipeline,
        reshapes back to original shape.
        """
        flat = data.data.reshape(-1)
        dim = min(len(flat), 128)
        input_vec = np.zeros(128, dtype=np.float32)
        input_vec[:dim] = flat[:dim]

        output_vec = self._hippocampal_process(input_vec)

        output_flat = np.zeros(len(flat), dtype=np.float32)
        out_dim = min(len(output_vec), len(flat))
        output_flat[:out_dim] = output_vec[:out_dim]
        output = output_flat.reshape(data.shape)

        return Tensor(data=output.astype(np.float32), shape=data.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Process state vector through full hippocampal pipeline.

        DG -> CA3 -> CA1 processing chain.
        """
        dim = min(data.dimension, 128)
        input_vec = np.zeros(128, dtype=np.float32)
        input_vec[:dim] = data.data[:dim]

        output_vec = self._hippocampal_process(input_vec)

        return StateVector(
            data=output_vec[:data.dimension].astype(np.float32),
            dimension=data.dimension
        )

    def _hippocampal_process(self, input_vec: np.ndarray) -> np.ndarray:
        """Core hippocampal processing: DG -> CA3 -> CA1.

        Step 1 - Dentate Gyrus (Pattern Separation):
        Sparse random projection maps input to a high-dimensional sparse
        representation where similar inputs become dissimilar.

        Step 2 - CA3 (Pattern Completion):
        Autoassociative network retrieves stored patterns that match
        the DG output. If a similar pattern exists, it is completed;
        otherwise, the DG output passes through.

        Step 3 - CA1 (Temporal Encoding):
        Integrates CA3 output with temporal context to encode sequences
        and detect novelty.
        """
        dg_proj = self.get_state("dg_projection", np.zeros((128, 256), dtype=np.float32))
        sparsity = self.get_state("dg_sparsity", np.float32(0.1))

        # Step 1: Dentate Gyrus - Pattern Separation
        dg_output = input_vec @ dg_proj  # (128,) @ (128, 256) -> (256,)

        # Apply sparsity (k-winners-take-all)
        k = max(1, int(sparsity * len(dg_output)))
        threshold = np.sort(np.abs(dg_output))[-k]
        dg_output = np.where(np.abs(dg_output) >= threshold, dg_output, 0.0)
        dg_output = dg_output.astype(np.float32)

        # Step 2: CA3 - Pattern Completion
        ca3_weights = self.get_state("ca3_weights", np.zeros((256, 256), dtype=np.float32))
        ca3_patterns = self.get_state("ca3_patterns", [])
        ca3_lr = self.get_state("ca3_learning_rate", np.float32(0.05))
        pattern_capacity = self.get_state("pattern_capacity", 50)

        # Recurrent dynamics: one step of attractor update
        ca3_activation = dg_output.copy()
        if np.any(ca3_weights):
            recurrent = ca3_activation @ ca3_weights
            ca3_activation = np.tanh(dg_output + 0.5 * recurrent).astype(np.float32)

        # Check for pattern match (content-addressable retrieval)
        best_match = -1
        best_similarity = 0.0
        for idx, pattern in enumerate(ca3_patterns):
            norm_p = np.linalg.norm(pattern)
            norm_a = np.linalg.norm(ca3_activation)
            if norm_p > 1e-6 and norm_a > 1e-6:
                sim = np.dot(ca3_activation, pattern) / (norm_p * norm_a)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = idx

        # Pattern completion: if good match, blend with stored pattern
        if best_match >= 0 and best_similarity > 0.7:
            completed = ca3_patterns[best_match]
            ca3_output = ca3_activation * (1.0 - best_similarity) + \
                         completed * best_similarity
        else:
            ca3_output = ca3_activation

        # Store new pattern (Hebbian learning in CA3)
        if best_similarity < 0.7 and len(ca3_patterns) < pattern_capacity:
            ca3_patterns.append(ca3_output.copy())
            # Update recurrent weights
            weight_update = np.outer(ca3_output, ca3_output) * ca3_lr
            ca3_weights = ca3_weights + weight_update
            self.set_state("ca3_weights", ca3_weights)
            self.set_state("ca3_patterns", ca3_patterns)

        # Step 3: CA1 - Temporal Sequence Encoding
        ca1_context = self.get_state("ca1_context", np.zeros(256, dtype=np.float32))
        ca1_prev = self.get_state("ca1_prev_input", np.zeros(256, dtype=np.float32))
        self.get_state("ca1_learning_rate", np.float32(0.02))

        # Temporal difference: compare current input with previous
        temporal_diff = ca3_output - ca1_prev

        # Novelty detection: high temporal difference = novel stimulus
        novelty = np.float32(np.linalg.norm(temporal_diff))

        # Update temporal context
        new_context = ca1_context * 0.9 + ca3_output * 0.1
        self.set_state("ca1_context", new_context)
        self.set_state("ca1_prev_input", ca3_output.copy())

        # CA1 output: combination of CA3 output and temporal context
        ca1_output = ca3_output * 0.7 + new_context * 0.3

        # Novelty modulation: enhance novel representations
        novelty_gain = np.float32(1.0 + 0.5 * np.tanh(novelty))
        ca1_output = ca1_output * novelty_gain

        return ca1_output

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal through hippocampal novelty detection.

        Signal magnitude indicates novelty; high values trigger
        enhanced encoding.
        """
        ca1_lr = self.get_state("ca1_learning_rate", np.float32(0.02))
        novelty = float(np.tanh(abs(data.value)))

        # Increase learning rate for novel stimuli
        new_lr = np.float32(ca1_lr * (1.0 + novelty))
        self.set_state("ca1_learning_rate", new_lr)

        return Signal(value=data.value, timestamp=data.timestamp)
