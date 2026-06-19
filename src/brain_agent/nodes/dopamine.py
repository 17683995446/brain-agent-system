"""L03: DopamineSystemNode - Reward prediction error and reinforcement learning.

Neuroscience basis:
The dopamine system is central to reinforcement learning and reward processing:
1. Reward Prediction Error (RPE): Dopamine neurons in VTA and substantia nigra
   pars compacta (SNc) fire phasically when outcomes exceed expectations
   (Schultz et al., 1997). RPE = R(t) - E[R(t)] where R is reward and E is expectation.
2. Temporal Difference (TD) learning: Dopamine implements the TD error signal
   that drives learning in cortical and striatal circuits (Sutton & Barto, 1998).
   TD error: delta = r + gamma * V(s') - V(s)
3. D1/D2 receptor pathways: Direct pathway (D1) facilitates desired actions,
   indirect pathway (D2) suppresses undesired actions (Frank et al., 2004).
   D1: Go (positive RPE enhances), D2: NoGo (negative RPE suppresses)
4. Tonic vs phasic dopamine: Tonic levels maintain baseline motivation,
   while phasic bursts signal unexpected rewards (Grace, 1991).
This node implements these core dopamine computations.
"""

from __future__ import annotations
import time
import numpy as np

from ..node import BrainNode, NodeConfig
from ..types import BrainData, Tensor, StateVector, Signal, Message


class DopamineSystemNode(BrainNode):
    """Dopamine system: reward prediction error.

    Implements reward prediction error (RPE = reward - expected), temporal
    difference learning signal, and D1/D2 receptor pathway modulation.

    Processing logic:
    1. Compute RPE from reward and expectation
    2. Update value estimates via TD learning
    3. Generate D1 (Go) and D2 (NoGo) signals
    4. Modulate target region activity
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)
        # Value function estimates
        self.set_state("value_estimate", np.float32(0.0))
        self.set_state("prev_value", np.float32(0.0))
        # Reward prediction error
        self.set_state("rpe", np.float32(0.0))
        self.set_state("smoothed_rpe", np.float32(0.0))
        # TD learning parameters
        self.set_state("learning_rate", np.float32(0.1))
        self.set_state("discount_factor", np.float32(0.95))
        self.set_state("eligibility_trace", np.float32(0.0))
        # D1/D2 pathway parameters
        self.set_state("d1_signal", np.float32(0.0))  # Go signal
        self.set_state("d2_signal", np.float32(0.0))  # NoGo signal
        self.set_state("d1_gain", np.float32(1.0))
        self.set_state("d2_gain", np.float32(1.0))
        # Tonic dopamine
        self.set_state("tonic_level", np.float32(0.3))
        # Reward history for expectation computation
        self.set_state("reward_history", [])

    def process(self, data: BrainData) -> BrainData:
        """Compute reward prediction error and modulate signals.

        Args:
            data: Input data (Tensor, StateVector, Signal, or Message)

        Returns:
            Dopamine-modulated output
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

    def _compute_rpe(self, reward: float) -> np.float32:
        """Compute Reward Prediction Error.

        RPE = reward - expected_reward
        Positive RPE: better than expected (dopamine burst)
        Negative RPE: worse than expected (dopamine dip)
        Zero RPE: as expected (tonic firing)
        """
        expected = self.get_state("value_estimate", np.float32(0.0))
        rpe = np.float32(reward - expected)
        self.set_state("rpe", rpe)

        # Smooth RPE for stable modulation
        smoothed = self.get_state("smoothed_rpe", np.float32(0.0))
        new_smoothed = np.float32(smoothed * 0.9 + rpe * 0.1)
        self.set_state("smoothed_rpe", new_smoothed)

        return rpe

    def _update_value(self, reward: float) -> None:
        """Update value estimate via TD learning.

        TD(0) update: V(s) = V(s) + alpha * [r + gamma * V(s') - V(s)]
        With eligibility trace: V(s) = V(s) + alpha * delta * e
        """
        lr = self.get_state("learning_rate", np.float32(0.1))
        gamma = self.get_state("discount_factor", np.float32(0.95))
        current_value = self.get_state("value_estimate", np.float32(0.0))
        eligibility = self.get_state("eligibility_trace", np.float32(0.0))

        # TD error
        td_error = reward + gamma * current_value - current_value

        # Update eligibility trace (decaying)
        new_eligibility = np.float32(gamma * eligibility + 1.0)

        # Value update
        new_value = np.float32(
            current_value + lr * td_error * new_eligibility
        )
        new_value = np.clip(new_value, -10.0, 10.0)

        self.set_state("value_estimate", new_value)
        self.set_state("prev_value", current_value)
        self.set_state("eligibility_trace", new_eligibility * 0.99)

        # Update reward history
        history = self.get_state("reward_history", [])
        history.append(float(reward))
        if len(history) > 100:
            history = history[-100:]
        self.set_state("reward_history", history)

    def _compute_d1_d2(self, rpe: np.float32) -> None:
        """Compute D1 (Go) and D2 (NoGo) pathway signals.

        D1 receptor (direct pathway): activated by positive RPE
        -> facilitates ongoing actions (Go)
        D2 receptor (indirect pathway): activated by negative RPE
        -> suppresses ongoing actions (NoGo)

        Based on Frank et al. (2004) model of basal ganglia dopamine.
        """
        d1_gain = self.get_state("d1_gain", np.float32(1.0))
        d2_gain = self.get_state("d2_gain", np.float32(1.0))

        # D1: sigmoid of positive RPE (Go signal)
        d1 = np.float32(d1_gain * np.tanh(max(0.0, float(rpe)) * 2.0))

        # D2: sigmoid of negative RPE (NoGo signal)
        d2 = np.float32(d2_gain * np.tanh(max(0.0, float(-rpe)) * 2.0))

        self.set_state("d1_signal", d1)
        self.set_state("d2_signal", d2)

    def _process_tensor(self, data: Tensor) -> Tensor:
        """Apply dopamine modulation to tensor input.

        Dopamine provides a global gain signal that modulates
        cortical-striatal processing. The gain depends on the
        current RPE and D1/D2 balance.
        """
        # Compute reward proxy from input
        reward = float(np.mean(data.data))
        rpe = self._compute_rpe(reward)
        self._update_value(reward)
        self._compute_d1_d2(rpe)

        # Dopamine gain: tonic + phasic
        tonic = self.get_state("tonic_level", np.float32(0.3))
        smoothed_rpe = self.get_state("smoothed_rpe", np.float32(0.0))

        # D1/D2 balanced gain
        d1 = self.get_state("d1_signal", np.float32(0.0))
        d2 = self.get_state("d2_signal", np.float32(0.0))

        # Net dopamine effect: D1 facilitates, D2 suppresses
        net_gain = np.float32(tonic + smoothed_rpe + d1 - d2 * 0.5)
        net_gain = np.clip(net_gain, 0.0, 2.0)

        output = data.data * net_gain
        return Tensor(data=output.astype(np.float32), shape=data.shape)

    def _process_state_vector(self, data: StateVector) -> StateVector:
        """Apply dopamine modulation to state vector.

        D1/D2 signals differentially modulate positive and negative
        components of the state vector, implementing Go/NoGo selection.
        """
        reward = float(np.mean(data.data))
        rpe = self._compute_rpe(reward)
        self._update_value(reward)
        self._compute_d1_d2(rpe)

        d1 = self.get_state("d1_signal", np.float32(0.0))
        d2 = self.get_state("d2_signal", np.float32(0.0))
        tonic = self.get_state("tonic_level", np.float32(0.3))

        # D1 enhances positive components (Go)
        positive = np.maximum(data.data, 0.0)
        # D2 suppresses negative components (NoGo)
        negative = np.minimum(data.data, 0.0)

        output = positive * (1.0 + d1) + negative * (1.0 - d2 * 0.5)
        output = output * (tonic + 0.5)

        return StateVector(data=output.astype(np.float32), dimension=data.dimension)

    def _process_signal(self, data: Signal) -> Signal:
        """Process scalar signal as reward input.

        Computes RPE, updates value function, and returns
        the dopamine signal (tonic + phasic).
        """
        reward = data.value
        rpe = self._compute_rpe(reward)
        self._update_value(reward)
        self._compute_d1_d2(rpe)

        tonic = self.get_state("tonic_level", np.float32(0.3))
        dopamine_signal = float(tonic + rpe)

        return Signal(value=dopamine_signal, timestamp=data.timestamp)
