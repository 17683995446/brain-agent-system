"""Tests for core brain data types."""

import sys
import os
import pytest
import numpy as np

# Add src to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from brain_agent.types import (
    Tensor,
    StateVector,
    SpikeTrain,
    Signal,
    Message,
    BrainData,
    BrainDataType,
    get_type_name,
    validate_tensor_shape,
    serialize,
    deserialize,
    to_json,
    from_json,
)


# ============================================================
# Tensor Tests
# ============================================================

class TestTensor:
    """Tests for the Tensor data type."""

    def test_zeros_creates_zero_array(self):
        """Tensor.zeros should create an all-zero float32 array."""
        t = Tensor.zeros((3, 4))
        assert t.data.shape == (3, 4)
        assert t.data.dtype == np.float32
        assert np.all(t.data == 0.0)
        assert t.shape == (3, 4)

    def test_random_creates_array_of_correct_shape(self):
        """Tensor.random should create a float32 array of the given shape."""
        t = Tensor.random((2, 3, 4))
        assert t.data.shape == (2, 3, 4)
        assert t.data.dtype == np.float32
        assert t.shape == (2, 3, 4)

    def test_random_values_vary(self):
        """Tensor.random should produce non-constant values (probabilistic)."""
        t = Tensor.random((100,))
        assert t.data.std() > 0.1  # random normal has std ~1

    def test_construction_with_matching_shape(self):
        """Tensor should accept data when shape matches."""
        data = np.ones((2, 2), dtype=np.float32)
        t = Tensor(data=data, shape=(2, 2))
        assert t.shape == (2, 2)

    def test_construction_auto_infers_shape(self):
        """Tensor should infer shape from data when shape is not provided."""
        data = np.ones((5, 6), dtype=np.float32)
        t = Tensor(data=data)
        assert t.shape == (5, 6)

    def test_construction_converts_dtype(self):
        """Tensor should automatically convert non-float32 data to float32."""
        data = np.ones((2, 2), dtype=np.float64)
        t = Tensor(data=data)
        assert t.data.dtype == np.float32

    def test_shape_mismatch_raises_value_error(self):
        """Tensor should raise ValueError when data shape doesn't match expected."""
        data = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Tensor shape mismatch"):
            Tensor(data=data, shape=(2, 2))

    def test_nbytes_property(self):
        """Tensor.nbytes should return the byte size of the underlying array."""
        t = Tensor.zeros((10, 10))
        # float32 = 4 bytes per element, 100 elements
        assert t.nbytes == 400

    def test_frozen_immutability(self):
        """Tensor should be immutable (frozen dataclass)."""
        t = Tensor.zeros((2, 2))
        with pytest.raises(AttributeError):
            t.data = np.zeros((2, 2))

    def test_is_instance_of_brain_data_type(self):
        """Tensor should be an instance of BrainDataType."""
        t = Tensor.zeros((2, 2))
        assert isinstance(t, BrainDataType)

    def test_1d_tensor(self):
        """Tensor should support 1D shapes."""
        t = Tensor.zeros((128,))
        assert t.data.shape == (128,)
        assert t.shape == (128,)


# ============================================================
# StateVector Tests
# ============================================================

class TestStateVector:
    """Tests for the StateVector data type."""

    def test_zeros_creates_zero_vector(self):
        """StateVector.zeros should create an all-zero float32 vector."""
        sv = StateVector.zeros(128)
        assert sv.data.shape == (128,)
        assert sv.data.dtype == np.float32
        assert np.all(sv.data == 0.0)
        assert sv.dimension == 128

    def test_random_creates_vector_of_correct_dimension(self):
        """StateVector.random should create a float32 vector of given dimension."""
        sv = StateVector.random(256)
        assert sv.data.shape == (256,)
        assert sv.data.dtype == np.float32
        assert sv.dimension == 256

    def test_construction_with_matching_dimension(self):
        """StateVector should accept data when dimension matches."""
        data = np.arange(64, dtype=np.float32)
        sv = StateVector(data=data, dimension=64)
        assert sv.dimension == 64

    def test_construction_auto_infers_dimension(self):
        """StateVector should infer dimension from data when not provided."""
        data = np.arange(32, dtype=np.float32)
        sv = StateVector(data=data)
        assert sv.dimension == 32

    def test_construction_converts_dtype(self):
        """StateVector should convert non-float32 data to float32."""
        data = np.arange(10, dtype=np.int32)
        sv = StateVector(data=data)
        assert sv.data.dtype == np.float32

    def test_dimension_mismatch_raises_value_error(self):
        """StateVector should raise ValueError when dimension doesn't match."""
        data = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="StateVector dimension mismatch"):
            StateVector(data=data, dimension=20)

    def test_non_1d_raises_value_error(self):
        """StateVector should raise ValueError when data is not 1D."""
        data = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="StateVector must be 1-dimensional"):
            StateVector(data=data)

    def test_normalize_unit_vector(self):
        """StateVector.normalize should produce a unit vector."""
        data = np.array([3.0, 4.0], dtype=np.float32)
        sv = StateVector(data=data)
        normalized = sv.normalize()
        assert abs(normalized.magnitude - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        """StateVector.normalize on a zero vector should return a zero vector."""
        sv = StateVector.zeros(10)
        normalized = sv.normalize()
        assert normalized.magnitude == 0.0
        assert normalized.dimension == 10

    def test_magnitude(self):
        """StateVector.magnitude should return the L2 norm."""
        data = np.array([3.0, 4.0], dtype=np.float32)
        sv = StateVector(data=data)
        assert abs(sv.magnitude - 5.0) < 1e-6

    def test_magnitude_zero_vector(self):
        """StateVector.magnitude of a zero vector should be 0.0."""
        sv = StateVector.zeros(5)
        assert sv.magnitude == 0.0

    def test_frozen_immutability(self):
        """StateVector should be immutable."""
        sv = StateVector.zeros(10)
        with pytest.raises(AttributeError):
            sv.data = np.zeros(10)

    def test_is_instance_of_brain_data_type(self):
        """StateVector should be an instance of BrainDataType."""
        sv = StateVector.zeros(10)
        assert isinstance(sv, BrainDataType)

    def test_nbytes_property(self):
        """StateVector.nbytes should return the byte size of the underlying array."""
        sv = StateVector.zeros(128)
        assert sv.nbytes == 128 * 4  # float32 = 4 bytes


# ============================================================
# SpikeTrain Tests
# ============================================================

class TestSpikeTrain:
    """Tests for the SpikeTrain data type."""

    def test_empty_creates_no_spikes(self):
        """SpikeTrain.empty should create a train with no spikes."""
        st = SpikeTrain.empty(duration=2.0)
        assert st.spike_count == 0
        assert st.duration == 2.0
        assert st.firing_rate == 0.0

    def test_construction_with_valid_data(self):
        """SpikeTrain should accept valid timestamps and neuron_ids."""
        timestamps = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        neuron_ids = np.array([1, 5, 3], dtype=np.int64)
        st = SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)
        assert st.spike_count == 3
        assert st.duration == 1.0

    def test_construction_converts_dtypes(self):
        """SpikeTrain should convert timestamps to float64 and neuron_ids to int64."""
        timestamps = np.array([0.1, 0.5], dtype=np.float32)
        neuron_ids = np.array([1, 2], dtype=np.int32)
        st = SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)
        assert st.timestamps.dtype == np.float64
        assert st.neuron_ids.dtype == np.int64

    def test_spike_count(self):
        """SpikeTrain.spike_count should return the number of spikes."""
        timestamps = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        neuron_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        st = SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)
        assert st.spike_count == 4

    def test_firing_rate(self):
        """SpikeTrain.firing_rate should return spikes per second."""
        timestamps = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        neuron_ids = np.array([0, 1, 2], dtype=np.int64)
        st = SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=2.0)
        assert st.firing_rate == 1.5

    def test_firing_rate_zero_duration(self):
        """SpikeTrain.firing_rate with duration=0 should return 0.0."""
        st = SpikeTrain.empty(duration=0.0)
        assert st.firing_rate == 0.0

    def test_length_mismatch_raises_value_error(self):
        """SpikeTrain should raise ValueError when timestamps and neuron_ids differ in length."""
        timestamps = np.array([0.1, 0.2], dtype=np.float64)
        neuron_ids = np.array([1], dtype=np.int64)
        with pytest.raises(ValueError, match="Timestamps and neuron IDs length mismatch"):
            SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)

    def test_negative_duration_raises_value_error(self):
        """SpikeTrain should raise ValueError when duration is negative."""
        with pytest.raises(ValueError, match="Duration cannot be negative"):
            SpikeTrain.empty(duration=-1.0)

    def test_timestamp_out_of_range_raises_value_error(self):
        """SpikeTrain should raise ValueError when a timestamp is outside [0, duration]."""
        timestamps = np.array([0.1, 1.5], dtype=np.float64)
        neuron_ids = np.array([1, 2], dtype=np.int64)
        with pytest.raises(ValueError, match="Timestamps must be within"):
            SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)

    def test_negative_timestamp_raises_value_error(self):
        """SpikeTrain should raise ValueError when a timestamp is negative."""
        timestamps = np.array([-0.1, 0.5], dtype=np.float64)
        neuron_ids = np.array([1, 2], dtype=np.int64)
        with pytest.raises(ValueError, match="Timestamps must be within"):
            SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)

    def test_frozen_immutability(self):
        """SpikeTrain should be immutable."""
        st = SpikeTrain.empty()
        with pytest.raises(AttributeError):
            st.duration = 5.0

    def test_is_instance_of_brain_data_type(self):
        """SpikeTrain should be an instance of BrainDataType."""
        st = SpikeTrain.empty()
        assert isinstance(st, BrainDataType)

    def test_nbytes_property(self):
        """SpikeTrain.nbytes should return total bytes of timestamps + neuron_ids."""
        timestamps = np.array([0.1, 0.2], dtype=np.float64)
        neuron_ids = np.array([1, 2], dtype=np.int64)
        st = SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)
        assert st.nbytes == timestamps.nbytes + neuron_ids.nbytes


# ============================================================
# Signal Tests
# ============================================================

class TestSignal:
    """Tests for the Signal data type."""

    def test_creation_with_value(self):
        """Signal should store a value and default timestamp."""
        s = Signal(value=0.75)
        assert s.value == 0.75
        assert s.timestamp == 0.0

    def test_creation_with_timestamp(self):
        """Signal should store a value and custom timestamp."""
        s = Signal(value=1.5, timestamp=3.0)
        assert s.value == 1.5
        assert s.timestamp == 3.0

    def test_magnitude_positive(self):
        """Signal.magnitude should return absolute value."""
        s = Signal(value=0.8)
        assert s.magnitude == 0.8

    def test_magnitude_negative(self):
        """Signal.magnitude should return absolute value for negative values."""
        s = Signal(value=-0.5)
        assert s.magnitude == 0.5

    def test_magnitude_zero(self):
        """Signal.magnitude should return 0 for zero value."""
        s = Signal(value=0.0)
        assert s.magnitude == 0.0

    def test_frozen_immutability(self):
        """Signal should be immutable."""
        s = Signal(value=1.0)
        with pytest.raises(AttributeError):
            s.value = 2.0

    def test_is_instance_of_brain_data_type(self):
        """Signal should be an instance of BrainDataType."""
        s = Signal(value=1.0)
        assert isinstance(s, BrainDataType)

    def test_nbytes_property(self):
        """Signal.nbytes should return 8 (float64 scalar)."""
        s = Signal(value=0.5)
        assert s.nbytes == 8


# ============================================================
# Message Tests
# ============================================================

class TestMessage:
    """Tests for the Message data type."""

    def test_creation_with_topic(self):
        """Message should store a topic with default payload and timestamp."""
        m = Message(topic="alert")
        assert m.topic == "alert"
        assert m.payload == {}
        assert m.timestamp == 0.0

    def test_creation_with_payload(self):
        """Message should store a topic and custom payload."""
        m = Message(topic="config", payload={"key": "value"})
        assert m.topic == "config"
        assert m.payload == {"key": "value"}

    def test_creation_with_timestamp(self):
        """Message should store a topic, payload, and timestamp."""
        m = Message(topic="event", payload={"x": 1}, timestamp=5.0)
        assert m.topic == "event"
        assert m.payload == {"x": 1}
        assert m.timestamp == 5.0

    def test_frozen_immutability(self):
        """Message should be immutable."""
        m = Message(topic="test")
        with pytest.raises(AttributeError):
            m.topic = "changed"

    def test_is_instance_of_brain_data_type(self):
        """Message should be an instance of BrainDataType."""
        m = Message(topic="test")
        assert isinstance(m, BrainDataType)

    def test_nbytes_property(self):
        """Message.nbytes should return positive byte count."""
        m = Message(topic="cmd", payload={"key": "value"})
        assert m.nbytes > 0


# ============================================================
# Utility Function Tests
# ============================================================

class TestUtilityFunctions:
    """Tests for get_type_name and validate_tensor_shape."""

    def test_get_type_name_tensor(self):
        """get_type_name should return 'Tensor' for Tensor instances."""
        t = Tensor.zeros((2, 2))
        assert get_type_name(t) == "Tensor"

    def test_get_type_name_state_vector(self):
        """get_type_name should return 'StateVector' for StateVector instances."""
        sv = StateVector.zeros(10)
        assert get_type_name(sv) == "StateVector"

    def test_get_type_name_spike_train(self):
        """get_type_name should return 'SpikeTrain' for SpikeTrain instances."""
        st = SpikeTrain.empty()
        assert get_type_name(st) == "SpikeTrain"

    def test_get_type_name_signal(self):
        """get_type_name should return 'Signal' for Signal instances."""
        s = Signal(value=1.0)
        assert get_type_name(s) == "Signal"

    def test_get_type_name_message(self):
        """get_type_name should return 'Message' for Message instances."""
        m = Message(topic="test")
        assert get_type_name(m) == "Message"

    def test_validate_tensor_shape_matching(self):
        """validate_tensor_shape should return True for matching shapes."""
        data = np.zeros((3, 4))
        assert validate_tensor_shape(data, (3, 4)) is True

    def test_validate_tensor_shape_mismatching(self):
        """validate_tensor_shape should return False for mismatching shapes."""
        data = np.zeros((3, 4))
        assert validate_tensor_shape(data, (4, 3)) is False


# ============================================================
# Type Validation Error Tests
# ============================================================

class TestTypeValidationErrors:
    """Comprehensive tests for validation errors across all types."""

    def test_tensor_wrong_shape_1d_vs_2d(self):
        """Tensor should reject 1D data when 2D shape is expected."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Tensor shape mismatch"):
            Tensor(data=data, shape=(3, 1))

    def test_tensor_wrong_shape_size_mismatch(self):
        """Tensor should reject data with wrong total element count."""
        data = np.ones((10,), dtype=np.float32)
        with pytest.raises(ValueError, match="Tensor shape mismatch"):
            Tensor(data=data, shape=(5,))

    def test_state_vector_2d_input_raises(self):
        """StateVector should reject 2D array input."""
        data = np.ones((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="StateVector must be 1-dimensional"):
            StateVector(data=data)

    def test_state_vector_3d_input_raises(self):
        """StateVector should reject 3D array input."""
        data = np.ones((2, 3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="StateVector must be 1-dimensional"):
            StateVector(data=data)

    def test_state_vector_dimension_mismatch(self):
        """StateVector should reject when data length differs from expected dimension."""
        data = np.zeros(50, dtype=np.float32)
        with pytest.raises(ValueError, match="StateVector dimension mismatch"):
            StateVector(data=data, dimension=100)

    def test_spike_train_mismatched_lengths(self):
        """SpikeTrain should reject when timestamps and neuron_ids have different lengths."""
        timestamps = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        neuron_ids = np.array([1, 2], dtype=np.int64)
        with pytest.raises(ValueError, match="Timestamps and neuron IDs length mismatch"):
            SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)

    def test_spike_train_negative_duration(self):
        """SpikeTrain should reject negative duration."""
        timestamps = np.array([0.1], dtype=np.float64)
        neuron_ids = np.array([1], dtype=np.int64)
        with pytest.raises(ValueError, match="Duration cannot be negative"):
            SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=-0.5)

    def test_spike_train_timestamp_exceeds_duration(self):
        """SpikeTrain should reject timestamps beyond the duration."""
        timestamps = np.array([0.5, 1.5], dtype=np.float64)
        neuron_ids = np.array([1, 2], dtype=np.int64)
        with pytest.raises(ValueError, match="Timestamps must be within"):
            SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)

    def test_spike_train_negative_timestamp(self):
        """SpikeTrain should reject negative timestamps."""
        timestamps = np.array([-0.01, 0.5], dtype=np.float64)
        neuron_ids = np.array([1, 2], dtype=np.int64)
        with pytest.raises(ValueError, match="Timestamps must be within"):
            SpikeTrain(timestamps=timestamps, neuron_ids=neuron_ids, duration=1.0)


# ============================================================
# BrainDataType Base Class Tests
# ============================================================

class TestBrainDataType:
    """Tests for the BrainDataType abstract base class."""

    def test_nbytes_raises_not_implemented(self):
        """BrainDataType.nbytes should raise NotImplementedError."""
        b = BrainDataType()
        with pytest.raises(NotImplementedError):
            _ = b.nbytes

    def test_all_subclasses_implement_nbytes(self):
        """All BrainDataType subclasses should implement nbytes without error."""
        t = Tensor.zeros((2, 2))
        sv = StateVector.zeros(10)
        st = SpikeTrain.empty()
        s = Signal(value=1.0)
        m = Message(topic="test")
        assert t.nbytes >= 0
        assert sv.nbytes >= 0
        assert st.nbytes >= 0
        assert s.nbytes >= 0
        assert m.nbytes >= 0


# ============================================================
# Serialization / Deserialization Tests
# ============================================================

class TestSerialization:
    """Tests for serialize / deserialize / to_json / from_json."""

    def test_serialize_tensor(self):
        """serialize should produce a dict with type and data keys."""
        t = Tensor.zeros((2, 3))
        s = serialize(t)
        assert s["type"] == "Tensor"
        assert "buffer" in s["data"]
        assert s["data"]["shape"] == [2, 3]

    def test_deserialize_tensor(self):
        """deserialize should reconstruct Tensor with identical data."""
        t = Tensor.random((4, 5))
        s = serialize(t)
        t_back = deserialize(s)
        assert isinstance(t_back, Tensor)
        assert t_back.shape == t.shape
        assert np.allclose(t_back.data, t.data)

    def test_serialize_state_vector(self):
        """serialize should produce correct dict for StateVector."""
        sv = StateVector.random(64)
        s = serialize(sv)
        assert s["type"] == "StateVector"
        assert s["data"]["dimension"] == 64

    def test_deserialize_state_vector(self):
        """deserialize should reconstruct StateVector with identical data."""
        sv = StateVector.random(128)
        s = serialize(sv)
        sv_back = deserialize(s)
        assert isinstance(sv_back, StateVector)
        assert sv_back.dimension == sv.dimension
        assert np.allclose(sv_back.data, sv.data)

    def test_serialize_spike_train(self):
        """serialize should produce correct dict for SpikeTrain."""
        st = SpikeTrain(
            timestamps=np.array([0.1, 0.3], dtype=np.float64),
            neuron_ids=np.array([0, 5], dtype=np.int64),
            duration=1.0,
        )
        s = serialize(st)
        assert s["type"] == "SpikeTrain"
        assert s["data"]["duration"] == 1.0

    def test_deserialize_spike_train(self):
        """deserialize should reconstruct SpikeTrain with identical data."""
        st = SpikeTrain(
            timestamps=np.array([0.1, 0.3, 0.5], dtype=np.float64),
            neuron_ids=np.array([0, 5, 3], dtype=np.int64),
            duration=1.0,
        )
        s = serialize(st)
        st_back = deserialize(s)
        assert isinstance(st_back, SpikeTrain)
        assert st_back.spike_count == st.spike_count
        assert np.allclose(st_back.timestamps, st.timestamps)
        assert np.array_equal(st_back.neuron_ids, st.neuron_ids)

    def test_serialize_signal(self):
        """serialize should produce correct dict for Signal."""
        sig = Signal(value=-0.75, timestamp=123.45)
        s = serialize(sig)
        assert s["type"] == "Signal"
        assert s["data"]["value"] == -0.75
        assert s["data"]["timestamp"] == 123.45

    def test_deserialize_signal(self):
        """deserialize should reconstruct Signal with identical values."""
        sig = Signal(value=0.99, timestamp=42.0)
        s = serialize(sig)
        sig_back = deserialize(s)
        assert isinstance(sig_back, Signal)
        assert sig_back.value == sig.value
        assert sig_back.timestamp == sig.timestamp

    def test_serialize_message(self):
        """serialize should produce correct dict for Message."""
        msg = Message(topic="control", payload={"cmd": "reset"}, timestamp=1.0)
        s = serialize(msg)
        assert s["type"] == "Message"
        assert s["data"]["topic"] == "control"
        assert s["data"]["payload"] == {"cmd": "reset"}

    def test_deserialize_message(self):
        """deserialize should reconstruct Message with identical data."""
        msg = Message(topic="config", payload={"lr": 0.001}, timestamp=99.9)
        s = serialize(msg)
        msg_back = deserialize(s)
        assert isinstance(msg_back, Message)
        assert msg_back.topic == msg.topic
        assert msg_back.payload == msg.payload
        assert msg_back.timestamp == msg.timestamp

    def test_to_json_from_json_tensor(self):
        """to_json / from_json should round-trip Tensor correctly."""
        t = Tensor.random((3, 4, 5))
        json_str = to_json(t)
        t_back = from_json(json_str)
        assert isinstance(t_back, Tensor)
        assert t_back.shape == t.shape
        assert np.allclose(t_back.data, t.data)

    def test_to_json_from_json_message(self):
        """to_json / from_json should round-trip Message correctly."""
        msg = Message(topic="event", payload={"x": [1, 2, 3]}, timestamp=7.5)
        json_str = to_json(msg)
        msg_back = from_json(json_str)
        assert isinstance(msg_back, Message)
        assert msg_back.topic == msg.topic
        assert msg_back.payload == msg.payload
        assert msg_back.timestamp == msg.timestamp

    def test_deserialize_unknown_type_raises(self):
        """deserialize should raise TypeError for unknown type strings."""
        with pytest.raises(TypeError, match="Unknown BrainData type"):
            deserialize({"type": "UnknownType", "data": {}})

    def test_serialize_unsupported_type_raises(self):
        """serialize should raise TypeError for unsupported types."""
        with pytest.raises(TypeError, match="Unsupported BrainData type"):
            serialize(BrainDataType())
