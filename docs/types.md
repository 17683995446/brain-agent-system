# BrainAgentSystem 类型系统文档

> 本文档描述 BrainAgentSystem 的核心数据类型系统，包括所有 BrainDataType 子类、序列化/反序列化机制、以及工具函数。
>
> 版本: v1.0
> 最后更新: 2026-06-20

---

## 目录

1. [概述](#概述)
2. [BrainDataType 抽象基类](#braindatatype-抽象基类)
3. [数据类型详解](#数据类型详解)
   - [Tensor](#tensor)
   - [StateVector](#statevector)
   - [SpikeTrain](#spiketrain)
   - [Signal](#signal)
   - [Message](#message)
4. [类型别名](#类型别名)
5. [序列化与反序列化](#序列化与反序列化)
6. [工具函数](#工具函数)
7. [使用示例](#使用示例)

---

## 概述

BrainAgentSystem 的类型系统基于以下设计原则：

- **不可变性**: 所有数据类型使用 `@dataclass(frozen=True)`，确保线程安全
- **numpy 后端**: 数值数据底层使用 numpy 数组，支持高效的向量化运算
- **严格的构造时验证**: 形状、维度、数据类型在构造时自动验证
- **统一的 nbytes 接口**: 所有类型实现 `nbytes` 属性，支持内存占用统计
- **序列化支持**: 所有类型支持 dict/JSON 序列化，便于网络传输

---

## BrainDataType 抽象基类

```python
class BrainDataType:
    """Base class for all brain data types."""

    @property
    def nbytes(self) -> int:
        """Size in bytes of the underlying data buffer."""
        raise NotImplementedError
```

所有脑数据类型必须继承 `BrainDataType` 并实现 `nbytes` 属性。直接实例化 `BrainDataType` 并访问 `nbytes` 会抛出 `NotImplementedError`。

---

## 数据类型详解

### Tensor

多维张量数据，用于视觉、听觉、体感处理。

| 属性 | 类型 | 说明 |
|------|------|------|
| `data` | `np.ndarray` | float32 数组 |
| `shape` | `Tuple[int, ...]` | 期望形状（构造时验证） |

**类方法**:
- `Tensor.zeros(shape)` - 创建全零张量
- `Tensor.random(shape)` - 创建随机张量（标准正态分布）

**示例**:
```python
from brain_agent.types import Tensor

# 创建 NCHW 格式图像张量
t = Tensor.zeros((1, 3, 224, 224))

# 随机张量
t2 = Tensor.random((32, 64, 64))

# 构造时自动转换为 float32
t3 = Tensor(data=np.ones((10, 10), dtype=np.float64), shape=(10, 10))
assert t3.data.dtype == np.float32
```

---

### StateVector

固定维度的状态向量，用于记忆系统（海马体、杏仁核、前额叶皮层）。

| 属性 | 类型 | 说明 |
|------|------|------|
| `data` | `np.ndarray` | 1D float32 数组 |
| `dimension` | `int` | 期望维度（构造时验证） |

**类方法**:
- `StateVector.zeros(dimension)` - 创建零向量
- `StateVector.random(dimension)` - 创建随机向量

**方法**:
- `normalize()` - L2 归一化，返回单位向量

**属性**:
- `magnitude` - L2 范数

**示例**:
```python
from brain_agent.types import StateVector

sv = StateVector.random(128)
unit = sv.normalize()
assert abs(unit.magnitude - 1.0) < 1e-6
```

---

### SpikeTrain

稀疏脉冲序列，用于神经放电模式。

| 属性 | 类型 | 说明 |
|------|------|------|
| `timestamps` | `np.ndarray` | 1D float64 数组，脉冲时间（秒） |
| `neuron_ids` | `np.ndarray` | 1D int64 数组，发放神经元的 ID |
| `duration` | `float` | 总时间窗口（秒） |

**类方法**:
- `SpikeTrain.empty(duration=1.0)` - 创建空脉冲序列

**属性**:
- `spike_count` - 脉冲数量
- `firing_rate` - 发放率（Hz）

**验证规则**:
- `timestamps` 和 `neuron_ids` 长度必须相同
- `timestamps` 必须在 `[0, duration]` 范围内
- `duration` 必须非负

**示例**:
```python
from brain_agent.types import SpikeTrain
import numpy as np

st = SpikeTrain(
    timestamps=np.array([0.1, 0.3, 0.5], dtype=np.float64),
    neuron_ids=np.array([0, 5, 3], dtype=np.int64),
    duration=1.0,
)
print(f"发放率: {st.firing_rate} Hz")  # -> 3.0 Hz
```

---

### Signal

标量信号值，用于神经调质（多巴胺、血清素等）。

| 属性 | 类型 | 说明 |
|------|------|------|
| `value` | `float` | 信号幅度 |
| `timestamp` | `float` | 信号时间（秒，默认 0.0） |

**属性**:
- `magnitude` - 绝对值

**示例**:
```python
from brain_agent.types import Signal

dopamine = Signal(value=0.8, timestamp=1.5)
print(dopamine.magnitude)  # -> 0.8
```

---

### Message

通用消息类型，用于控制信号、配置更新。

| 属性 | 类型 | 说明 |
|------|------|------|
| `topic` | `str` | 消息主题 |
| `payload` | `dict` | 消息载荷（默认空字典） |
| `timestamp` | `float` | 消息时间（默认 0.0） |

**示例**:
```python
from brain_agent.types import Message

msg = Message(
    topic="control",
    payload={"cmd": "reset", "target": 42},
    timestamp=1.0,
)
```

---

## 类型别名

```python
BrainData = Tensor | StateVector | SpikeTrain | Signal | Message
```

所有脑数据类型的联合类型，用于类型注解。

---

## 序列化与反序列化

所有 BrainDataType 支持以下序列化函数：

### `serialize(data: BrainData) -> Dict[str, Any]`

将 BrainData 实例序列化为普通 dict。numpy 数组使用 base64 编码。

### `deserialize(serialized: Dict[str, Any]) -> BrainData`

将 dict 反序列化为 BrainData 实例。

### `to_json(data: BrainData) -> str`

序列化为 JSON 字符串。

### `from_json(json_str: str) -> BrainData`

从 JSON 字符串反序列化。

**示例**:
```python
from brain_agent.types import Tensor, serialize, deserialize, to_json, from_json

t = Tensor.random((4, 5))

# Dict 序列化
s = serialize(t)
t_back = deserialize(s)

# JSON 序列化
json_str = to_json(t)
t_back2 = from_json(json_str)
```

---

## 工具函数

### `get_type_name(data: BrainData) -> str`

获取 BrainData 实例的类型名称字符串。

```python
from brain_agent.types import get_type_name, Tensor

t = Tensor.zeros((2, 2))
print(get_type_name(t))  # -> "Tensor"
```

### `validate_tensor_shape(data: np.ndarray, expected_shape: Tuple[int, ...]) -> bool`

验证 numpy 数组是否匹配期望形状。

```python
from brain_agent.types import validate_tensor_shape
import numpy as np

data = np.zeros((3, 4))
assert validate_tensor_shape(data, (3, 4)) is True
assert validate_tensor_shape(data, (4, 3)) is False
```

---

## 使用示例

### 完整数据流示例

```python
import numpy as np
from brain_agent.types import (
    Tensor, StateVector, SpikeTrain, Signal, Message,
    serialize, deserialize, to_json, from_json,
)

# 1. 创建各种类型的数据
visual_input = Tensor.random((1, 3, 224, 224))
memory_state = StateVector.random(256)
spikes = SpikeTrain(
    timestamps=np.array([0.05, 0.12, 0.33], dtype=np.float64),
    neuron_ids=np.array([10, 25, 7], dtype=np.int64),
    duration=1.0,
)
dopamine = Signal(value=0.85, timestamp=0.5)
control_msg = Message(topic="action", payload={"move": "forward"})

# 2. 序列化为 JSON 进行网络传输
json_data = to_json(visual_input)

# 3. 在接收端反序列化
received = from_json(json_data)

# 4. 节点间通过 process() 处理并传递数据
#    （节点内部使用上述类型，无需额外的传输层）
processed = some_node.process(memory_state)
```

---

## 测试覆盖

类型系统测试覆盖：

| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|----------|
| `TestTensor` | 10 | 构造、形状验证、zeros、random、nbytes、不可变性 |
| `TestStateVector` | 13 | 构造、维度验证、normalize、magnitude、nbytes |
| `TestSpikeTrain` | 13 | 构造、空创建、firing_rate、时间戳验证、nbytes |
| `TestSignal` | 8 | 构造、magnitude、nbytes、不可变性 |
| `TestMessage` | 6 | 构造、payload、nbytes、不可变性 |
| `TestUtilityFunctions` | 7 | get_type_name、validate_tensor_shape |
| `TestTypeValidationErrors` | 10 | 各类型的错误输入验证 |
| `TestBrainDataType` | 2 | 抽象基类 nbytes、子类实现检查 |
| `TestSerialization` | 14 | serialize、deserialize、to_json、from_json |

**总计: 83 个测试**
