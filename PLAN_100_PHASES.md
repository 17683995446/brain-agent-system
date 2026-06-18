# BrainAgentSystem 100阶段实现计划

> 本文档提供BrainAgentSystem（2047节点大脑仿真Agent系统）的完整100阶段实施路线图。
> 
> 版本: v1.0
> 最后更新: 2026-06-04

---

## 目录

1. [基础设施 (阶段1-10)](#阶段1-10-基础设施)
2. [核心类型和接口 (阶段11-20)](#阶段11-20-核心类型和接口)
3. [节点基类 (阶段21-30)](#阶段21-30-节点基类)
4. [通信协议 (阶段31-40)](#阶段31-40-通信协议)
5. [树构建 (阶段41-50)](#阶段41-50-树构建)
6. [核心节点实现 L00-L03 (阶段51-60)](#阶段51-60-核心节点实现-l00-l03)
7. [代码生成 L04-L10 (阶段61-70)](#阶段61-70-代码生成-l04-l10)
8. [调度器和时序 (阶段71-80)](#阶段71-80-调度器和时序)
9. [监控和可观测性 (阶段81-90)](#阶段81-90-监控和可观测性)
10. [测试、部署、文档 (阶段91-100)](#阶段91-100-测试部署文档)

---

## 阶段1-10: 基础设施

### 阶段1: 项目目录结构初始化

| 属性 | 内容 |
|------|------|
| **目标** | 创建标准化的Python项目目录结构 |
| **交付物** | 目录结构: `src/brain_agent/`, `tests/`, `configs/`, `data/`, `docs/` |
| **技术细节** | 使用`mkdir -p`创建嵌套目录；遵循Python包规范 |
| **验证标准** | `ls -R src/` 显示完整目录树；无权限错误 |
| **工作量** | S |
| **依赖** | 无 |

**具体任务**:
```bash
mkdir -p src/brain_agent/{nodes,protocols,monitoring,generated}
mkdir -p tests/{unit,integration,benchmarks}
mkdir -p configs data docs scripts
```

---

### 阶段2: Python包配置 (pyproject.toml)

| 属性 | 内容 |
|------|------|
| **目标** | 配置现代Python包管理，支持可编辑安装 |
| **交付物** | `/workspace/BrainAgentSystem/pyproject.toml` |
| **技术细节** | 使用`[project]`元数据；定义`[project.optional-dependencies]`的dev依赖；配置`[tool.pytest.ini_options]` |
| **验证标准** | `pip install -e .` 成功；`python -c "import brain_agent"` 无错误 |
| **工作量** | S |
| **依赖** | 阶段1 |

**核心配置项**:
```toml
[project]
name = "brain-agent-system"
version = "0.1.0"
dependencies = ["numpy>=1.24.0"]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov", "black", "mypy"]
```

---

### 阶段3: Git仓库初始化与.gitignore

| 属性 | 内容 |
|------|------|
| **目标** | 初始化版本控制并配置忽略规则 |
| **交付物** | `.git/`, `.gitignore`, `.gitattributes` |
| **技术细节** | 忽略`__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.egg-info/`, 生成的代码文件 |
| **验证标准** | `git status` 显示干净状态；`git log` 显示初始提交 |
| **工作量** | S |
| **依赖** | 阶段1 |

---

### 阶段4: Docker容器化配置

| 属性 | 内容 |
|------|------|
| **目标** | 创建可复现的运行环境 |
| **交付物** | `/workspace/BrainAgentSystem/Dockerfile`, `docker-compose.yml` |
| **技术细节** | 基于`python:3.11-slim`；多阶段构建；安装gcc/g++用于numpy编译 |
| **验证标准** | `docker build -t brain-agent .` 成功；`docker run brain-agent pytest` 通过 |
| **工作量** | M |
| **依赖** | 阶段2 |

---

### 阶段5: 开发环境脚本

| 属性 | 内容 |
|------|------|
| **目标** | 创建自动化开发任务脚本 |
| **交付物** | `scripts/setup.sh`, `scripts/test.sh`, `scripts/lint.sh` |
| **技术细节** | 使用bash脚本封装常见任务；支持虚拟环境自动创建 |
| **验证标准** | `./scripts/setup.sh` 完成环境配置；`./scripts/test.sh` 运行pytest |
| **工作量** | S |
| **依赖** | 阶段2, 阶段3 |

---

### 阶段6: 代码质量工具配置

| 属性 | 内容 |
|------|------|
| **目标** | 配置代码格式化、类型检查和静态分析 |
| **交付物** | `.black.toml`, `mypy.ini`, `.flake8` |
| **技术细节** | Black行长度100；mypy严格模式；flake8忽略E203,W503 |
| **验证标准** | `black --check src/` 通过；`mypy src/brain_agent` 无错误 |
| **工作量** | S |
| **依赖** | 阶段2 |

---

### 阶段7: 持续集成配置 (GitHub Actions)

| 属性 | 内容 |
|------|------|
| **目标** | 配置自动化CI流水线 |
| **交付物** | `.github/workflows/ci.yml` |
| **技术细节** | 矩阵测试Python 3.10/3.11；缓存pip依赖；上传覆盖率报告 |
| **验证标准** | PR触发工作流；所有检查通过 |
| **工作量** | M |
| **依赖** | 阶段3, 阶段6 |

---

### 阶段8: 日志系统配置

| 属性 | 内容 |
|------|------|
| **目标** | 实现结构化日志记录 |
| **交付物** | `src/brain_agent/logging_config.py` |
| **技术细节** | 使用Python `logging`模块；支持JSON格式；可配置日志级别 |
| **验证标准** | `python -c "from brain_agent.logging_config import setup_logging; setup_logging()"` 无错误 |
| **工作量** | S |
| **依赖** | 阶段1 |

**函数签名**:
```python
def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """配置全局日志系统"""
```

---

### 阶段9: 配置管理系统

| 属性 | 内容 |
|------|------|
| **目标** | 实现分层配置管理（文件+环境变量+代码） |
| **交付物** | `src/brain_agent/config.py`, `configs/default.yaml` |
| **技术细节** | 使用`pydantic-settings`；支持YAML/JSON；环境变量覆盖 |
| **验证标准** | 配置加载测试通过；环境变量覆盖生效 |
| **工作量** | M |
| **依赖** | 阶段2 |

---

### 阶段10: 项目文档骨架

| 属性 | 内容 |
|------|------|
| **目标** | 创建项目级文档结构 |
| **交付物** | `README.md`, `docs/architecture.md`, `docs/api.md` |
| **技术细节** | 使用Markdown格式；包含架构图占位符；API文档结构 |
| **验证标准** | 所有文档文件存在且可渲染；链接有效 |
| **工作量** | S |
| **依赖** | 阶段1 |

---

## 阶段11-20: 核心类型和接口

### 阶段11: BrainDataType 基类定义

| 属性 | 内容 |
|------|------|
| **目标** | 定义所有脑数据类型的抽象基类 |
| **交付物** | `src/brain_agent/types.py` 中的 `BrainDataType` 类 |
| **技术细节** | 使用`@dataclass(frozen=True)`确保不可变性；定义`nbytes`属性 |
| **验证标准** | `python -c "from brain_agent.types import BrainDataType"` 成功 |
| **工作量** | S |
| **依赖** | 阶段1 |

**代码结构**:
```python
class BrainDataType:
    """Base class for all brain data types."""
    @property
    def nbytes(self) -> int:
        raise NotImplementedError
```

---

### 阶段12: Tensor 数据类型实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现NCHW格式的多维张量类型 |
| **交付物** | `Tensor` 类，`zeros()` 类方法，`random()` 类方法 |
| **技术细节** | dtype强制为float32；形状验证；支持NCHW布局 |
| **验证标准** | `Tensor.zeros((1,8,32,32))` 创建成功；形状不匹配抛出ValueError |
| **工作量** | M |
| **依赖** | 阶段11 |

**测试用例**:
```python
# 形状验证
t = Tensor(data=np.random.randn(1,8,32,32), shape=(1,8,32,32))  # 通过
t = Tensor(data=np.random.randn(32,32), shape=(1,8,32,32))      # 抛出ValueError
```

---

### 阶段13: StateVector 数据类型实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现固定维度的状态向量类型 |
| **交付物** | `StateVector` 类，`normalize()` 方法，`magnitude` 属性 |
| **技术细节** | 支持128/256/512维；L2归一化；一维数组强制验证 |
| **验证标准** | `StateVector.random(128).normalize().magnitude == 1.0` |
| **工作量** | M |
| **依赖** | 阶段11 |

---

### 阶段14: Signal 数据类型实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现标量信号类型 |
| **交付物** | `Signal` 类，`magnitude` 属性 |
| **技术细节** | float32标量；时间戳float64；值范围[-1.0, 1.0] |
| **验证标准** | `Signal(value=0.5, timestamp=time.time())` 创建成功 |
| **工作量** | S |
| **依赖** | 阶段11 |

---

### 阶段15: SpikeTrain 数据类型实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现稀疏脉冲序列类型 |
| **交付物** | `SpikeTrain` 类，`empty()` 类方法，`firing_rate` 属性 |
| **技术细节** | timestamps(float64)和neuron_ids(int64)数组；持续时间验证 |
| **验证标准** | `SpikeTrain.empty(duration=1.0).spike_count == 0` |
| **工作量** | M |
| **依赖** | 阶段11 |

---

### 阶段16: Message 数据类型实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现控制消息类型 |
| **交付物** | `Message` 类 |
| **技术细节** | topic字符串；payload字典；时间戳 |
| **验证标准** | `Message(topic="control", payload={"cmd": "reset"})` 创建成功 |
| **工作量** | S |
| **依赖** | 阶段11 |

---

### 阶段17: 类型验证工具函数

| 属性 | 内容 |
|------|------|
| **目标** | 实现通用类型验证工具 |
| **交付物** | `get_type_name()`, `validate_tensor_shape()` 函数 |
| **技术细节** | 使用`typing.Union`类型提示；运行时类型检查 |
| **验证标准** | `get_type_name(Tensor(...)) == "Tensor"` |
| **工作量** | S |
| **依赖** | 阶段12-16 |

---

### 阶段18: 类型单元测试

| 属性 | 内容 |
|------|------|
| **目标** | 为所有数据类型创建全面测试 |
| **交付物** | `tests/unit/test_types.py` |
| **技术细节** | 测试构造、验证、不可变性、类方法、属性 |
| **验证标准** | `pytest tests/unit/test_types.py -v` 全部通过 |
| **工作量** | M |
| **依赖** | 阶段12-17 |

**测试覆盖率要求**:
- Tensor: 构造、形状验证、zeros、random、nbytes
- StateVector: 构造、维度验证、normalize、magnitude
- SpikeTrain: 构造、空创建、firing_rate、时间戳验证
- Signal: 构造、magnitude
- Message: 构造

---

### 阶段19: RegionType 枚举定义

| 属性 | 内容 |
|------|------|
| **目标** | 定义脑区类型分类枚举 |
| **交付物** | `src/brain_agent/node.py` 中的 `RegionType` 枚举 |
| **技术细节** | SENSORY, MOTOR, MEMORY, MODULATION, RELAY, UNKNOWN |
| **验证标准** | `RegionType.SENSORY.value == "sensory"` |
| **工作量** | S |
| **依赖** | 无 |

---

### 阶段20: Protocol 枚举定义

| 属性 | 内容 |
|------|------|
| **目标** | 定义通信协议类型枚举 |
| **交付物** | `src/brain_agent/node.py` 中的 `Protocol` 枚举 |
| **技术细节** | ASYNC_PUSH, SYNC_CALL, PUB_SUB |
| **验证标准** | `Protocol.ASYNC_PUSH.value == "ASYNC_PUSH"` |
| **工作量** | S |
| **依赖** | 阶段19 |

---

## 阶段21-30: 节点基类

### 阶段21: PortSpec 数据类

| 属性 | 内容 |
|------|------|
| **目标** | 定义端口规范数据类 |
| **交付物** | `PortSpec` dataclass |
| **技术细节** | 字段: port_id, name, data_type, shape, frequency, protocol, required |
| **验证标准** | `PortSpec(port_id="IN", name="输入", data_type=Tensor, ...)` 创建成功 |
| **工作量** | S |
| **依赖** | 阶段19, 阶段20 |

---

### 阶段22: NodeConfig 数据类

| 属性 | 内容 |
|------|------|
| **目标** | 定义节点配置数据类 |
| **交付物** | `NodeConfig` dataclass |
| **技术细节** | 包含节点ID、名称、层级、父ID、region_type、协议等所有配置字段 |
| **验证标准** | 使用`@dataclass`装饰器；所有字段有类型注解 |
| **工作量** | S |
| **依赖** | 阶段21 |

---

### 阶段23: BrainNode 抽象基类框架

| 属性 | 内容 |
|------|------|
| **目标** | 实现BrainNode ABC框架 |
| **交付物** | `BrainNode` 类，`__init__`, `process` 抽象方法 |
| **技术细节** | 继承`ABC`；`@abstractmethod`装饰器；存储config和state |
| **验证标准** | 抽象类不能直接实例化；子类必须实现process |
| **工作量** | M |
| **依赖** | 阶段22 |

**代码框架**:
```python
class BrainNode(ABC):
    def __init__(self, config: NodeConfig):
        self._config = config
        self._state: Dict[str, Any] = {}
        
    @abstractmethod
    def process(self, data: BrainData) -> BrainData:
        ...
```

---

### 阶段24: 父子节点连接管理

| 属性 | 内容 |
|------|------|
| **目标** | 实现节点树连接API |
| **交付物** | `connect_parent()`, `connect_child()`, `disconnect_child()` 方法 |
| **技术细节** | 维护`_children: Dict[int, BrainNode]`和`_parent: Optional[BrainNode]` |
| **验证标准** | 连接后`child.parent == parent`；断开连接后`child.parent is None` |
| **工作量** | M |
| **依赖** | 阶段23 |

---

### 阶段25: 端口自动配置

| 属性 | 内容 |
|------|------|
| **目标** | 根据RegionType自动配置输入/输出端口 |
| **交付物** | `_setup_ports()` 方法，`REGION_TYPE_DATA_MAP`, `REGION_TYPE_SHAPE_MAP` |
| **技术细节** | SENSORY->Tensor(1,8,32,32), MEMORY->StateVector(128,), 等 |
| **验证标准** | SENSORY节点自动配置Tensor类型端口；MEMORY节点配置StateVector |
| **工作量** | M |
| **依赖** | 阶段23, 阶段19 |

---

### 阶段26: Tensor形状解析器

| 属性 | 内容 |
|------|------|
| **目标** | 实现灵活的形状字符串解析 |
| **交付物** | `_parse_tensor_shape()` 静态方法 |
| **技术细节** | 支持`[1,8,32,32]`, `(1, 8, 32, 32)`, `1x8x32x32`格式 |
| **验证标准** | 所有三种格式解析为相同元组；无效输入返回None |
| **工作量** | S |
| **依赖** | 阶段25 |

---

### 阶段27: 输入数据验证

| 属性 | 内容 |
|------|------|
| **目标** | 实现输入数据类型和形状验证 |
| **交付物** | `_validate_input()` 方法 |
| **技术细节** | 检查数据类型匹配；Tensor形状验证；不匹配抛出ValueError |
| **验证标准** | 类型错误抛出`ValueError: 输入数据类型不匹配`；形状错误抛出形状不匹配错误 |
| **工作量** | M |
| **依赖** | 阶段25, 阶段26 |

---

### 阶段28: 输出数据验证

| 属性 | 内容 |
|------|------|
| **目标** | 实现输出数据类型和形状验证 |
| **交付物** | `_validate_output()` 方法 |
| **技术细节** | 与输入验证逻辑相同；确保process返回值符合规范 |
| **验证标准** | 输出类型/形状不匹配抛出相应ValueError |
| **工作量** | S |
| **依赖** | 阶段27 |

---

### 阶段29: 节点指标追踪

| 属性 | 内容 |
|------|------|
| **目标** | 实现处理性能指标收集 |
| **交付物** | `_update_metrics()`, `get_metrics()`, `avg_process_time_ms` 属性 |
| **技术细节** | 追踪process_count, last_process_time, total_process_time |
| **验证标准** | 多次process后`avg_process_time_ms > 0`；`process_count`正确递增 |
| **工作量** | M |
| **依赖** | 阶段23 |

---

### 阶段30: 节点状态管理

| 属性 | 内容 |
|------|------|
| **目标** | 实现节点内部状态存取和重置 |
| **交付物** | `get_state()`, `set_state()`, `reset()` 方法 |
| **技术细节** | `_state: Dict[str, Any]`存储；reset清空state和metrics |
| **验证标准** | `set_state("key", value)`后`get_state("key") == value`；reset后所有计数归零 |
| **工作量** | S |
| **依赖** | 阶段29 |

---

## 阶段31-40: 通信协议

### 阶段31: BaseProtocol 抽象基类

| 属性 | 内容 |
|------|------|
| **目标** | 定义协议抽象接口 |
| **交付物** | `src/brain_agent/protocols/base.py` 中的 `BaseProtocol` |
| **技术细节** | 抽象方法`send()`, `receive()`；统计计数器 |
| **验证标准** | 不能直接实例化；子类必须实现抽象方法 |
| **工作量** | S |
| **依赖** | 阶段23 |

---

### 阶段32: ASYNC_PUSH 协议实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现异步推送协议（fire-and-forget） |
| **交付物** | `AsyncPushProtocol` 类 |
| **技术细节** | 使用`deque`实现FIFO队列；超时5ms；at-most-once交付 |
| **验证标准** | `send()`后立即返回；`receive()`按FIFO顺序返回；空队列返回None |
| **工作量** | M |
| **依赖** | 阶段31 |

---

### 阶段33: SYNC_CALL 协议实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现同步调用协议 |
| **交付物** | `SyncCallProtocol` 类，`call_and_collect()` 方法 |
| **技术细节** | 直接调用`target.process()`；等待所有子节点；超时20ms |
| **验证标准** | 返回所有子节点结果列表；空目标列表返回空列表 |
| **工作量** | M |
| **依赖** | 阶段31 |

---

### 阶段34: PUB_SUB 协议实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现发布订阅协议 |
| **交付物** | `PubSubProtocol` 类，`subscribe()`, `unsubscribe()`, `publish()` 方法 |
| **技术细节** | topic-based订阅；`defaultdict(set)`存储订阅者；超时100ms |
| **验证标准** | `publish()`返回订阅者数量；不同topic订阅者隔离 |
| **工作量** | M |
| **依赖** | 阶段31 |

---

### 阶段35: 协议统计接口

| 属性 | 内容 |
|------|------|
| **目标** | 实现协议性能统计 |
| **交付物** | `get_stats()` 方法（所有协议类） |
| **技术细节** | 返回message_count, error_count, avg_latency_ms |
| **验证标准** | 发送消息后stats正确更新；latency为毫秒单位 |
| **工作量** | S |
| **依赖** | 阶段32-34 |

---

### 阶段36: 协议单元测试

| 属性 | 内容 |
|------|------|
| **目标** | 为所有协议创建测试 |
| **交付物** | `tests/unit/test_protocols.py` |
| **技术细节** | 测试send/receive、统计、边界条件 |
| **验证标准** | 三种协议的所有测试通过 |
| **工作量** | M |
| **依赖** | 阶段32-35 |

---

### 阶段37: BrainNode协议集成

| 属性 | 内容 |
|------|------|
| **目标** | 在BrainNode中集成协议方法 |
| **交付物** | `push_async()`, `call_sync()`, `publish()` 方法 |
| **技术细节** | 调用process()后按协议传播到子节点；更新指标 |
| **验证标准** | `push_async()`调用后子节点收到数据；`call_sync()`返回子节点结果 |
| **工作量** | M |
| **依赖** | 阶段32-34, 阶段29 |

---

### 阶段38: 协议延迟约束验证

| 属性 | 内容 |
|------|------|
| **目标** | 实现协议级延迟检查 |
| **交付物** | 协议实现中的超时检查 |
| **技术细节** | ASYNC_PUSH: 5ms, SYNC_CALL: 20ms, PUB_SUB: 100ms |
| **验证标准** | 超时操作记录警告日志 |
| **工作量** | S |
| **依赖** | 阶段37 |

---

### 阶段39: 多协议共存测试

| 属性 | 内容 |
|------|------|
| **目标** | 验证不同协议节点可在同一树中工作 |
| **交付物** | 集成测试用例 |
| **技术细节** | 创建混合协议的小型树（3-5节点） |
| **验证标准** | 数据正确流经不同协议节点 |
| **工作量** | M |
| **依赖** | 阶段37 |

---

### 阶段40: 协议错误处理

| 属性 | 内容 |
|------|------|
| **目标** | 实现协议级错误处理策略 |
| **交付物** | 错误处理逻辑（按协议规范） |
| **技术细节** | ASYNC_PUSH: 子节点错误不影响其他；SYNC_CALL: 任一错误导致整体失败 |
| **验证标准** | 错误场景测试通过；错误计数器正确递增 |
| **工作量** | M |
| **依赖** | 阶段36, 阶段37 |

---

## 阶段41-50: 树构建

### 阶段41: DESIGN.md 解析器框架

| 属性 | 内容 |
|------|------|
| **目标** | 创建DESIGN.md文件解析器 |
| **交付物** | `src/brain_agent/tree.py` 中的 `DesignDocParser` 类 |
| **技术细节** | 静态方法`parse()`；正则表达式提取字段 |
| **验证标准** | 能解析示例DESIGN.md文件 |
| **工作量** | M |
| **依赖** | 阶段22 |

---

### 阶段42: 字段提取方法

| 属性 | 内容 |
|------|------|
| **目标** | 实现元数据字段提取 |
| **交付物** | `_extract_field()`, `_extract_section()` 方法 |
| **技术细节** | 正则匹配`- **字段名**: 值`格式；处理缺失字段 |
| **验证标准** | 正确提取节点ID、层级、region_type等字段 |
| **工作量** | S |
| **依赖** | 阶段41 |

---

### 阶段43: 接口表解析

| 属性 | 内容 |
|------|------|
| **目标** | 解析输入/输出接口表格 |
| **交付物** | `_extract_tensor_shape()`, `_extract_frequency()`, `_extract_protocol()` 方法 |
| **技术细节** | 从Markdown表格提取Tensor形状、频率、协议 |
| **验证标准** | 正确提取`Tensor[float32, 1x8x32x32]`中的形状 |
| **工作量** | M |
| **依赖** | 阶段42 |

---

### 阶段44: 名称推导逻辑

| 属性 | 内容 |
|------|------|
| **目标** | 从文件夹名推导short_name和full_name |
| **交付物** | `_derive_short_name()`, `_derive_full_name()` 方法 |
| **技术细节** | `L01_Brain--SensorimotorSystem` -> `SensorimotorSystem` |
| **验证标准** | 各种命名格式正确处理 |
| **工作量** | S |
| **依赖** | 阶段43 |

---

### 阶段45: PassThroughBrainNode 实现

| 属性 | 内容 |
|------|------|
| **目标** | 创建透传节点用于树构建测试 |
| **交付物** | `PassThroughBrainNode` 类 |
| **技术细节** | `process()`直接返回输入数据 |
| **验证标准** | 输入输出相同 |
| **工作量** | S |
| **依赖** | 阶段23 |

---

### 阶段46: BrainTreeBuilder 框架

| 属性 | 内容 |
|------|------|
| **目标** | 创建树构建器框架 |
| **交付物** | `BrainTreeBuilder` 类，`build_from_filesystem()` 方法框架 |
| **技术细节** | 遍历目录；解析所有DESIGN.md；构建NodeConfig |
| **验证标准** | 能遍历目录并收集元数据 |
| **工作量** | M |
| **依赖** | 阶段41, 阶段45 |

---

### 阶段47: 父子关系解析

| 属性 | 内容 |
|------|------|
| **目标** | 从目录层次结构解析父子关系 |
| **交付物** | 目录到parent_id的映射逻辑 |
| **技术细节** | 向上遍历目录找到最近的DESIGN.md父目录 |
| **验证标准** | 正确识别L00无父节点；L01节点的parent_id=0 |
| **工作量** | M |
| **依赖** | 阶段46 |

---

### 阶段48: 树组装与连接

| 属性 | 内容 |
|------|------|
| **目标** | 创建节点实例并建立连接 |
| **交付物** | `_assemble_tree()` 方法 |
| **技术细节** | 创建所有节点实例；调用`connect_child()`建立边 |
| **验证标准** | 构建的树中所有父子关系正确 |
| **工作量** | M |
| **依赖** | 阶段47, 阶段24 |

---

### 阶段49: BrainTree 容器类

| 属性 | 内容 |
|------|------|
| **目标** | 实现树容器和遍历方法 |
| **交付物** | `BrainTree` 类，`bfs_traverse()`, `dfs_traverse()` 方法 |
| **技术细节** | 使用`deque`实现BFS；使用栈实现DFS |
| **验证标准** | BFS按层级顺序访问；DFS按深度优先访问 |
| **工作量** | M |
| **依赖** | 阶段48 |

---

### 阶段50: 树验证与统计

| 属性 | 内容 |
|------|------|
| **目标** | 实现树结构验证和统计 |
| **交付物** | `validate()`, `get_layer_stats()` 方法 |
| **技术细节** | 验证2047节点、11层、完整二叉树；统计每层的节点类型分布 |
| **验证标准** | 完整树验证通过；统计信息准确 |
| **工作量** | M |
| **依赖** | 阶段49 |

---

## 阶段51-60: 核心节点实现 L00-L03

### 阶段51: BrainSystemNode (L00) 框架

| 属性 | 内容 |
|------|------|
| **目标** | 实现根节点框架 |
| **交付物** | `src/brain_agent/nodes/brain.py` 中的 `BrainSystemNode` |
| **技术细节** | 继承`BrainNode`；初始化全局状态（arousal, attention_focus） |
| **验证标准** | 节点可实例化；初始状态正确 |
| **工作量** | M |
| **依赖** | 阶段23, 阶段30 |

---

### 阶段52: BrainSystemNode 处理逻辑

| 属性 | 内容 |
|------|------|
| **目标** | 实现全局整合处理逻辑 |
| **交付物** | `process()` 方法，`_update_arousal()`, `_process_tensor()`, `_process_state_vector()`, `_process_signal()` |
| **技术细节** | 全局工作空间模型；arousal动态更新；注意力权重计算 |
| **验证标准** | 输入Tensor/StateVector/Signal正确处理；输出类型与输入一致 |
| **工作量** | L |
| **依赖** | 阶段51 |

---

### 阶段53: SensorimotorSystemNode (L01) 实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现感觉运动系统节点 |
| **交付物** | `src/brain_agent/nodes/sensorimotor.py` 中的 `SensorimotorSystemNode` |
| **技术细节** | 处理感觉输入和运动输出；感知-动作闭环 |
| **验证标准** | 输入输出为Tensor；处理延迟<2ms |
| **工作量** | M |
| **依赖** | 阶段23 |

---

### 阶段54: CognitiveSystemNode (L01) 实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现认知系统节点 |
| **交付物** | `src/brain_agent/nodes/cognitive.py` 中的 `CognitiveSystemNode` |
| **技术细节** | 高级认知处理；记忆和调制系统的协调 |
| **验证标准** | 正确处理StateVector输入 |
| **工作量** | M |
| **依赖** | 阶段23 |

---

### 阶段55: L02 节点实现 (4个)

| 属性 | 内容 |
|------|------|
| **目标** | 实现第二层4个节点 |
| **交付物** | `SensoryPathwayNode`, `MotorPathwayNode`, `MemorySystemNode`, `ModulationSystemNode` |
| **技术细节** | 每个节点有特定的region_type和process逻辑 |
| **验证标准** | 4个节点全部通过单元测试 |
| **工作量** | L |
| **依赖** | 阶段53, 阶段54 |

---

### 阶段56: L03 感觉通路节点 (3个)

| 属性 | 内容 |
|------|------|
| **目标** | 实现感觉通路节点 |
| **交付物** | `VisualPathwayNode`, `AuditoryPathwayNode`, `SomatosensoryPathwayNode` |
| **技术细节** | SENSORY类型；Tensor处理；感受野计算 |
| **验证标准** | 输出Tensor形状与输入一致；特征提取逻辑正确 |
| **工作量** | L |
| **依赖** | 阶段55 |

---

### 阶段57: L03 中继和记忆节点 (2个)

| 属性 | 内容 |
|------|------|
| **目标** | 实现丘脑和中继节点 |
| **交付物** | `ThalamicRelayStationNode`, `HippocampalCircuitNode` |
| **技术细节** | RELAY类型（丘脑）；MEMORY类型（海马） |
| **验证标准** | 丘脑实现信号门控；海马实现记忆编码 |
| **工作量** | M |
| **依赖** | 阶段55 |

---

### 阶段58: L03 调制系统节点 (2个)

| 属性 | 内容 |
|------|------|
| **目标** | 实现神经调制系统节点 |
| **交付物** | `AmygdalaCircuitNode`, `DopamineSystemNode`, `NoradrenalineSystemNode` |
| **技术细节** | MODULATION类型；Signal处理；inverted-U增益函数 |
| **验证标准** | Signal输入正确处理；调制增益在[0,1]范围内 |
| **工作量** | M |
| **依赖** | 阶段55 |

---

### 阶段59: NODE_CLASS_MAP 注册

| 属性 | 内容 |
|------|------|
| **目标** | 创建手动节点类映射 |
| **交付物** | `src/brain_agent/nodes/__init__.py` 中的 `NODE_CLASS_MAP` |
| **技术细节** | 映射node_id (0-14)到对应的类 |
| **验证标准** | `NODE_CLASS_MAP[0] == BrainSystemNode`；共15个条目 |
| **工作量** | S |
| **依赖** | 阶段51-58 |

---

### 阶段60: 核心节点单元测试

| 属性 | 内容 |
|------|------|
| **目标** | 为15个核心节点创建测试 |
| **交付物** | `tests/unit/test_core_nodes.py` |
| **技术细节** | 测试每个节点的process方法；验证输入输出类型 |
| **验证标准** | 15个节点全部通过测试；覆盖所有region_type |
| **工作量** | L |
| **依赖** | 阶段59 |

---

## 阶段61-70: 代码生成 L04-L10

### 阶段61: 代码生成器框架

| 属性 | 内容 |
|------|------|
| **目标** | 创建代码生成器框架 |
| **交付物** | `src/brain_agent/codegen.py` 中的 `NodeCodeGenerator` 类 |
| **技术细节** | 支持从DESIGN.md或JSON加载节点定义 |
| **验证标准** | 能加载节点定义并存储在内存中 |
| **工作量** | M |
| **依赖** | 阶段41 |

---

### 阶段62: RegionType处理模板

| 属性 | 内容 |
|------|------|
| **目标** | 创建各region_type的process模板 |
| **交付物** | `PROCESS_TEMPLATES` 字典 |
| **技术细节** | sensory/motor/memory/modulation/relay各一个模板 |
| **验证标准** | 每个模板生成有效的Python代码 |
| **工作量** | L |
| **依赖** | 阶段61 |

---

### 阶段63: 类名生成器

| 属性 | 内容 |
|------|------|
| **目标** | 从文件夹名生成有效的Python类名 |
| **交付物** | `_make_class_name()` 方法 |
| **技术细节** | `L04_Visual--OnChannelDetector` -> `OnChannelDetectorNode` |
| **验证标准** | 所有示例生成有效标识符；无重复 |
| **工作量** | S |
| **依赖** | 阶段61 |

---

### 阶段64: 节点类代码生成

| 属性 | 内容 |
|------|------|
| **目标** | 生成单个节点类源代码 |
| **交付物** | `_generate_node_class()` 方法 |
| **技术细节** | 包含类文档字符串、process方法；使用对应模板 |
| **验证标准** | 生成的代码语法正确；可导入 |
| **工作量** | M |
| **依赖** | 阶段62, 阶段63 |

---

### 阶段65: 模块头部和尾部生成

| 属性 | 内容 |
|------|------|
| **目标** | 生成模块级代码 |
| **交付物** | `_module_header()`, `_module_footer()` 方法 |
| **技术细节** | 头部：imports和文档；尾部：GENERATED_NODE_MAP |
| **验证标准** | 头部包含所有必要导入；尾部映射正确 |
| **工作量** | S |
| **依赖** | 阶段64 |

---

### 阶段66: 批量代码生成

| 属性 | 内容 |
|------|------|
| **目标** | 实现所有L04-L10节点的批量生成 |
| **交付物** | `generate_all()` 方法 |
| **技术细节** | 遍历所有layer>=4的节点；写入单个文件 |
| **验证标准** | 生成2032个节点类；文件可导入 |
| **工作量** | M |
| **依赖** | 阶段65 |

---

### 阶段67: 生成代码验证

| 属性 | 内容 |
|------|------|
| **目标** | 验证生成代码的语法正确性 |
| **交付物** | 验证脚本 `scripts/verify_generated.py` |
| **技术细节** | 使用`ast.parse()`检查语法；尝试导入 |
| **验证标准** | 无语法错误；所有类可实例化 |
| **工作量** | M |
| **依赖** | 阶段66 |

---

### 阶段68: 代码生成单元测试

| 属性 | 内容 |
|------|------|
| **目标** | 为代码生成器创建测试 |
| **交付物** | `tests/unit/test_codegen.py` |
| **技术细节** | 测试类名生成、模板渲染、完整生成流程 |
| **验证标准** | 所有生成器测试通过 |
| **工作量** | M |
| **依赖** | 阶段67 |

---

### 阶段69: JSON树数据生成

| 属性 | 内容 |
|------|------|
| **目标** | 生成完整的2047节点JSON定义 |
| **交付物** | `data/brain_tree_v5.json` |
| **技术细节** | 包含所有节点的完整元数据；支持树构建 |
| **验证标准** | JSON有效；包含2047个节点定义 |
| **工作量** | M |
| **依赖** | 阶段44 |

---

### 阶段70: 生成节点集成测试

| 属性 | 内容 |
|------|------|
| **目标** | 验证生成节点与手动节点协同工作 |
| **交付物** | 集成测试用例 |
| **技术细节** | 构建包含L00-L10的完整树；运行仿真 |
| **验证标准** | 树构建成功；仿真运行无错误 |
| **工作量** | L |
| **依赖** | 阶段68, 阶段69 |

---

## 阶段71-80: 调度器和时序

### 阶段71: FrequencySpec 数据类

| 属性 | 内容 |
|------|------|
| **目标** | 定义频率规范数据类 |
| **交付物** | `src/brain_agent/scheduler.py` 中的 `FrequencySpec` |
| **技术细节** | min_hz, max_hz, nominal_hz, dt_ms, dt_s |
| **验证标准** | 三个频段的spec值正确 |
| **工作量** | S |
| **依赖** | 无 |

---

### 阶段72: FrequencyBand 枚举

| 属性 | 内容 |
|------|------|
| **目标** | 定义频率带枚举和规格 |
| **交付物** | `FrequencyBand` 枚举，`spec` 属性 |
| **技术细节** | FAST(60-200Hz), MEDIUM(5-50Hz), SLOW(0.1-10Hz) |
| **验证标准** | `FrequencyBand.FAST.spec.dt_ms == 5.0` |
| **工作量** | S |
| **依赖** | 阶段71 |

---

### 阶段73: SchedulerConfig 数据类

| 属性 | 内容 |
|------|------|
| **目标** | 定义调度器配置 |
| **交付物** | `SchedulerConfig` dataclass |
| **技术细节** | dt_fast, dt_medium, dt_slow, max_latency_ms, max_steps, enable_metrics |
| **验证标准** | 默认值与FrequencySpec一致 |
| **工作量** | S |
| **依赖** | 阶段72 |

---

### 阶段74: StepResult 数据类

| 属性 | 内容 |
|------|------|
| **目标** | 定义单步执行结果 |
| **交付物** | `StepResult` dataclass |
| **技术细节** | step, sim_time, band, nodes_processed, total_latency_ms, max_node_latency_ms, errors |
| **验证标准** | 所有字段有正确类型注解 |
| **工作量** | S |
| **依赖** | 阶段73 |

---

### 阶段75: FrequencyScheduler 框架

| 属性 | 内容 |
|------|------|
| **目标** | 创建调度器框架 |
| **交付物** | `FrequencyScheduler` 类，`__init__`, `_classify_nodes()` |
| **技术细节** | 根据protocol将节点分类到频段；存储band_nodes映射 |
| **验证标准** | ASYNC_PUSH节点分类到FAST；SYNC_CALL到MEDIUM；PUB_SUB到SLOW |
| **工作量** | M |
| **依赖** | 阶段49, 阶段72 |

---

### 阶段76: 单步执行逻辑

| 属性 | 内容 |
|------|------|
| **目标** | 实现step()方法 |
| **交付物** | `step()` 方法 |
| **技术细节** | 每步执行FAST节点；每4步执行MEDIUM；每20步执行SLOW |
| **验证标准** | 步进计数器正确；节点执行频率符合规范 |
| **工作量** | L |
| **依赖** | 阶段75 |

---

### 阶段77: 多步运行和统计

| 属性 | 内容 |
|------|------|
| **目标** | 实现run()和统计收集 |
| **交付物** | `run()`, `get_stats()`, `reset()` 方法 |
| **技术细节** | 运行N步并收集结果；统计节点数和错误 |
| **验证标准** | `run(100)`返回100个StepResult；stats包含正确计数 |
| **工作量** | M |
| **依赖** | 阶段76 |

---

### 阶段78: 调度器错误处理

| 属性 | 内容 |
|------|------|
| **目标** | 实现节点错误隔离 |
| **交付物** | step()中的try-except错误处理 |
| **技术细节** | 单个节点错误不中断其他节点；错误记录到StepResult |
| **验证标准** | 错误节点被隔离；其他节点继续执行 |
| **工作量** | M |
| **依赖** | 阶段76 |

---

### 阶段79: 调度器单元测试

| 属性 | 内容 |
|------|------|
| **目标** | 为调度器创建全面测试 |
| **交付物** | `tests/unit/test_scheduler.py` |
| **技术细节** | 测试分类、单步、多步、频率、错误、统计 |
| **验证标准** | 所有调度器测试通过 |
| **工作量** | L |
| **依赖** | 阶段77, 阶段78 |

---

### 阶段80: 性能基准测试

| 属性 | 内容 |
|------|------|
| **目标** | 实现性能基准测试 |
| **交付物** | `tests/benchmarks/test_performance.py` |
| **技术细节** | 测量单步延迟、吞吐量、树构建时间 |
| **验证标准** | 单步<5s；吞吐量>1 step/s；构建<30s |
| **工作量** | M |
| **依赖** | 阶段79 |

---

## 阶段81-90: 监控和可观测性

### 阶段81: MetricPoint 数据类

| 属性 | 内容 |
|------|------|
| **目标** | 定义指标数据点 |
| **交付物** | `src/brain_agent/monitoring/metrics.py` 中的 `MetricPoint` |
| **技术细节** | name, value, timestamp, labels |
| **验证标准** | dataclass正确创建 |
| **工作量** | S |
| **依赖** | 无 |

---

### 阶段82: MetricsCollector 框架

| 属性 | 内容 |
|------|------|
| **目标** | 创建指标收集器 |
| **交付物** | `MetricsCollector` 类，`__init__` |
| **技术细节** | 存储指标字典；线程锁保护 |
| **验证标准** | 可实例化；线程安全 |
| **工作量** | S |
| **依赖** | 阶段81 |

---

### 阶段83: 节点指标收集

| 属性 | 内容 |
|------|------|
| **目标** | 实现节点指标收集 |
| **交付物** | `collect_node_metrics()` 方法 |
| **技术细节** | 遍历所有节点；收集latency, process_count, queue_depth |
| **验证标准** | 指标正确记录；标签包含node_id, layer, type |
| **工作量** | M |
| **依赖** | 阶段82, 阶段29 |

---

### 阶段84: 步骤结果指标收集

| 属性 | 内容 |
|------|------|
| **目标** | 从StepResult收集指标 |
| **交付物** | `collect_step_result()` 方法 |
| **技术细节** | 记录step_latency, nodes_processed, errors_total |
| **验证标准** | StepResult正确转换为指标 |
| **工作量** | S |
| **依赖** | 阶段83, 阶段74 |

---

### 阶段85: Prometheus导出格式

| 属性 | 内容 |
|------|------|
| **目标** | 实现Prometheus兼容导出 |
| **交付物** | `export_prometheus()` 方法 |
| **技术细节** | 生成Prometheus文本格式；包含标签 |
| **验证标准** | 输出格式符合Prometheus规范 |
| **工作量** | M |
| **依赖** | 阶段84 |

---

### 阶段86: TraceSpan 数据类

| 属性 | 内容 |
|------|------|
| **目标** | 定义追踪跨度 |
| **交付物** | `src/brain_agent/monitoring/tracer.py` 中的 `TraceSpan` |
| **技术细节** | node_id, node_name, layer, start_time, end_time, duration_ms, children |
| **验证标准** | `finish()`方法正确计算duration |
| **工作量** | S |
| **依赖** | 无 |

---

### 阶段87: Tracer 实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现请求追踪器 |
| **交付物** | `Tracer` 类，`start_span()`, `finish_span()` |
| **技术细节** | 维护span栈；构建父子关系树 |
| **验证标准** | 追踪数据完整记录；层级关系正确 |
| **工作量** | M |
| **依赖** | 阶段86 |

---

### 阶段88: 追踪格式化输出

| 属性 | 内容 |
|------|------|
| **目标** | 实现追踪数据格式化 |
| **交付物** | `format_trace()` 方法 |
| **技术细节** | 树形文本格式；显示层级和耗时 |
| **验证标准** | 输出可读；包含所有span |
| **工作量** | S |
| **依赖** | 阶段87 |

---

### 阶段89: 监控单元测试

| 属性 | 内容 |
|------|------|
| **目标** | 为监控组件创建测试 |
| **交付物** | `tests/unit/test_monitoring.py` |
| **技术细节** | 测试指标收集、导出、追踪 |
| **验证标准** | MetricsCollector和Tracer测试通过 |
| **工作量** | M |
| **依赖** | 阶段85, 阶段88 |

---

### 阶段90: 监控集成测试

| 属性 | 内容 |
|------|------|
| **目标** | 验证监控与调度器集成 |
| **交付物** | 集成测试用例 |
| **技术细节** | 运行仿真并收集指标和追踪 |
| **验证标准** | 指标和追踪数据完整；Prometheus格式有效 |
| **工作量** | M |
| **依赖** | 阶段89, 阶段77 |

---

## 阶段91-100: 测试、部署、文档

### 阶段91: 节点注册表实现

| 属性 | 内容 |
|------|------|
| **目标** | 创建统一节点注册表 |
| **交付物** | `src/brain_agent/registry.py` |
| **技术细节** | 合并NODE_CLASS_MAP和GENERATED_NODE_MAP；提供`create_node()` |
| **验证标准** | `get_registry_stats()`返回total=2047, manual=15, generated=2032 |
| **工作量** | M |
| **依赖** | 阶段59, 阶段66 |

---

### 阶段92: 注册表单元测试

| 属性 | 内容 |
|------|------|
| **目标** | 为注册表创建测试 |
| **交付物** | `tests/unit/test_registry.py` |
| **技术细节** | 测试节点创建、统计、错误处理 |
| **验证标准** | 所有注册表测试通过 |
| **工作量** | S |
| **依赖** | 阶段91 |

---

### 阶段93: 主入口CLI实现

| 属性 | 内容 |
|------|------|
| **目标** | 实现命令行入口 |
| **交付物** | `src/brain_agent/main.py` |
| **技术细节** | 使用`argparse`；支持--steps, --json, --log-level, --output |
| **验证标准** | `python -m brain_agent.main --help` 显示中文帮助 |
| **工作量** | M |
| **依赖** | 阶段91 |

---

### 阶段94: 端到端集成测试

| 属性 | 内容 |
|------|------|
| **目标** | 创建完整系统测试 |
| **交付物** | `tests/integration/test_end_to_end.py` |
| **技术细节** | 从JSON构建2047节点树；运行100步；验证数据流 |
| **验证标准** | 100步零错误；BFS/DFS访问全部节点 |
| **工作量** | L |
| **依赖** | 阶段93 |

---

### 阶段95: 接口规范文档 (INTERFACE_SPEC.md)

| 属性 | 内容 |
|------|------|
| **目标** | 完成接口规范文档 |
| **交付物** | `/workspace/BrainAgentSystem/INTERFACE_SPEC.md` |
| **技术细节** | 数据类型、节点接口、频率、协议、性能规范 |
| **验证标准** | 文档完整；所有映射表准确 |
| **工作量** | M |
| **依赖** | 阶段12-20 |

---

### 阶段96: 设计模板文档 (DESIGN_TEMPLATE.md)

| 属性 | 内容 |
|------|------|
| **目标** | 完成设计模板文档 |
| **交付物** | `/workspace/BrainAgentSystem/DESIGN_TEMPLATE.md` |
| **技术细节** | 标准化DESIGN.md格式；字段说明；验证清单 |
| **验证标准** | 模板可用于所有2047个节点 |
| **工作量** | M |
| **依赖** | 阶段41 |

---

### 阶段97: API文档生成

| 属性 | 内容 |
|------|------|
| **目标** | 生成API参考文档 |
| **交付物** | `docs/api.md` 或自动生成的文档 |
| **技术细节** | 使用pdoc或mkdocstring从docstring生成 |
| **验证标准** | 所有公共类和函数有文档 |
| **工作量** | M |
| **依赖** | 阶段93 |

---

### 阶段98: 部署文档

| 属性 | 内容 |
|------|------|
| **目标** | 创建部署和运维文档 |
| **交付物** | `docs/deployment.md` |
| **技术细节** | Docker部署、配置、监控、故障排除 |
| **验证标准** | 按文档可完成部署 |
| **工作量** | S |
| **依赖** | 阶段4 |

---

### 阶段99: 完整测试套件运行

| 属性 | 内容 |
|------|------|
| **目标** | 运行完整测试套件 |
| **交付物** | 测试报告 |
| **技术细节** | `pytest tests/ -v --tb=short --cov=brain_agent` |
| **验证标准** | 所有测试通过；覆盖率>80% |
| **工作量** | M |
| **依赖** | 阶段60, 阶段68, 阶段79, 阶段89, 阶段92, 阶段94 |

---

### 阶段100: 项目发布与归档

| 属性 | 内容 |
|------|------|
| **目标** | 完成项目发布准备 |
| **交付物** | 版本标签、发布说明、归档包 |
| **技术细节** | Git标签v0.1.0；CHANGELOG.md；源代码归档 |
| **验证标准** | 可重现构建；文档完整；测试通过 |
| **工作量** | S |
| **依赖** | 阶段95-99 |

---

## 附录

### A. 工作量定义

| 级别 | 描述 | 预估时间 |
|------|------|---------|
| S (Small) | 简单任务，单一文件修改 | 1-2小时 |
| M (Medium) | 中等复杂度，多文件协调 | 3-6小时 |
| L (Large) | 复杂任务，需要深入设计 | 1-2天 |

### B. 关键文件路径汇总

| 类别 | 文件路径 |
|------|---------|
| 配置 | `/workspace/BrainAgentSystem/pyproject.toml` |
| 类型定义 | `/workspace/BrainAgentSystem/src/brain_agent/types.py` |
| 节点基类 | `/workspace/BrainAgentSystem/src/brain_agent/node.py` |
| 协议 | `/workspace/BrainAgentSystem/src/brain_agent/protocols/` |
| 树构建 | `/workspace/BrainAgentSystem/src/brain_agent/tree.py` |
| 核心节点 | `/workspace/BrainAgentSystem/src/brain_agent/nodes/` |
| 代码生成 | `/workspace/BrainAgentSystem/src/brain_agent/codegen.py` |
| 调度器 | `/workspace/BrainAgentSystem/src/brain_agent/scheduler.py` |
| 监控 | `/workspace/BrainAgentSystem/src/brain_agent/monitoring/` |
| 注册表 | `/workspace/BrainAgentSystem/src/brain_agent/registry.py` |
| 主入口 | `/workspace/BrainAgentSystem/src/brain_agent/main.py` |
| 测试 | `/workspace/BrainAgentSystem/tests/` |
| 文档 | `/workspace/BrainAgentSystem/docs/` |

### C. 节点ID分配表

| 层级 | 节点ID范围 | 节点数 | 说明 |
|------|-----------|-------|------|
| L00 | 0 | 1 | 根节点 |
| L01 | 1-2 | 2 | 感觉运动、认知 |
| L02 | 3-6 | 4 | 感觉通路、运动通路、记忆、调制 |
| L03 | 7-14 | 8 | 视觉、听觉、体感、丘脑、海马、杏仁核、多巴胺、去甲肾上腺素 |
| L04-L10 | 15-2046 | 2032 | 代码生成 |

### D. 验收标准检查清单

- [ ] 2047个节点全部可实例化
- [ ] 树结构验证通过（11层完整二叉树）
- [ ] 三种通信协议正常工作
- [ ] 多速率调度器按频率执行
- [ ] 所有单元测试通过
- [ ] 集成测试通过（100步零错误）
- [ ] 性能基准达标（单步<5s）
- [ ] Prometheus指标导出正常
- [ ] Docker构建成功
- [ ] 文档完整

---

*本文档由BrainAgentSystem项目团队维护*
