# BrainAgentSystem API 文档

> 本文档描述 BrainAgentSystem 的公开 API，包括高层 `Simulation` 接口、实验框架、I/O 模块、配置系统以及测试工具。
>
> 版本: v0.5.0
> 最后更新: 2026-06-20

---

## 目录

1. [Simulation 类（高层 API）](#simulation-类高层-api)
2. [实验 API](#实验-api)
3. [I/O 模块](#io-模块)
4. [配置](#配置)
5. [测试工具](#测试工具)
6. [核心类型](#核心类型)
7. [节点操作](#节点操作)
8. [树操作](#树操作)
9. [调度器](#调度器)
10. [监控](#监控)

---

## Simulation 类（高层 API）

`Simulation` 是最常用的高层入口，将 `BrainTree`、`FrequencyScheduler` 和 `MetricsCollector` 封装为一个对象，提供简洁的运行/导出/统计接口。

```python
from brain_agent import Simulation
```

### Simulation.from_json()

工厂方法，从脑树 JSON 文件创建 `Simulation` 实例。

```python
@classmethod
def from_json(
    cls,
    json_path: Optional[str] = None,
    config: Optional[SchedulerConfig] = None,
) -> "Simulation"
```

- `json_path`：脑树 JSON 路径。为 `None` 时自动搜索标准位置（详见 [路径发现](#路径发现)）。
- `config`：可选的 `SchedulerConfig`。为 `None` 时使用默认调度参数。

```python
# 自动发现 brain_tree_v5.json
sim = Simulation.from_json()

# 指定路径与配置
from brain_agent import SchedulerConfig
config = SchedulerConfig(dt_fast=0.002, dt_medium=0.01, dt_slow=0.05)
sim = Simulation.from_json("data/brain_tree_v5.json", config=config)
```

### Simulation.run()

运行若干步仿真，返回每步的 `StepResult` 列表。

```python
def run(
    self,
    steps: int = 100,
    input_data: Optional[BrainData] = None,
    input_callback: Optional[Callable[[int], BrainData]] = None,
    shutdown_checker: Optional[Callable[[], bool]] = None,
    on_step: Optional[StepCallback] = None,
    progress: bool = False,
) -> List[StepResult]
```

- `steps`：运行步数。
- `input_data`：每步喂给根节点的静态输入。
- `input_callback`：每步以步号为参数调用，返回该步输入；会覆盖 `input_data`。
- `shutdown_checker`：可选回调，返回 `True` 时优雅停止。
- `on_step`：每步完成后以 `(step_index, step_result)` 调用，用于进度反馈、日志或副作用式提前终止。
- `progress`：为 `True` 时向 stderr 输出 tqdm 进度条（需安装 `tqdm`）；若已提供 `on_step` 则忽略此项。

```python
# 静态输入
results = sim.run(steps=100, input_data=Signal(value=1.0))

# 动态输入
def my_input(step):
    return Signal(value=float(step) * 0.01)
results = sim.run(steps=100, input_callback=my_input)

# 进度回调
def on_step(i, result):
    print(f"Step {i}: {result.nodes_processed} nodes, "
          f"{result.total_latency_ms:.1f}ms")
results = sim.run(steps=100, on_step=on_step)

# 进度条（需要 tqdm）
results = sim.run(steps=1000, progress=True)
```

### Simulation.iter_steps()

生成器 API，逐步产出 `StepResult`，提供对仿真循环的细粒度控制。

```python
def iter_steps(
    self,
    steps: int = 100,
    input_data: Optional[BrainData] = None,
    input_callback: Optional[Callable[[int], BrainData]] = None,
    shutdown_checker: Optional[Callable[[], bool]] = None,
    progress: bool = False,
) -> Iterator[StepResult]
```

```python
# 逐个处理结果
for result in sim.iter_steps(steps=100, progress=True):
    if result.errors:
        print(f"Step {result.step} had errors: {result.errors}")
    if result.total_latency_ms > 100:
        print(f"Slow step: {result.total_latency_ms:.1f}ms")

# 提前终止
for result in sim.iter_steps(steps=1000):
    if result.total_latency_ms > 500:
        break  # 提前停止
```

> `Simulation.run()` 内部即调用 `iter_steps()`，二者共享同一套输入/停止逻辑。

### Simulation.export()

导出仿真结果到文件或内存。

```python
def export(
    self,
    path: Optional[str] = None,
    format: Optional[str] = None,
    include_metrics: bool = True,
    include_node_states: bool = False,
) -> "ResultExporter"
```

- `path`：输出文件路径。为 `None` 时导出到内存。
- `format`：输出格式（`"json"`、`"csv"`、`"npz"`）。为 `None` 时从 `path` 后缀自动推断。
- `include_metrics`：是否包含 Prometheus 指标。
- `include_node_states`：是否包含各节点状态快照。

导出到文件时，文件写入后 exporter 自动关闭；导出到内存时，exporter 保持打开，可通过 `.data` 访问结果。

```python
# 文件导出（自动关闭）
sim.run(steps=100)
sim.export("results.json")
sim.export("results.csv")
sim.export("results.npz")
sim.export("full_results.json", include_node_states=True)

# 内存导出（访问 .data）
exporter = sim.export()
data = exporter.data
```

### Simulation.summary()

返回人类可读的仿真统计摘要字符串。

```python
sim.run(steps=100)
print(sim.summary())
```

输出示例：

```
Brain Simulation Summary
========================================
  Total nodes:    2047
  Steps run:      100
  Sim time:       0.500s
  Fast band:      1024 nodes
  Medium band:    512 nodes
  Slow band:      511 nodes
  Avg latency:    3.21ms
  Max latency:    8.74ms
  Total errors:   0
```

### Simulation.get_node() / get_node_strict()

按 ID 访问节点。

```python
def get_node(self, node_id: int) -> Optional[BrainNode]
def get_node_strict(self, node_id: int) -> BrainNode
```

- `get_node()`：找不到时返回 `None`。
- `get_node_strict()`：找不到时抛出 `KeyError`，错误信息中列出有效 ID 范围。

```python
node = sim.get_node(0)            # 返回 BrainNode 或 None
root = sim.get_node_strict(0)     # 返回 BrainNode，不存在则 KeyError
```

### 上下文管理器

`Simulation` 支持上下文管理器协议：进入时返回自身，退出时调用 `reset()` 重置仿真状态。

```python
with Simulation.from_json() as sim:
    results = sim.run(steps=100)
    print(sim.summary())
# 退出 with 块后自动 reset()
```

### 其他方法与属性

| 成员 | 说明 |
|------|------|
| `sim.step(input_data=None)` | 执行单步，返回 `StepResult` |
| `sim.reset()` | 重置调度器、指标、追踪器与结果 |
| `sim.export_metrics(format="prometheus")` | 导出指标字符串（目前支持 `"prometheus"`） |
| `sim.stats` | 当前仿真统计字典 |
| `sim.results` | 本轮所有 `StepResult` 列表 |
| `sim.step_count` | 当前步号 |
| `sim.sim_time` | 当前仿真时间（秒） |
| `sim.tree` | 脑节点树 |
| `sim.scheduler` | 频率调度器 |
| `sim.metrics` | 指标收集器 |
| `sim.tracer` | 执行追踪器 |

### 模块级便捷函数

```python
from brain_agent import create_simulation, run_simulation

# 创建实例
sim = create_simulation(json_path=None, config=None)

# 一次性运行
results = run_simulation(json_path=None, steps=100, input_data=None, config=None)
```

---

## 实验 API

`brain_agent.experiment` 模块提供批量实验与参数扫描框架，支持随机种子复现与检查点。

```python
from brain_agent import Experiment, ExperimentConfig, ParameterSweep
```

### ExperimentConfig

单个实验的配置数据类。

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | `"experiment"` | 实验名（用于输出文件） |
| `steps` | `int` | `100` | 仿真步数 |
| `seed` | `Optional[int]` | `None` | 随机种子（`None` 表示非确定性） |
| `dt_fast` | `float` | `0.005` | 快频段时间步长（秒） |
| `dt_medium` | `float` | `0.02` | 中频段时间步长（秒） |
| `dt_slow` | `float` | `0.1` | 慢频段时间步长（秒） |
| `json_path` | `Optional[str]` | `None` | 脑树 JSON 路径（`None` 自动发现） |
| `output_dir` | `Optional[str]` | `None` | 输出目录 |
| `tags` | `List[str]` | `[]` | 分类标签 |

### Experiment

带可复现性的单次仿真实验，封装 `Simulation` 并附加实验元数据、检查点与种子控制。

构造函数接受 `ExperimentConfig`、名称字符串或 `None`（配合 `**kwargs` 覆盖字段）：

```python
exp = Experiment("baseline", steps=100, seed=42)
# 等价于
exp = Experiment(ExperimentConfig(name="baseline", steps=100, seed=42))
```

主要方法：

| 方法 | 说明 |
|------|------|
| `run(steps=None, input_callback=None)` | 运行实验，返回 `List[StepResult]`。设置相同 `seed` 可复现结果 |
| `checkpoint(path=None)` | 保存检查点 JSON 以便后续恢复，返回文件路径 |
| `export(path=None, format="json")` | 导出结果（`"json"`/`"csv"`/`"npz"`），返回文件路径 |
| `summary()` | 返回人类可读摘要字符串 |

属性：`results`（结果列表）、`metadata`（实验元数据字典）。

```python
exp = Experiment("baseline", steps=100, seed=42)
results = exp.run()
print(exp.summary())
exp.export("results.json")

# 复现
exp2 = Experiment("baseline", steps=100, seed=42)
results2 = exp2.run()
```

### ParameterSweep

跨参数网格运行实验，每个参数组合运行一次完整实验并聚合指标。

```python
ParameterSweep(name="sweep", steps=100, json_path=None, seed=42)
```

| 方法 | 说明 |
|------|------|
| `add_param(name, values)` | 添加一个扫描参数（`name` 须匹配 `ExperimentConfig` 字段），返回 `self` 支持链式调用 |
| `run()` | 运行所有参数组合，返回结果字典列表 |
| `best(metric="avg_latency_ms", minimize=True)` | 按指标找最优组合 |
| `export(path=None)` | 导出扫描结果到 JSON，返回文件路径 |
| `summary()` | 返回摘要字符串（含最优/最差组合） |

属性：`results`（所有组合的结果列表）。

```python
sweep = ParameterSweep("dt_sweep", steps=50)
sweep.add_param("dt_fast", [0.001, 0.002, 0.005])
sweep.add_param("dt_medium", [0.01, 0.02])
results = sweep.run()

best = sweep.best(metric="avg_latency_ms", minimize=True)
print(f"Best: {best}")
sweep.export("sweep_results.json")
print(sweep.summary())
```

---

## I/O 模块

`brain_agent.io` 模块提供统一的输入适配器与结果导出器。

```python
from brain_agent.io import create_input, create_exporter
from brain_agent.io import ListInput, CallbackInput, ArrayInput, FileInput, ResultExporter
```

### create_input() 工厂

根据源类型自动选择合适的输入适配器：

| 源类型 | 适配器 |
|--------|--------|
| `str` / `Path` | `FileInput`（从文件读取） |
| `List[BrainData]` | `ListInput` |
| `Callable[[int], BrainData]` | `CallbackInput` |
| `List[np.ndarray]` | `ArrayInput` |

```python
from brain_agent import Signal, create_input

# 从文件
inp = create_input("data.json")

# 从列表
inp = create_input([Signal(value=1.0, timestamp=0.0)])

# 从回调
inp = create_input(lambda step: Signal(value=step, timestamp=step * 0.005))
```

返回的适配器本身可调用，可直接作为 `Simulation.run(input_callback=...)` 的参数。

### InputAdapter 类型

所有输入适配器继承 `InputAdapter`，实现 `__call__(step) -> BrainData`，并提供 `length` 属性与 `close()` 方法。

#### ListInput

由预构建的 `BrainData` 列表支撑。列表耗尽后重复最后一项。

```python
from brain_agent.io import ListInput

data = [Signal(value=float(i), timestamp=i * 0.005) for i in range(100)]
inp = ListInput(data)
```

#### CallbackInput

由 Python 可调用对象支撑，回调接收步号并返回 `BrainData`，适合过程化或实时输入生成。

```python
from brain_agent.io import CallbackInput

def generate(step):
    return Signal(value=np.sin(step * 0.1), timestamp=step * 0.005)

inp = CallbackInput(generate)
```

#### ArrayInput

由 numpy 数组列表支撑。多维数组转为 `Tensor`，标量转为 `Signal`。

```python
from brain_agent.io import ArrayInput

arrays = [np.random.randn(1, 8, 8).astype(np.float32) for _ in range(100)]
inp = ArrayInput(arrays, dt=0.005)
```

#### FileInput

从文件读取，支持格式：

- **JSON**：对象数组，每项含 `type` 与数据字段。
- **NPZ**：NumPy 压缩归档，键 `step_0`、`step_1`、… 为各步数组。
- **CSV**：表格数据，每行一步。`value` 列转为 `Signal`，数值列打包为 `Tensor`，`timestamp` 列设置时间戳。

```python
from brain_agent.io import FileInput

inp = FileInput("input_data.json")
inp = FileInput("signals.npz", data_type="signal")
```

### create_exporter() 工厂

创建结果导出器。`path` 为 `None` 时使用内存存储。

```python
from brain_agent.io import create_exporter

exporter = create_exporter("results.json")   # 文件导出
exporter = create_exporter(format="json")     # 内存导出
```

### ResultExporter

高层结果导出器，支持 JSON / CSV / NPZ 三种格式，并支持上下文管理器。

| 方法 | 说明 |
|------|------|
| `export_results(results)` | 导出仿真步结果 |
| `export_metrics(metrics)` | 导出已收集指标 |
| `export_trace(tracer)` | 导出执行追踪 |
| `export_node_states(tree)` | 导出所有节点当前状态，返回 `{node_id: state_dict}` |
| `save()` | 将数据写入输出文件 |
| `close()` | 刷新并关闭输出 |
| `summary()` | 返回导出数据统计字典 |

属性：`data`（内存导出时的数据列表）。

```python
# 上下文管理器用法
with create_exporter("results.json") as exporter:
    exporter.export_results(results)
    exporter.export_metrics(sim.metrics)
# 退出时自动 close()
```

---

## 配置

BrainAgentSystem 支持分层配置：YAML 文件 -> 环境变量覆盖 -> 代码内配置。

### BrainConfig

```python
from brain_agent import BrainConfig
```

| 方法 | 说明 |
|------|------|
| `BrainConfig(config_dict=None)` | 从字典构造，自动应用环境变量覆盖 |
| `BrainConfig.from_yaml(path)` | 从 YAML 文件加载 |
| `BrainConfig.from_defaults()` | 加载内置默认配置（`configs/default.yaml`） |
| `to_yaml()` | 序列化为 YAML 字符串 |
| `save(path)` | 保存到 YAML 文件 |
| `to_dict()` | 返回配置字典副本 |
| `get(key, default=None)` | 按点分键取值（如 `"simulation.dt_fast"`） |
| `get_int` / `get_float` / `get_bool` | 类型化取值 |
| `validate()` | 校验配置，返回错误消息列表（空列表表示有效） |
| `to_scheduler_config()` | 转换为运行时 `SchedulerConfig` |

```python
# 从 YAML
config = BrainConfig.from_yaml("my_config.yaml")

# 修改后保存
config.save("my_config.yaml")

# 转为调度器配置
sched_config = config.to_scheduler_config()
sim = Simulation.from_json(config=sched_config)
```

### 环境变量覆盖

环境变量使用 `BRAIN_` 前缀，格式为 `BRAIN_SECTION_KEY`。第一个下划线分隔顶层 section，其余部分作为该 section 内的键（保留键中的下划线）。

例如：

| 环境变量 | 映射到配置键 |
|----------|--------------|
| `BRAIN_SIMULATION_DT_FAST` | `simulation.dt_fast` |
| `BRAIN_SIMULATION_DT_MEDIUM` | `simulation.dt_medium` |
| `BRAIN_SIMULATION_DT_SLOW` | `simulation.dt_slow` |
| `BRAIN_MONITORING_METRICS_PORT` | `monitoring.metrics_port` |
| `BRAIN_LOGGING_LEVEL` | `logging.level` |

```bash
BRAIN_SIMULATION_DT_FAST=0.002 brain-agent run --steps 100
```

> 注意：键中的下划线会被保留。`BRAIN_SIMULATION_DT_FAST` 正确映射到 `simulation.dt_fast`，而不是 `simulation.dt.fast`。

### SchedulerConfig 直接使用

无需 YAML 时，可直接构造 `SchedulerConfig` 传给 `Simulation`。

```python
from brain_agent import SchedulerConfig, Simulation

config = SchedulerConfig(
    dt_fast=0.002,
    dt_medium=0.01,
    dt_slow=0.05,
    max_latency_ms=10.0,
    max_steps=0,        # 0 = 无限
    enable_metrics=True,
)
sim = Simulation.from_json(config=config)
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dt_fast` | `float` | `0.005` | 快循环步长（100Hz 标称） |
| `dt_medium` | `float` | `0.02` | 中循环步长（20Hz 标称） |
| `dt_slow` | `float` | `0.1` | 慢循环步长（1Hz 标称） |
| `max_latency_ms` | `float` | `10.0` | 单节点最大延迟 |
| `max_steps` | `int` | `0` | 最大步数（0 = 无限） |
| `enable_metrics` | `bool` | `True` | 是否启用指标收集 |

### 路径发现

`find_brain_tree_json()` 按以下顺序查找 `brain_tree_v5.json`：

1. 显式传入的路径参数
2. `BRAIN_JSON_PATH` 环境变量
3. 相对于包安装位置的 `data/brain_tree_v5.json`
4. 当前工作目录下的 `data/brain_tree_v5.json` 或 `brain_tree_v5.json`

找不到时抛出 `FileNotFoundError`，提示设置 `BRAIN_JSON_PATH` 或显式传参。

---

## 测试工具

`brain_agent.testing` 模块提供编写自定义节点与仿真场景测试的辅助工具。

```python
from brain_agent.testing import (
    make_node_config, MockNode, RecordingNode,
    build_small_tree,
    make_test_signal, make_test_tensor, make_test_state_vector,
)
```

### make_node_config()

创建用于测试的 `NodeConfig`，参数均为字符串友好形式（自动映射到枚举）。

```python
config = make_node_config(
    node_id=0,
    short_name="TestNode",
    region_type="sensory",   # sensory/motor/memory/modulation/relay/cognitive
    protocol="async_push",   # async_push/sync_call/pub_sub
    layer=0,
    parent_id=None,
)
```

### MockNode

用于测试的 mock 节点。记录所有输入，`process()` 默认原样返回（pass-through），或应用可配置的 `transform`。

属性：`processed_inputs`（收到的所有 `BrainData` 列表）、`process_count`。方法：`reset_mock()` 清空记录。

```python
node = MockNode(config, transform=lambda d: d)
result = node.process(Signal(value=1.0))
assert node.process_count == 1
```

### RecordingNode

记录所有处理数据的节点。与 `MockNode` 不同，它应用真实的处理函数（默认恒等），并将 `(input, output)` 元组存入 `records`。

```python
node = RecordingNode(config, process_fn=lambda d: d)
node.process(Signal(value=1.0))
assert len(node.records) == 1
```

### build_small_tree()

构建小型脑树用于测试。`n_layers=4` 生成 2^4 - 1 = 15 个节点，远快于完整 2047 节点树。

```python
tree = build_small_tree(n_layers=4)   # 15 节点
tree = build_small_tree(n_layers=3)   # 7 节点
for node in tree.bfs_traverse():
    print(node.node_id, node.short_name)
```

`n_layers` 取值范围 1-11。

### 测试数据工厂

| 函数 | 说明 |
|------|------|
| `make_test_signal(value=1.0, timestamp=0.0)` | 创建测试 `Signal` |
| `make_test_tensor(shape=(1,1,4,4), fill_value=1.0)` | 创建测试 `Tensor` |
| `make_test_state_vector(dimension=16, fill_value=0.5)` | 创建测试 `StateVector` |

```python
sig = make_test_signal(value=0.8)
tensor = make_test_tensor(shape=(1, 3, 8, 8))
sv = make_test_state_vector(dimension=128)
```

---

## 核心类型

### Tensor

```python
from brain_agent.types import Tensor
import numpy as np

t = Tensor(data=np.zeros((1, 8, 32, 32), dtype=np.float32), shape=(1, 8, 32, 32))
```

### StateVector

```python
from brain_agent.types import StateVector

sv = StateVector(data=np.zeros(128, dtype=np.float32), dimension=128)
sv_norm = sv.normalize()
```

### Signal

```python
from brain_agent.types import Signal

s = Signal(value=0.5, timestamp=0.0)
```

> 完整类型说明见 [types.md](types.md)。

---

## 节点操作

### 创建节点

```python
from brain_agent.node import BrainNode, NodeConfig, RegionType, Protocol
from brain_agent.registry import create_node

config = NodeConfig(
    node_id=0,
    name_cn="大脑",
    short_name="Brain",
    folder_name="L00_BrainSystem",
    layer=0,
    parent_id=None,
    responsibility="全局整合",
    region_type=RegionType.UNKNOWN,
    tensor_shape="Composite[All]",
    frequency="0.01-200Hz",
    protocol=Protocol.PUB_SUB,
    algorithm="全局整合",
)
node = create_node(config)
```

### 处理数据

`process()` 是模板方法：记录计时、按输入类型分发到对应的 `_process_*` 钩子、更新指标并返回结果。子类通过覆盖钩子自定义行为：

- `_process_tensor(data: Tensor)`
- `_process_state_vector(data: StateVector)`
- `_process_signal(data: Signal)`
- `_process_message(data: Message)`（默认 pass-through）

```python
from brain_agent.types import Signal

result = node.process(Signal(value=1.0, timestamp=0.0))
```

> 自定义节点实现详见 [examples/06_custom_node.py](../examples/06_custom_node.py)。

---

## 树操作

### 构建树

```python
from brain_agent.tree import BrainTreeBuilder

builder = BrainTreeBuilder()
tree = builder.build_full_tree("/path/to/brain_tree_v5.json")
```

### 遍历树

```python
# BFS
for node in tree.bfs_traverse():
    print(node.short_name)

# DFS
for node in tree.dfs_traverse():
    print(node.short_name)
```

---

## 调度器

### 运行仿真

```python
from brain_agent.scheduler import FrequencyScheduler

scheduler = FrequencyScheduler(tree)
results = scheduler.run(steps=100)
```

多频段调度器按三个频段驱动仿真：

- **Fast band**：每步执行（`dt_fast` 间隔）
- **Medium band**：每 `dt_medium/dt_fast` 步执行
- **Slow band**：每 `dt_slow/dt_fast` 步执行

---

## 监控

### 收集指标

```python
from brain_agent.monitoring import MetricsCollector

metrics = MetricsCollector(tree)
metrics.collect_node_metrics()
print(metrics.export_prometheus())
```
