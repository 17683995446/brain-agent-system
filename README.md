# Brain Agent System

脑模拟智能体系统 —— 基于完整二叉树拓扑结构的大规模节点协作框架。

## 概述

本项目构建一个包含 **2047 个节点** 的脑模拟系统，节点按 11 层完整二叉树组织（2^11 - 1 = 2047）。每个节点模拟大脑中不同区域的信号处理单元，通过多频率时间步进（快 5ms / 中 20ms / 慢 100ms）实现从高频感知到低频决策的全频段覆盖。

## 快速开始

### 安装

```bash
# 安装项目（开发模式）
pip install -e ".[dev]"
```

### Python API（5行代码运行仿真）

```python
from brain_agent import Simulation

sim = Simulation.from_json()       # 加载2047节点脑树
results = sim.run(steps=100)       # 运行100步
print(sim.summary())               # 打印统计摘要
sim.export("results.json")         # 导出结果
```

### 命令行运行

```bash
# 基本运行
python -m brain_agent.main --steps 100

# 使用自定义配置
python -m brain_agent.main --config configs/default.yaml --steps 100

# 导出结果到JSON
python -m brain_agent.main --steps 100 --export results.json

# 导出指标到文件
python -m brain_agent.main --steps 100 --output metrics.txt

# 禁用HTTP服务
python -m brain_agent.main --steps 100 --no-server
```

### 自定义输入数据

```python
from brain_agent import Simulation, Signal, create_input

sim = Simulation.from_json()

# 方法1: 列表输入
data = [Signal(value=float(i), timestamp=i*0.005) for i in range(100)]
inp = create_input(data)
results = sim.run(steps=100, input_callback=inp)

# 方法2: 回调函数（正弦波）
inp = create_input(lambda step: Signal(
    value=__import__('numpy').sin(step * 0.1),
    timestamp=step * 0.005,
))
results = sim.run(steps=100, input_callback=inp)

# 方法3: 从文件读取
inp = create_input("input_data.json")
results = sim.run(steps=100, input_callback=inp)
```

### 结果导出

```python
# 导出到JSON
sim.export("results.json")

# 导出到CSV
sim.export("results.csv")

# 导出到NPZ（NumPy格式）
sim.export("results.npz")

# 包含节点状态
sim.export("full_results.json", include_node_states=True)
```

## 示例

```bash
# 基础仿真
python examples/01_basic_simulation.py

# 自定义输入
python examples/02_custom_input.py

# 结果导出与分析
python examples/03_export_results.py

# 自定义配置
python examples/04_custom_config.py

# 文件I/O
python examples/05_file_io.py
```

## 配置说明

系统默认配置位于 `configs/default.yaml`，支持 YAML 文件、环境变量覆盖、代码内配置三种方式：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `simulation.dt_fast` | 0.005s | 快循环步长（100-200Hz感知通路） |
| `simulation.dt_medium` | 0.02s | 中循环步长（10-50Hz运动/记忆） |
| `simulation.dt_slow` | 0.1s | 慢循环步长（1-10Hz调制系统） |
| `simulation.max_latency_ms` | 10.0 | 单节点最大延迟 |
| `monitoring.metrics_port` | 9090 | 监控指标服务端口 |
| `logging.level` | INFO | 日志级别 |

```python
# 使用自定义配置
from brain_agent import BrainConfig, SchedulerConfig, Simulation

# 方法1: YAML文件
config = BrainConfig.from_yaml("my_config.yaml")
sched_config = config.to_scheduler_config()
sim = Simulation.from_json(config=sched_config)

# 方法2: 代码内配置
config = SchedulerConfig(dt_fast=0.002, dt_medium=0.01, dt_slow=0.05)
sim = Simulation.from_json(config=config)
```

环境变量覆盖（`BRAIN_` 前缀）：
```bash
BRAIN_SIMULATION_DT_FAST=0.002 python -m brain_agent.main --steps 100
```

## 架构

### 数据类型

| 类型 | 用途 | 示例 |
|------|------|------|
| `Tensor` | 多维张量（视觉/听觉/体感） | shape=(1,8,32,32) |
| `StateVector` | 状态向量（记忆系统） | dimension=128 |
| `Signal` | 标量信号（神经调制） | value=0.5 |
| `SpikeTrain` | 脉冲序列 | timestamps=[0.1,0.2] |
| `Message` | 消息（控制信号） | topic="attention" |

### 通信协议

| 协议 | 频段 | 语义 |
|------|------|------|
| `ASYNC_PUSH` | 快(60-200Hz) | 异步推入队列，子节点按自身频率处理 |
| `SYNC_CALL` | 中(5-50Hz) | 同步调用，父节点处理完后立即处理子节点 |
| `PUB_SUB` | 慢(0.1-10Hz) | 广播发布，所有订阅者接收 |

### 调度器

多频段调度器按三个频段驱动仿真：
- **Fast band**: 每步执行（dt_fast间隔）
- **Medium band**: 每 `dt_medium/dt_fast` 步执行
- **Slow band**: 每 `dt_slow/dt_fast` 步执行

## 项目结构

```
BrainAgentSystem/
├── src/brain_agent/
│   ├── __init__.py             # 公共API导出
│   ├── api.py                  # 高层API（Simulation类）
│   ├── types.py                # 核心数据类型
│   ├── node.py                 # 节点基类（process()模板方法）
│   ├── tree.py                 # 脑树构建器
│   ├── scheduler.py            # 多频段调度器
│   ├── config.py               # 配置管理（BrainConfig→SchedulerConfig）
│   ├── paths.py                # 路径发现（单一JSON搜索逻辑）
│   ├── main.py                 # CLI入口
│   ├── server.py               # HTTP健康检查服务
│   ├── lifecycle.py            # 优雅关闭
│   ├── codegen.py              # 节点代码生成器（构建期工具）
│   ├── experiment.py           # 批量实验框架
│   ├── io/                     # 数据I/O
│   │   ├── input.py            # 输入适配器（JSON/NPZ/CSV/数组/回调）
│   │   └── output.py           # 结果导出器（JSON/CSV/NPZ）
│   ├── nodes/                  # 手写核心节点（15个脑区）
│   ├── generated/              # 自动生成的节点（2032个，单一文件）
│   └── monitoring/             # 监控与可观测性
├── examples/                   # 使用示例（5个）
├── configs/                    # 配置文件
├── docs/                       # 文档
│   ├── api.md                  # API参考
│   ├── architecture.md         # 架构说明
│   ├── types.md                # 类型系统
│   ├── INTERFACE_SPEC.md       # 接口规范
│   ├── DESIGN_TEMPLATE.md      # 设计模板
│   └── planning/               # 规划文档
├── tests/                      # 测试套件
└── data/                       # 脑树定义JSON
```

## 开发

```bash
# 运行测试
pytest

# 类型检查
mypy src/brain_agent/

# 代码风格检查
flake8 src/brain_agent/ --max-line-length=120
```

## 许可证

MIT License
