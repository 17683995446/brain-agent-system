# BrainAgentSystem API 文档

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
```python
from brain_agent.types import Signal

result = node.process(Signal(value=1.0, timestamp=0.0))
```

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

## 调度器

### 运行仿真
```python
from brain_agent.scheduler import FrequencyScheduler

scheduler = FrequencyScheduler(tree)
results = scheduler.run(steps=100)
```

## 监控

### 收集指标
```python
from brain_agent.monitoring import MetricsCollector

metrics = MetricsCollector(tree)
metrics.collect_node_metrics()
print(metrics.export_prometheus())
```
