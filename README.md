# Brain Agent System

脑模拟智能体系统 —— 基于完整二叉树拓扑结构的大规模节点协作框架。

## 概述

本项目构建一个包含 **2047 个节点** 的脑模拟系统，节点按 11 层完整二叉树组织（2^11 - 1 = 2047）。每个节点模拟大脑中不同区域的信号处理单元，通过多频率时间步进（快 5ms / 中 20ms / 慢 100ms）实现从高频感知到低频决策的全频段覆盖。

## 项目结构

```
BrainAgentSystem/
├── pyproject.toml              # 项目配置与依赖管理
├── configs/
│   └── default.yaml            # 默认仿真参数配置
├── src/brain_agent/
│   ├── __init__.py             # 包入口，定义版本号
│   ├── types.py                # 核心数据类型定义
│   ├── node.py                 # 节点基类
│   ├── protocols/              # 节点间通信协议
│   ├── nodes/                  # 具体节点实现
│   ├── generated/              # 自动生成的节点实现
│   └── monitoring/             # 监控与可观测性
└── tests/
    ├── conftest.py             # 测试公共配置
    └── ...
```

## 快速开始

```bash
# 安装项目（开发模式）
pip install -e ".[dev]"

# 运行测试
pytest
```

## 配置说明

系统默认配置位于 `configs/default.yaml`，包含：

- **brain.total_nodes**: 节点总数，默认 2047（2^11 - 1）
- **brain.max_layers**: 树的最大层数，默认 11
- **simulation.dt_fast**: 快速时间步长 5ms，对应 100-200Hz 频段
- **simulation.dt_medium**: 中速时间步长 20ms，对应 10-50Hz 频段
- **simulation.dt_slow**: 慢速时间步长 100ms，对应 1-10Hz 频段
- **monitoring.enabled**: 是否启用监控指标采集
- **monitoring.metrics_port**: 监控指标服务端口

## 许可证

MIT License
