# BrainAgentSystem 项目计划

## 概述

BrainAgentSystem 是一个2047节点的大脑仿真Agent系统，采用11层二叉树结构模拟大脑神经系统的层级组织。本计划详细描述各阶段目标、子步骤和验收标准。

---

## 阶段 1: 基础设施搭建

### 1.1 项目结构初始化
- **子步骤**:
  1. 创建目录结构 (src/brain_agent/, tests/, data/)
  2. 配置 Python 包管理 (setup.py, pyproject.toml)
  3. 初始化 Git 仓库和 .gitignore
  4. 配置 pytest 和代码质量工具
- **验收标准**:
  - `python -m pytest` 可以运行（即使无测试也成功）
  - `pip install -e .` 成功安装包
  - 目录结构符合规范

### 1.2 核心数据类型定义
- **子步骤**:
  1. 定义 BrainDataType 基类
  2. 实现 Tensor (NCHW float32)
  3. 实现 StateVector (L2归一化, 128/256/512维)
  4. 实现 Signal (float32 标量, [-1,1])
  5. 实现 SpikeTrain (float64时间戳, int64神经元ID)
  6. 实现 Message (控制信号)
- **验收标准**:
  - 所有类型通过 test_types.py 测试
  - 构造时自动类型转换和形状验证
  - 不可变性 (frozen dataclass) 生效

---

## 阶段 2: 节点基类与配置

### 2.1 BrainNode 抽象基类
- **子步骤**:
  1. 定义 RegionType 枚举 (SENSORY/MOTOR/MEMORY/MODULATION/RELAY/UNKNOWN)
  2. 定义 Protocol 枚举 (ASYNC_PUSH/SYNC_CALL/PUB_SUB)
  3. 定义 PortSpec 和 NodeConfig dataclass
  4. 实现 BrainNode ABC，含 process() 抽象方法
  5. 实现父子节点连接管理
  6. 实现处理指标追踪 (process_count, avg_process_time_ms)
- **验收标准**:
  - test_node.py 中 TestNodeCreation 全部通过
  - 抽象类不可直接实例化
  - 端口配置根据 region_type 自动映射

### 2.2 端口验证系统
- **子步骤**:
  1. 实现 _parse_tensor_shape() 支持多种格式
  2. 实现 _validate_input() 数据类型检查
  3. 实现 _validate_output() 数据类型检查
  4. 实现 Tensor 形状验证
  5. 定义 REGION_TYPE_DATA_MAP 和 REGION_TYPE_SHAPE_MAP
- **验收标准**:
  - 类型不匹配时抛出 ValueError
  - 形状不匹配时抛出 ValueError
  - 空/无效 shape_str 安全处理为 None

---

## 阶段 3: 通信协议

### 3.1 ASYNC_PUSH 协议
- **子步骤**:
  1. 实现 BaseProtocol 抽象类
  2. 实现 AsyncPushProtocol (fire-and-forget)
  3. 实现消息队列 (FIFO)
  4. 实现 pending_count 和 stats
- **验收标准**:
  - test_protocols.py 中 TestAsyncPushProtocol 全部通过
  - 多发送者到同一目标正常工作
  - 空队列接收返回 None

### 3.2 SYNC_CALL 协议
- **子步骤**:
  1. 实现 SyncCallProtocol (同步调用)
  2. 实现 call_and_collect() 收集子节点结果
  3. 实现 get_last_results() 缓存
- **验收标准**:
  - test_protocols.py 中 TestSyncCallProtocol 全部通过
  - 空目标列表返回空列表
  - 直接调用 process() 不更新节点指标

### 3.3 PUB_SUB 协议
- **子步骤**:
  1. 实现 PubSubProtocol (发布订阅)
  2. 实现 subscribe/unsubscribe/get_subscribers
  3. 实现 topic 隔离
- **验收标准**:
  - test_protocols.py 中 TestPubSubProtocol 全部通过
  - 不同 topic 订阅者独立
  - 发布到空 topic 返回 count=0

---

## 阶段 4: 二叉树构建器

### 4.1 DESIGN.md 解析器
- **子步骤**:
  1. 实现 DesignDocParser.parse()
  2. 实现 _extract_field, _extract_section
  3. 实现 _parse_layer, _parse_region_type
  4. 实现 _derive_short_name, _derive_full_name
  5. 从文件系统提取 tensor_shape, frequency, protocol
- **验收标准**:
  - test_tree.py 中 TestDesignDocParser 全部通过
  - 根节点 (L00) 正确解析
  - 叶节点 (L10) 正确解析

### 4.2 树构建与验证
- **子步骤**:
  1. 实现 BrainTreeBuilder.build_from_filesystem()
  2. 实现 BrainTreeBuilder.build_from_json()
  3. 实现 BrainTree 容器 (BFS/DFS 遍历)
  4. 实现 get_path_to_root, get_nodes_by_layer
  5. 实现 validate() 检查2047节点和完整二叉树
- **验收标准**:
  - 小树的 BFS/DFS 顺序正确
  - 完整2047节点树构建成功
  - validate() 通过所有检查

---

## 阶段 5: 核心节点实现 (L00-L03)

### 5.1 15个手动节点
- **子步骤**:
  1. BrainSystemNode (L00, 全局整合)
  2. SensorimotorSystemNode / CognitiveSystemNode (L01)
  3. SensoryPathwayNode / MotorPathwayNode / MemorySystemNode / ModulationSystemNode (L02)
  4. Visual/Auditory/Somatosensory/Thalamic/Hippocampal/Amygdala/Dopamine/Noradrenaline (L03)
  5. 每个节点实现特定 process() 逻辑
- **验收标准**:
  - test_core_nodes.py 全部通过
  - 每个节点返回正确的数据类型
  - NODE_CLASS_MAP 包含15个条目

### 5.2 代码生成 (L04-L10)
- **子步骤**:
  1. 实现 codegen.py 模板引擎
  2. 为2032个节点生成 PassThrough 变体
  3. 生成 GENERATED_NODE_MAP
  4. 集成到 registry.py
- **验收标准**:
  - get_registry_stats() 返回 total=2047
  - manual=15, generated=2032

---

## 阶段 6: 多速率调度器

### 6.1 频率调度器
- **子步骤**:
  1. 定义 FrequencyBand 枚举 (FAST/MEDIUM/SLOW)
  2. 定义 FrequencySpec dataclass (min/max/nominal Hz, dt)
  3. 实现 FrequencyScheduler._classify_nodes()
  4. 实现 step() 多速率执行
  5. 实现 run(), get_stats(), reset()
- **验收标准**:
  - test_scheduler.py 全部通过
  - MEDIUM 每4步执行一次
  - SLOW 每20步执行一次
  - 错误不中断其他节点处理

### 6.2 性能基准
- **子步骤**:
  1. 单步延迟 < 5s (2047节点)
  2. 吞吐量 > 1 step/s
  3. 树构建时间 < 30s
- **验收标准**:
  - test_integration.py 中 TestPerformanceBenchmarks 通过

---

## 阶段 7: 监控与追踪

### 7.1 指标收集
- **子步骤**:
  1. 实现 MetricsCollector
  2. 实现 collect_node_metrics()
  3. 实现 export_prometheus()
- **验收标准**:
  - test_monitoring.py 通过
  - Prometheus 格式输出非空

### 7.2 追踪系统
- **子步骤**:
  1. 实现 Tracer.start_span/finish_span
  2. 实现 get_traces(), get_latest_trace()
- **验收标准**:
  - 追踪数据完整记录

---

## 阶段 8: 端到端集成

### 8.1 完整系统测试
- **子步骤**:
  1. 从 JSON 构建完整2047节点树
  2. 运行调度器100步无错误
  3. 验证数据从根节点流到所有1024叶节点
  4. 验证所有 region_type 都有节点
  5. 验证层级分布正确
- **验收标准**:
  - test_integration.py 全部通过
  - 50步运行零错误
  - BFS/DFS 访问全部2047节点

### 8.2 主入口
- **子步骤**:
  1. 实现 main.py CLI
  2. 支持 --steps, --log-level 参数
- **验收标准**:
  - test_main.py 通过
  - --help 显示中文帮助

---

## 阶段 9: 文档与规范

### 9.1 接口规范
- **子步骤**:
  1. 创建 INTERFACE_SPEC.md
  2. 定义数据类型规范 (Tensor/StateVector/Signal/SpikeTrain)
  3. 定义节点接口规范 (IN/OUT端口)
  4. 定义频率规范 (FAST/MEDIUM/SLOW)
  5. 定义协议规范 (ASYNC_PUSH/SYNC_CALL/PUB_SUB)
  6. 定义性能规范 (延迟/内存约束)
- **验收标准**:
  - INTERFACE_SPEC.md 包含所有章节
  - RegionType -> data_type 映射完整
  - 频率带 min/max/nominal Hz 和 dt 精确

### 9.2 设计模板
- **子步骤**:
  1. 创建 DESIGN_TEMPLATE.md
  2. 标准化 DESIGN.md 格式
- **验收标准**:
  - 模板可被所有2047个节点使用

---

## 验收标准汇总

| 测试文件 | 测试类数 | 关键验证点 |
|---------|---------|-----------|
| test_types.py | 6 | 数据类型构造、验证、不可变性 |
| test_node.py | 10 | 节点创建、process、连接、指标、重置 |
| test_protocols.py | 5 | 三种协议的发送/接收/统计 |
| test_tree.py | 10 | 解析、构建、遍历、验证 |
| test_scheduler.py | 10 | 分类、单步/多步、频率、错误、统计 |
| test_core_nodes.py | 16 | 15个核心节点 + NODE_CLASS_MAP |
| test_integration.py | 6 | 端到端、性能、监控、注册表 |
| test_main.py | 2 | CLI 帮助和运行 |
| test_monitoring.py | - | 指标和追踪 |
| test_codegen.py | - | 代码生成 |
| test_registry.py | - | 注册表统计 |

**总体目标**: `pytest tests/ -q` 全部通过，零失败。
