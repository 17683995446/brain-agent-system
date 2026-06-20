# BrainAgentSystem 实用性提升多阶段迭代方案

## 当前状态
- 测试: 591 passed
- mypy: 0 errors
- flake8: 0 issues
- 实用性评分: 4/10

## 目标
将实用性评分从 4/10 提升到 8/10，使系统可作为 Python 库和 CLI 工具用于真实脑仿真实验。

## 阶段计划

### Phase 1: 公共 API 与包接口
- `__init__.py` 导出核心类（BrainTree, FrequencyScheduler, BrainNode, 类型系统）
- 创建 `api.py` 高层 API（单行创建/运行/导出）
- 确保 `import brain_agent` 即可用

### Phase 2: 数据输入管道
- 创建 `io/input.py` 输入适配器（文件/数组/流/回调）
- 支持 JSON、NPZ、CSV、内存数组输入
- 调度器接入输入源，每步可喂入不同数据

### Phase 3: 结果导出系统
- 创建 `io/output.py` 结果导出器
- 支持导出：节点状态快照、输出数据、Trace、指标
- 格式：JSON、CSV、NPZ（NumPy原生）
- CLI 新增 `--export-format` `--export-path` 参数

### Phase 4: 配置系统统一
- BrainConfig → SchedulerConfig 自动映射
- dt_fast/dt_medium/dt_slow 从配置读取（不再硬编码）
- CLI 新增 `--config` 参数支持 YAML 配置文件
- 所有运行时参数可配置

### Phase 5: 协议语义实现
- ASYNC_PUSH: 异步入队，不等待（当前行为，保留）
- SYNC_CALL: 同步调用，等待所有子节点处理完成
- PUB_SUB: 主题订阅广播
- 调度器根据协议类型执行不同通信行为

### Phase 6: 示例与文档
- 创建 `examples/` 目录
- 示例1: 基础仿真（5行代码运行）
- 示例2: 自定义输入数据
- 示例3: 结果导出与分析
- 示例4: 自定义节点
- 更新 README.md 为完整使用指南

### Phase 7: 性能优化
- 性能分析：识别热点函数
- 向量化：用 NumPy 批量操作替代循环
- 缓存：预计算常量、复用缓冲区
- 目标：单步 < 1s（当前约 5s）

### Phase 8: 仿真实验 API
- 创建 `experiment.py` 批量实验框架
- 支持参数扫描、多次运行、结果聚合
- 支持检查点保存/恢复
- 支持确定性复现（随机种子）

## 验证规则
每执行一个新阶段前，验证前五个阶段的正确性：
1. 所有测试通过
2. mypy 0 errors
3. flake8 0 issues
4. 新增功能有测试覆盖
