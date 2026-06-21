# BrainAgentSystem 用户易用性提升方案

## 第一性原理
1. **即时反馈**: 用户应始终知道系统在做什么
2. **错误可操作**: 错误信息必须告诉用户如何修复
3. **最少惊讶**: API行为应符合直觉
4. **渐进式复杂度**: 简单任务简单做，复杂任务可达
5. **可发现性**: 功能应通过类型提示、文档、示例可被发现

## 阶段计划

### Stage 1: 仿真进度反馈
- Simulation.run() 增加 on_step 回调
- 增加 iter_steps() 生成器方法
- 集成 tqdm 进度条
- HTTP /progress 端点

### Stage 2: 错误信息可操作化
- 所有 "Unsupported X" 错误列出合法选项
- 所有 "Unknown X" 错误列出合法值
- NotImplementedError 解释模板方法模式
- 统一错误语言为英文
- 错误包含文件路径

### Stage 3: 修复配置系统
- 修复 env var 覆盖 bug（下划线键名）
- 增加 BrainConfig.to_yaml() 回写
- 增加 Literal 类型提示
- 配置 schema 文档

### Stage 4: 上下文管理器与 API 语义修复
- Simulation 增加 __enter__/__exit__
- ResultExporter 增加 __enter__/__exit__
- 修复 export() 自动 close 矛盾
- get_node() 增加严格版本

### Stage 5: 测试辅助工具
- 创建 brain_agent.testing 模块
- make_node_config(), MockNode, build_small_tree()
- 修复 conftest.py 硬编码路径
- 小规模测试树 fixture

### Stage 6: 自定义节点示例与教程
- examples/06_custom_node.py
- examples/07_experiment_sweep.py
- 修复现有 examples 使用私有属性
- README 自定义节点教程

### Stage 7: CLI 改进
- pyproject.toml 增加 console_scripts
- argparse 子命令 (run/validate/info/export)
- --version, --verbose, --quiet
- 改进 help 文本

### Stage 8: 文档全面更新
- docs/api.md 增加 Simulation 类文档
- 修复 docs/types.md 引用不存在的 transport
- README FAQ/常见坑
- CHANGELOG.md

## 验证规则
每执行新阶段前验证前五个阶段:
1. 所有测试通过
2. mypy 0 errors
3. flake8 0 issues
