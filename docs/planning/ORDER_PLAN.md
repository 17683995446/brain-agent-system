# BrainAgentSystem 秩序化多阶段迭代方案

## 当前状态
- 测试: 677 passed
- mypy: 0 errors
- flake8: 0 issues
- 重复代码: ~64k行（生成代码双份）
- 孤儿模块: transport.py, protocols/

## 第一性原理
1. **单一真相源**: 每个概念只有一个定义位置
2. **无死代码**: 所有代码必须被运行时使用
3. **一致模式**: 相同职责的代码遵循相同模式
4. **清晰依赖**: 模块依赖单向无环，层次分明
5. **封装完整**: 不跨模块访问私有属性

## 阶段计划

### Stage 1: 消除重复生成代码
- 删除 generated_nodes.py（64k行）或 layer_04..10.py（64k行）之一
- 保留单一生成代码源
- 更新 generated/__init__.py 导入

### Stage 2: 统一版本号与工具配置
- 版本号单一来源: pyproject.toml
- __init__.py 从 pyproject.toml 读取版本
- 合并 mypy.ini, .flake8, .black.toml 到 pyproject.toml
- 修正 pyarrow 依赖分类

### Stage 3: 清理孤儿模块
- transport.py: 移除或标记为实验性
- protocols/: 移除或接入运行时
- 清理未使用依赖

### Stage 4: 提取 process() 模板方法
- 在 BrainNode 基类实现 process() 模板方法
- 消除15个手写节点的分派样板
- 子类只需实现 _process_tensor/_process_signal 等

### Stage 5: 消除重复函数与硬编码路径
- 合并 _find_json_path() 为单一实现
- 移除硬编码绝对路径
- 修复 main.py 死对象 (Tracer())

### Stage 6: 修复封装泄露
- api.py 不访问 scheduler._step/_sim_time
- main.py 不访问 scheduler._medium_ratio/_slow_ratio
- 添加公共属性/方法替代

### Stage 7: 统一节点实现模式
- codegen.py 生成带 __init__/计时/完整分派的节点
- 消除手写/生成节点模式差异
- 移除 PassThroughBrainNode 错位定义

### Stage 8: 整理文档与数据源
- 整合散落的规划文档到 docs/
- 明确 DESIGN.md ↔ JSON 关系
- 更新 README 项目结构

## 验证规则
每执行一个新阶段前，验证前五个阶段:
1. 所有测试通过
2. mypy 0 errors
3. flake8 0 issues
