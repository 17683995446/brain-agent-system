# 杏仁核情绪回路·Encoder·LongTerm·Episodic·Consolidator·PatternComp·SpatialMem

## 基本信息
- **节点ID**: 849
- **文件夹名**: `L09_PatternComp--SpatialMemoryUnit`
- **层级**: 第9层 (L09)
- **父模块**: 杏仁核情绪回路·Encoder·LongTerm·Episodic·Consolidator·PatternComp
- **子节点数**: 2
- **节点类型**: 非叶节点（功能模块）
- **脑区类型**: memory

## 职责功能
空间记忆单元，处理位置导航

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | StateVector[float32, 128] | 5-10Hz | PatternComp | SYNC_CALL |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | StateVector[float32, 128] | 5-10Hz | 子模块 | SYNC_CALL |

## 处理逻辑
- **算法**: 突触可塑性 + 模式分离/补全 + 时序编码
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L08_Consolidator--PatternCompleter`
- **下游**: 子模块 (2个)
