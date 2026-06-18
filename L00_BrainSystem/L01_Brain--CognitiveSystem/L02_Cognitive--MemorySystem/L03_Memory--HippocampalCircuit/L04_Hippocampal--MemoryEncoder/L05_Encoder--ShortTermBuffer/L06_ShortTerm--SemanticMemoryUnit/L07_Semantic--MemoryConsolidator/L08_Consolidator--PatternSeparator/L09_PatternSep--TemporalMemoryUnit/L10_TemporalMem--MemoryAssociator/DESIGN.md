# 海马记忆回路·Encoder·ShortTerm·Semantic·Consolidator·PatternSep·TemporalMem·Associator

## 基本信息
- **节点ID**: 1553
- **文件夹名**: `L10_TemporalMem--MemoryAssociator`
- **层级**: 第10层 (L10)
- **父模块**: 海马记忆回路·Encoder·ShortTerm·Semantic·Consolidator·PatternSep·TemporalMem
- **子节点数**: 0
- **节点类型**: 叶节点（功能单元）
- **脑区类型**: memory

## 职责功能
记忆关联器，建立信息间联系

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | StateVector[float32, 128] | 5-10Hz | TemporalMem | SYNC_CALL |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | StateVector[float32, 128] | 5-10Hz | 子模块 | SYNC_CALL |

## 处理逻辑
- **算法**: 突触可塑性 + 模式分离/补全 + 时序编码
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L09_PatternSep--TemporalMemoryUnit`
