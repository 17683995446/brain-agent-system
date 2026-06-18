# 海马记忆回路·Encoder·LongTerm·Semantic·Forgetting

## 基本信息
- **节点ID**: 198
- **文件夹名**: `L07_Semantic--ForgettingProcessor`
- **层级**: 第7层 (L07)
- **父模块**: 海马记忆回路·Encoder·LongTerm·Semantic
- **子节点数**: 2
- **节点类型**: 非叶节点（功能模块）
- **脑区类型**: memory

## 职责功能
遗忘处理器，清除冗余信息

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | StateVector[float32, 128] | 5-10Hz | Semantic | SYNC_CALL |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | StateVector[float32, 128] | 5-10Hz | 子模块 | SYNC_CALL |

## 处理逻辑
- **算法**: 突触可塑性 + 模式分离/补全 + 时序编码
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L06_LongTerm--SemanticMemoryUnit`
- **下游**: 子模块 (2个)
