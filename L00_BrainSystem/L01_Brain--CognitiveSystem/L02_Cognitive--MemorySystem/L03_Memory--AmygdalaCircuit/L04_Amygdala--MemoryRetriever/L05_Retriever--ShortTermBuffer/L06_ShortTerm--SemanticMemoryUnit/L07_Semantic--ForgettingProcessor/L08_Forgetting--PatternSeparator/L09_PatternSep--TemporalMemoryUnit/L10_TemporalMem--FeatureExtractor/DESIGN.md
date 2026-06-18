# 杏仁核情绪回路·Retriever·ShortTerm·Semantic·Forgetting·PatternSep·TemporalMem·Extractor

## 基本信息
- **节点ID**: 1754
- **文件夹名**: `L10_TemporalMem--FeatureExtractor`
- **层级**: 第10层 (L10)
- **父模块**: 杏仁核情绪回路·Retriever·ShortTerm·Semantic·Forgetting·PatternSep·TemporalMem
- **子节点数**: 0
- **节点类型**: 叶节点（功能单元）
- **脑区类型**: memory

## 职责功能
特征提取器，抽取关键信息

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
