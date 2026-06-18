# 丘脑中继站·Efferent·Ipsilateral·Crossed·Ascending·NonSpecGate

## 基本信息
- **节点ID**: 368
- **文件夹名**: `L08_Ascending--NonspecificGateController`
- **层级**: 第8层 (L08)
- **父模块**: 丘脑中继站·Efferent·Ipsilateral·Crossed·Ascending
- **子节点数**: 2
- **节点类型**: 非叶节点（功能模块）
- **脑区类型**: relay

## 职责功能
非特异性门控器，广泛扩散信号

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | Tensor[float32, 1x32x32] | 50-100Hz | Ascending | ASYNC_PUSH |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | Tensor[float32, 1x32x32] | 50-100Hz | 子模块 | ASYNC_PUSH |

## 处理逻辑
- **算法**: 信号中继 + 门控开关 + 拓扑映射保留
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L07_Crossed--AscendingRelay`
- **下游**: 子模块 (2个)
