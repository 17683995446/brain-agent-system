# 丘脑中继站·Efferent·Ipsilateral·Crossed·Descending·SpecGate·GateOpen·HighFreqR

## 基本信息
- **节点ID**: 1479
- **文件夹名**: `L10_GateOpen--HighFreqRelay`
- **层级**: 第10层 (L10)
- **父模块**: 丘脑中继站·Efferent·Ipsilateral·Crossed·Descending·SpecGate·GateOpen
- **子节点数**: 0
- **节点类型**: 叶节点（功能单元）
- **脑区类型**: relay

## 职责功能
高频中继器，快速传递精确信号

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | Tensor[float32, 1x32x32] | 50-100Hz | GateOpen | ASYNC_PUSH |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | Tensor[float32, 1x32x32] | 50-100Hz | 子模块 | ASYNC_PUSH |

## 处理逻辑
- **算法**: 信号中继 + 门控开关 + 拓扑映射保留
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L09_SpecGate--GateOpener`
