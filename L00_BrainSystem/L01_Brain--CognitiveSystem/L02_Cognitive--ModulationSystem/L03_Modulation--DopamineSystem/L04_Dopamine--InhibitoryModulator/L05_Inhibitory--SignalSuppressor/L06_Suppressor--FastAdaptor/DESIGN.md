# 多巴胺系统·Inhibitory·Suppressor·FastAdapt

## 基本信息
- **节点ID**: 117
- **文件夹名**: `L06_Suppressor--FastAdaptor`
- **层级**: 第6层 (L06)
- **父模块**: 多巴胺系统·Inhibitory·Suppressor
- **子节点数**: 2
- **节点类型**: 非叶节点（功能模块）
- **脑区类型**: modulation

## 职责功能
快适应调节器，快速响应变化

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | Signal[float32] | 1-10Hz | Suppressor | PUB_SUB |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | Signal[float32] | 1-10Hz | 子模块 | PUB_SUB |

## 处理逻辑
- **算法**: 神经递质释放 + 增益调节 + 稳态反馈
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L05_Inhibitory--SignalSuppressor`
- **下游**: 子模块 (2个)
