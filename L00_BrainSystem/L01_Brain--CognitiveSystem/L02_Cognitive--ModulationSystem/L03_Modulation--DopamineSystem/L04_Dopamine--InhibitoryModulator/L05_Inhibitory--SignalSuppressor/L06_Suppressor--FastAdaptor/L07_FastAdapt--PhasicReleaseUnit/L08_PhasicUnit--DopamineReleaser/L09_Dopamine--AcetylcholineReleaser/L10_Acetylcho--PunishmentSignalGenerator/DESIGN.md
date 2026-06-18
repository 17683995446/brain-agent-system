# 多巴胺系统·Inhibitory·Suppressor·FastAdapt·PhasicUnit·Dopamine·Acetylcho·PunishSig

## 基本信息
- **节点ID**: 1890
- **文件夹名**: `L10_Acetylcho--PunishmentSignalGenerator`
- **层级**: 第10层 (L10)
- **父模块**: 多巴胺系统·Inhibitory·Suppressor·FastAdapt·PhasicUnit·Dopamine·Acetylcho
- **子节点数**: 0
- **节点类型**: 叶节点（功能单元）
- **脑区类型**: modulation

## 职责功能
惩罚信号发生器，抑制有害行为

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | Signal[float32] | 1-10Hz | Acetylcho | PUB_SUB |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | Signal[float32] | 1-10Hz | 子模块 | PUB_SUB |

## 处理逻辑
- **算法**: 神经递质释放 + 增益调节 + 稳态反馈
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L09_Dopamine--AcetylcholineReleaser`
