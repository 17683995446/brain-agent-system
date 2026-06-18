# 去甲肾上腺素系统·Excitatory·Enhancer·SlowAdapt·PhasicUnit·Dopamine·Acetylcho·RewardSig

## 基本信息
- **节点ID**: 1937
- **文件夹名**: `L10_Acetylcho--RewardSignalGenerator`
- **层级**: 第10层 (L10)
- **父模块**: 去甲肾上腺素系统·Excitatory·Enhancer·SlowAdapt·PhasicUnit·Dopamine·Acetylcho
- **子节点数**: 0
- **节点类型**: 叶节点（功能单元）
- **脑区类型**: modulation

## 职责功能
奖赏信号发生器，增强有益行为

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
