# 去甲肾上腺素系统·Inhibitory·Suppressor·FastAdapt·PhasicUnit·Serotonin

## 基本信息
- **节点ID**: 504
- **文件夹名**: `L08_PhasicUnit--SerotoninReleaser`
- **层级**: 第8层 (L08)
- **父模块**: 去甲肾上腺素系统·Inhibitory·Suppressor·FastAdapt·PhasicUnit
- **子节点数**: 2
- **节点类型**: 非叶节点（功能模块）
- **脑区类型**: modulation

## 职责功能
血清素释放器，调控情绪睡眠

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | Signal[float32] | 1-10Hz | PhasicUnit | PUB_SUB |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | Signal[float32] | 1-10Hz | 子模块 | PUB_SUB |

## 处理逻辑
- **算法**: 神经递质释放 + 增益调节 + 稳态反馈
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L07_FastAdapt--PhasicReleaseUnit`
- **下游**: 子模块 (2个)
