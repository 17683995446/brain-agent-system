# 体感通路·OnChannel·LowFreq·CenterRF·FastPath·MagnoCell·OrientSel·Luminance

## 基本信息
- **节点ID**: 1312
- **文件夹名**: `L10_OrientSel--LuminanceEncoder`
- **层级**: 第10层 (L10)
- **父模块**: 体感通路·OnChannel·LowFreq·CenterRF·FastPath·MagnoCell·OrientSel
- **子节点数**: 0
- **节点类型**: 叶节点（功能单元）
- **脑区类型**: sensory

## 职责功能
亮度编码器，提取明暗对比

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | Tensor[float32, 32x32x8] | 60-100Hz | OrientSel | ASYNC_PUSH |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | Tensor[float32, 32x32x8] | 60-100Hz | 子模块 | ASYNC_PUSH |

## 处理逻辑
- **算法**: 感受野计算 + 特征检测 + 信号滤波
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L09_MagnoCell--OrientationSelector`
