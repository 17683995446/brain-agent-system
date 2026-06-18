# 体感通路·OnChannel·HighFreq·CenterRF·SlowPath

## 基本信息
- **节点ID**: 160
- **文件夹名**: `L07_CenterRF--SlowPathway`
- **层级**: 第7层 (L07)
- **父模块**: 体感通路·OnChannel·HighFreq·CenterRF
- **子节点数**: 2
- **节点类型**: 非叶节点（功能模块）
- **脑区类型**: sensory

## 职责功能
慢速传导通路，处理稳态持续信号

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | Tensor[float32, 32x32x8] | 60-100Hz | CenterRF | ASYNC_PUSH |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | Tensor[float32, 32x32x8] | 60-100Hz | 子模块 | ASYNC_PUSH |

## 处理逻辑
- **算法**: 感受野计算 + 特征检测 + 信号滤波
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L06_HighFreq--CenterReceptor`
- **下游**: 子模块 (2个)
