# 去甲肾上腺素系统·Inhibitory·Suppressor·FastAdapt·TonicUnit

## 基本信息
- **节点ID**: 252
- **文件夹名**: `L07_FastAdapt--TonicReleaseUnit`
- **层级**: 第7层 (L07)
- **父模块**: 去甲肾上腺素系统·Inhibitory·Suppressor·FastAdapt
- **子节点数**: 2
- **节点类型**: 非叶节点（功能模块）
- **脑区类型**: modulation

## 职责功能
紧张型释放单元，持续稳定释放

## 接口规范
### 输入接口
| 端口 | 数据类型 | 频率 | 来源 | 协议 |
|------|----------|------|------|------|
| IN | Signal[float32] | 1-10Hz | FastAdapt | PUB_SUB |

### 输出接口
| 端口 | 数据类型 | 频率 | 目标 | 协议 |
|------|----------|------|------|------|
| OUT | Signal[float32] | 1-10Hz | 子模块 | PUB_SUB |

## 处理逻辑
- **算法**: 神经递质释放 + 增益调节 + 稳态反馈
- **延迟约束**: < 10ms

## 信号流
- **上游**: `L06_Suppressor--FastAdaptor`
- **下游**: 子模块 (2个)
