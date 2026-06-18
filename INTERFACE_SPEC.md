# BrainAgentSystem 接口规范 v1.0

## 1. 数据类型规范

### 1.1 Tensor
- **布局**: NCHW (batch, channels, height, width)
- **dtype**: float32
- **标准形状**:
  - 视觉通路: (1, 8, 32, 32)  -- 8个特征通道，32x32空间分辨率
  - 听觉通路: (1, 16, 32, 32) -- 16个频率通道
  - 运动通路: (1, 1, 32, 32)  -- 肌肉激活图
  - 中继通路: (1, 1, 32, 32)  -- 信号中继

### 1.2 StateVector
- **维度**: 128 (默认) | 256 (海马) | 512 (PFC)
- **dtype**: float32
- **归一化**: L2归一化 (||v|| = 1.0)

### 1.3 Signal
- **类型**: float32 标量
- **范围**: [-1.0, 1.0] (标准化)
- **时间戳**: float64 秒

### 1.4 SpikeTrain
- **时间戳**: float64, 单位秒
- **神经元ID**: int64
- **持续时间**: float64, 单位秒

## 2. 节点接口规范

### 2.1 输入端口 (IN)
| 属性 | 说明 |
|------|------|
| port_id | "IN" |
| name | "输入信号" |
| data_type | 根据region_type确定 |
| shape | 根据region_type确定 |
| frequency | 节点指定频率 |
| protocol | 节点指定协议 |
| required | true |

### 2.2 输出端口 (OUT)
| 属性 | 说明 |
|------|------|
| port_id | "OUT" |
| name | "输出信号" |
| data_type | 与输入相同 |
| shape | 与输入相同 (类型保持) |
| frequency | 与输入相同 |
| protocol | 与输入相同 |

### 2.3 RegionType -> 数据类型映射
| RegionType | 输入数据类型 | 标准形状 |
|-----------|-------------|---------|
| SENSORY | Tensor | (1, 8, 32, 32) |
| MOTOR | Tensor | (1, 1, 32, 32) |
| MEMORY | StateVector | (128,) |
| MODULATION | Signal | scalar |
| RELAY | Tensor | (1, 1, 32, 32) |
| UNKNOWN | Composite | any |

## 3. 频率规范

### 3.1 频率带定义
| 频段 | 最小频率 | 最大频率 | 标称频率 | 时间步长 |
|------|---------|---------|---------|---------|
| FAST | 60 Hz | 200 Hz | 100 Hz | 5 ms |
| MEDIUM | 5 Hz | 50 Hz | 20 Hz | 20 ms |
| SLOW | 0.1 Hz | 10 Hz | 1 Hz | 100 ms |

### 3.2 RegionType -> 频段映射
| RegionType | 频段 | 协议 |
|-----------|------|------|
| SENSORY | FAST | ASYNC_PUSH |
| MOTOR | MEDIUM | SYNC_CALL |
| MEMORY | MEDIUM | SYNC_CALL |
| MODULATION | SLOW | PUB_SUB |
| RELAY | FAST | ASYNC_PUSH |
| UNKNOWN | SLOW | PUB_SUB |

## 4. 协议规范

### 4.1 ASYNC_PUSH
- **语义**: 异步推送，fire-and-forget
- **交付保证**: at-most-once
- **超时**: 5ms
- **重试**: 无
- **顺序保证**: 有序
- **错误处理**: 子节点错误不影响其他节点

### 4.2 SYNC_CALL
- **语义**: 同步调用，等待所有子节点完成
- **交付保证**: exactly-once
- **超时**: 20ms
- **重试**: 无
- **顺序保证**: 有序
- **错误处理**: 任一子节点错误导致整体失败

### 4.3 PUB_SUB
- **语义**: 发布订阅，广播到所有订阅者
- **交付保证**: at-least-once
- **超时**: 100ms
- **重试**: 无
- **顺序保证**: 无序
- **错误处理**: 订阅者错误不影响发布者

## 5. 性能规范

### 5.1 延迟约束
| 层级 | 最大延迟 |
|------|---------|
| L00-L03 | < 2 ms |
| L04-L07 | < 3 ms |
| L08-L10 | < 5 ms |
| 端到端 (根->叶) | < 50 ms |

### 5.2 内存约束
| 节点类型 | 最大内存 |
|---------|---------|
| 叶节点 | < 1 MB |
| 非叶节点 | < 500 KB |
| 整棵树 | < 2 GB |
