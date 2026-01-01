# 当前实现状态总结

## ✅ 已完成

### 1. 架构参数 (`arch/memory.py`)
- `read_bandwidth_limit`: 读带宽限制 (bytes/cycle)
- `write_bandwidth_limit`: 写带宽限制
- `num_read_ports`, `num_write_ports`: 端口数

### 2. 带宽模型 (`model/bandwidth.py`)
- `build_2d_bandwidth_constraints()`: 2D PE 阵列带宽约束
- `add_spatial_hw_variables()`: H/W 方向空间映射变量
- `get_reduction_axes()`: 识别规约轴 (R, S, C)
- `build_reduction_constraints()`: 规约约束
- 分析函数: `compute_dataflow_analysis()`, `compute_reduction_cost()`

### 3. 配置文件 (`examples/configs/arch.yaml`)
```yaml
GlobalBuffer:
  read_bandwidth: 256
  write_bandwidth: 64
  num_read_ports: 4
```

## ⏳ 待完成

### 1. 计算单元模型
需要在 `arch/pim_arch.py` 添加:
```python
@dataclass
class ComputeUnit:
    type: str  # "scalar", "systolic", "tensor_core"
    size: tuple  # (H, W) for 2D, or (N,) for 1D
    reduction_tree_depth: int = 0  # 0 = no tree
    reduction_latency: float = 1.0  # cycles per tree stage
```

### 2. 与优化器集成
在 `optimizer.py` 中调用:
```python
# 添加 H/W 空间变量
add_spatial_hw_variables(model, vars, arch, workloads, pe_h, pe_w)

# 添加带宽约束
build_2d_bandwidth_constraints(model, vars, arch, workloads, pe_h, pe_w)

# 添加规约约束
build_reduction_constraints(model, vars, arch, workloads, pe_h, pe_w)
```

### 3. 规约代价加入目标函数
```python
# 在目标函数中添加
obj += reduction_latency * weight_latency
```

## 📋 关键公式速查

### 带宽需求 (数据类型 t)
```
BW[t] = ∏_{j: O[j][t]=1} spatial[j]
```
只有**相关维度**的空间并行度会影响带宽。

### 广播规则
```
H 方向并行 → W 方向广播 (带宽 = H)
W 方向并行 → H 方向广播 (带宽 = W)
```

### 规约条件
```
如果 spatial[R] > 1 或 spatial[S] > 1 或 spatial[C] > 1:
    需要规约 Output 的部分和
```

### 规约延迟
```
latency = log₂(reduction_parallelism) × base_latency  # 如果有硬件树
latency = 2 × reduction_parallelism                   # 如果用 buffer
```

## 🔗 Interstellar 对照

| 我们的概念 | Interstellar 等价物 |
|-----------|-------------------|
| `xb_h[j]`, `xb_w[j]` | `loop_partitionings[j][level]` + `para_loop_dim` |
| `O[j][t]` | 隐式在 access 计算公式中 |
| `reduction_tree_depth` | `access_mode=1` (neighbor PE) |
| `broadcast` | `access_mode=2` |
