#!/usr/bin/env python3
"""
PE Array Dataflow 与 Parallel_for 映射关系总结
================================================

这个文件总结了：
1. parallel_for 配置 → PE Array Dataflow
2. 不同计算单元类型的特性
3. ILP 需要支持的配置空间
"""

# =============================================================================
# Part 1: Parallel_for → Dataflow 映射表
# =============================================================================

DATAFLOW_TABLE = """
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    Parallel_for 配置 → PE Array Dataflow 映射表                          │
├───────────────┬───────────────┬─────────────────┬─────────────┬─────────────────────────┤
│   H 方向       │   W 方向       │   Dataflow       │  Reduction  │  典型架构               │
├───────────────┼───────────────┼─────────────────┼─────────────┼─────────────────────────┤
│ K             │ P, Q, N       │ Weight Stationary│ None        │ TPU (Weight Stationary) │
│ K             │ C             │ Output Stationary│ C (W方向)   │ ShiDianNao             │
│ P, Q, N       │ K             │ Weight Stationary│ None        │ (rotated WS)           │
│ C             │ K             │ Input Stationary │ C (H方向)   │ NVDLA                  │
│ K, C          │ P, Q          │ Mixed            │ C (H方向)   │ Custom                 │
│ R, S          │ K             │ Row Stationary   │ R,S (H方向) │ Eyeriss               │
│ R, S          │ P, Q          │ Row Stationary   │ R,S (H方向) │ Eyeriss v2            │
│ P, Q          │ P, Q          │ Output Stationary│ None        │ (2D output tiling)     │
│ C             │ C             │ ⚠️ 2D Reduction  │ C (H+W)     │ 需要特殊硬件            │
├───────────────┴───────────────┴─────────────────┴─────────────┴─────────────────────────┤
│ 说明:                                                                                    │
│ • H方向: PE阵列的行方向（垂直），同一列的PE处理不同的H维度值                                │
│ • W方向: PE阵列的列方向（水平），同一行的PE处理不同的W维度值                                │
│ • Reduction: 当parallel_for在Output的无关维度(R,S,C)上时，需要累加部分和                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
"""

# =============================================================================
# Part 2: 计算单元类型
# =============================================================================

COMPUTE_UNIT_TABLE = """
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              计算单元类型对比                                             │
├──────────────┬────────────┬───────────────┬───────────────┬─────────────────────────────┤
│ 类型          │ 内部结构    │ Reduction 方式 │ 延迟特性       │ 代表架构                    │
├──────────────┼────────────┼───────────────┼───────────────┼─────────────────────────────┤
│              │            │               │               │                             │
│ Scalar PE    │ 1×1 MAC    │ 外部 (buffer  │ 1 cycle/MAC   │ Eyeriss, NVDLA             │
│              │            │ 或 PE网络)    │               │                             │
├──────────────┼────────────┼───────────────┼───────────────┼─────────────────────────────┤
│              │ N×N MAC    │ Systolic Flow │ N cycles      │                             │
│ Systolic     │ PE间直连   │ (PE到PE传递   │ (pipeline     │ TPU, Gemmini              │
│ Array        │            │  psum)        │  填充)        │                             │
├──────────────┼────────────┼───────────────┼───────────────┼─────────────────────────────┤
│              │ M×N×K MAC  │ 内部 Reduction│ 1 cycle       │                             │
│ Tensor Core  │ + 硬件累加树│ Tree (K路累加)│ (fully       │ NVIDIA Tensor Core         │
│              │            │               │  pipelined)   │ AMD Matrix Core            │
├──────────────┼────────────┼───────────────┼───────────────┼─────────────────────────────┤
│              │ Vector MAC │ 向量累加      │ 1 cycle       │                             │
│ Vector Unit  │ (SIMD)     │ (内部或外部)  │ per vector    │ ARM SVE, AVX-512          │
│              │            │               │               │                             │
└──────────────┴────────────┴───────────────┴───────────────┴─────────────────────────────┘
"""

# =============================================================================
# Part 3: Reduction 策略
# =============================================================================

REDUCTION_TABLE = """
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              Reduction 策略对比                                          │
├──────────────────┬───────────────┬───────────────┬──────────────────────────────────────┤
│ 策略              │ 延迟          │ 硬件需求       │ 适用场景                              │
├──────────────────┼───────────────┼───────────────┼──────────────────────────────────────┤
│                  │               │               │                                      │
│ Hardware Tree    │ O(log₂ N)     │ 加法树电路     │ Tensor Core 内部                     │
│ (硬件累加树)      │ stages        │ (固定拓扑)    │ 固定 reduction 维度                   │
│                  │               │               │                                      │
├──────────────────┼───────────────┼───────────────┼──────────────────────────────────────┤
│                  │               │               │                                      │
│ Systolic Flow    │ O(N) cycles   │ PE间直连      │ TPU Weight Stationary               │
│ (脉动传递)        │ (可pipeline)  │ (灵活拓扑)    │ 大规模矩阵乘                          │
│                  │               │               │                                      │
├──────────────────┼───────────────┼───────────────┼──────────────────────────────────────┤
│                  │               │               │                                      │
│ Buffer Reduction │ O(2N) cycles  │ 只需 buffer   │ 灵活 dataflow                        │
│ (写回再读)        │ (read+write)  │ (最灵活)      │ 软件可配置                            │
│                  │               │               │                                      │
├──────────────────┼───────────────┼───────────────┼──────────────────────────────────────┤
│                  │               │               │                                      │
│ NoC Reduction    │ O(log₂ N) ~   │ 片上网络      │ 大规模多核                            │
│ (网络规约)        │ O(N)          │ (路由开销)    │ 多 chiplet                           │
│                  │               │               │                                      │
└──────────────────┴───────────────┴───────────────┴──────────────────────────────────────┘
"""

# =============================================================================
# Part 4: 我们需要支持的配置空间
# =============================================================================

SUPPORT_MATRIX = """
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         ILP 优化器需要支持的配置空间                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  1. PE Array 拓扑                                                                       │
│     ├─ 1D Array (向量)                                                                  │
│     ├─ 2D Array (矩阵)        ← 当前重点                                                │
│     └─ 3D Array (张量, 少见)                                                            │
│                                                                                         │
│  2. 计算单元类型                                                                         │
│     ├─ Scalar PE              ← 已支持 (默认)                                           │
│     ├─ Systolic Array         ← 需要支持 (TPU风格)                                      │
│     └─ Tensor Core            ← 需要支持 (GPU风格)                                      │
│                                                                                         │
│  3. 数据流模式 (ILP 自动探索)                                                            │
│     ├─ Weight Stationary      parallel_for on K                                        │
│     ├─ Output Stationary      parallel_for on P,Q,N                                    │
│     ├─ Input Stationary       parallel_for on C (需要reduction)                        │
│     ├─ Row Stationary         parallel_for on R,S (需要reduction)                      │
│     └─ Mixed/Custom           parallel_for on multiple dims                            │
│                                                                                         │
│  4. Reduction 支持                                                                      │
│     ├─ None                   Output-relevant dims only → 无需规约                      │
│     ├─ 1D Reduction           H方向 或 W方向 规约                                       │
│     └─ 2D Reduction           H方向 和 W方向 都需规约 (复杂，可选禁用)                   │
│                                                                                         │
│  5. 多维度映射                                                                          │
│     ├─ Single-dim per axis    每个方向只映射一个维度 (简单)                              │
│     ├─ Multi-dim per axis     每个方向映射多个维度 (如 H={K,C})  ← 已支持              │
│     └─ Split dimension        同一维度拆分到H和W (如 C_h×C_w)    ← 可选支持            │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
"""

# =============================================================================
# Part 5: Tensor Core 详细建模
# =============================================================================

TENSOR_CORE_MODEL = """
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         Tensor Core 建模 (以 16×16×16 为例)                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   计算: C[16×16] += A[16×16] × B[16×16]                                                 │
│                                                                                         │
│   内部结构:                                                                              │
│   ┌─────────────────────────────────────────────────────────┐                           │
│   │  16×16 = 256 个 PE，每个 PE:                             │                           │
│   │    • 接收 A 的一行 (16 元素)                             │                           │
│   │    • 接收 B 的一列 (16 元素)                             │                           │
│   │    • 计算 16 个乘法                                      │                           │
│   │    • 内部 16-way reduction tree → 1 个累加结果           │                           │
│   │    • 累加到 C 的对应位置                                 │                           │
│   └─────────────────────────────────────────────────────────┘                           │
│                                                                                         │
│   等效 parallel_for:                                                                    │
│     H 方向: K (Output 的行)        → 16 并行                                            │
│     W 方向: N 或 Q (Output 的列)   → 16 并行                                            │
│     内部:   C (Reduction 维度)     → 16-way reduction tree                              │
│                                                                                         │
│   Dataflow: Output Stationary + 内部 Reduction Tree                                     │
│                                                                                         │
│   ILP 建模方式:                                                                          │
│     • compute_unit.type = "tensor_core"                                                 │
│     • compute_unit.size = (16, 16)          # Output tile size                          │
│     • compute_unit.reduction_dim = 16       # 内部 reduction 维度                       │
│     • compute_unit.reduction_latency = 1    # Pipelined, 1 cycle per output tile       │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
"""

# =============================================================================
# Part 6: 建议的 ComputeUnit 数据结构
# =============================================================================

COMPUTE_UNIT_SPEC = """
建议的 ComputeUnit 数据结构:

```python
@dataclass
class ComputeUnit:
    '''计算单元规格'''
    
    # 基本信息
    type: str = "scalar"           # "scalar", "systolic", "tensor_core", "vector"
    
    # 尺寸 (对外表现的并行度)
    pe_array_h: int = 16           # H 方向 PE 数量
    pe_array_w: int = 16           # W 方向 PE 数量
    
    # 内部结构 (Tensor Core / Systolic 特有)
    internal_reduction_dim: int = 1   # 内部 reduction 维度 (Tensor Core = K)
    
    # Reduction 能力
    reduction_type: str = "buffer"    # "none", "tree", "systolic", "buffer"
    reduction_tree_depth: int = 0     # 硬件树深度, 0 = 无硬件树
    reduction_latency: float = 1.0    # 每级 reduction 延迟
    
    # 数据流约束
    supported_dataflows: list = None  # None = 全部支持, 或 ["ws", "os", "is"]
    
    # 性能参数
    mac_per_cycle: int = 1            # 每 PE 每周期 MAC 数
    frequency_mhz: float = 1000.0     # 工作频率
```

使用示例:

```python
# Eyeriss 风格: Scalar PE, 无内部 reduction
eyeriss_cu = ComputeUnit(
    type="scalar",
    pe_array_h=12, pe_array_w=14,
    reduction_type="buffer",
)

# TPU 风格: Systolic Array
tpu_cu = ComputeUnit(
    type="systolic",
    pe_array_h=128, pe_array_w=128,
    reduction_type="systolic",
    reduction_latency=1.0,  # 1 cycle per PE hop
)

# Tensor Core 风格
tensor_core_cu = ComputeUnit(
    type="tensor_core",
    pe_array_h=16, pe_array_w=16,
    internal_reduction_dim=16,  # K=16 内部累加
    reduction_type="tree",
    reduction_tree_depth=4,     # log2(16) = 4
    reduction_latency=0.25,     # Pipelined
    mac_per_cycle=16,           # 每 PE 16 MAC (因为 K=16)
)
```
"""

def print_all():
    print(DATAFLOW_TABLE)
    print(COMPUTE_UNIT_TABLE)
    print(REDUCTION_TABLE)
    print(SUPPORT_MATRIX)
    print(TENSOR_CORE_MODEL)
    print(COMPUTE_UNIT_SPEC)


if __name__ == "__main__":
    print_all()
