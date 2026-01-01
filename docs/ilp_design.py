#!/usr/bin/env python3
"""
ILP 设计方案：三层并行度映射空间

目标：给定 workload 和 architecture，找到最优的维度映射
  - H 方向映射哪些维度，顺序如何
  - W 方向映射哪些维度，顺序如何
  - PE 内部映射哪些维度（由硬件决定，可能是固定的）
"""

print("""
================================================================================
                           ILP 设计方案
================================================================================

输入:
  • Workload: 7个维度 {R, S, P, Q, C, K, N} 及其大小
  • Architecture:
    - PE Array: H_size × W_size
    - 计算单元类型: scalar / systolic / tensor_core
    - PE 内部并行度: internal_size (如 Tensor Core = 16)
    - 内存带宽限制: BW_limit

输出:
  • 每个维度映射到哪个层级 (H / W / Internal / Temporal)
  • 每个层级内维度的顺序 (内层/外层)
  • 总延迟和/或能耗估计

================================================================================
                          Part 1: ILP 变量
================================================================================

1. 维度-层级分配变量 (Binary)
────────────────────────────────────────────────────────────────────────────────

  assign[j, level] ∈ {0, 1}
  
  其中:
    j ∈ {0,1,2,3,4,5,6} = {R, S, P, Q, C, K, N}
    level ∈ {H, W, Internal, Temporal}
  
  含义: assign[j, H] = 1 表示维度 j 被映射到 H 方向

  约束: 每个维度恰好映射到一个层级
    ∀j: Σ_level assign[j, level] = 1

────────────────────────────────────────────────────────────────────────────────

2. 并行因子选择变量 (Binary)
────────────────────────────────────────────────────────────────────────────────

  xb[j, level, i] ∈ {0, 1}
  
  其中:
    j = 维度索引
    level ∈ {H, W, Internal}
    i = 因子索引 (dim_j 的第 i 个因子)
  
  含义: xb[K, H, 3] = 1 表示 K 维度在 H 方向使用第3个因子 (如 K=16 的因子 4)

  约束: 如果维度 j 映射到 level，则恰好选择一个因子
    ∀j, level: Σ_i xb[j, level, i] = assign[j, level]

────────────────────────────────────────────────────────────────────────────────

3. 维度顺序变量 (用于同一层级内多个维度)
────────────────────────────────────────────────────────────────────────────────

  方法A: Position 变量
    pos[j, level] ∈ {0, 1, ..., max_dims_per_level - 1}
    
    含义: pos[K, H] = 0 表示 K 在 H 方向是最内层
          pos[C, H] = 1 表示 C 在 H 方向是第二层
  
  方法B: Pairwise Order 变量 (更适合ILP)
    order[j1, j2, level] ∈ {0, 1}
    
    含义: order[K, C, H] = 1 表示在 H 方向，K 比 C 更内层
    
    约束: 传递性 (如果 A < B 且 B < C，则 A < C)
    
  方法C: 简化 - 只关心 Reduction 维度的位置
    red_innermost[level] ∈ {0, 1}
    
    含义: 如果有 Reduction 维度在该 level，它是否在最内层
    (假设近邻 reduction 更高效时使用)

────────────────────────────────────────────────────────────────────────────────

4. 辅助变量 (用于线性化)
────────────────────────────────────────────────────────────────────────────────

  log_parallel[level] = Σ_j Σ_i xb[j, level, i] × log(div[j][i])
  
  含义: level 层级的总并行度 (log域)

  log_bw[t, level] = Σ_{j: O[j][t]=1} Σ_i xb[j, level, i] × log(div[j][i])
  
  含义: 数据类型 t 在 level 层级的带宽贡献 (log域)

================================================================================
                          Part 2: ILP 约束
================================================================================

1. 并行度约束
────────────────────────────────────────────────────────────────────────────────

  H 方向总并行度 ≤ H_size:
    log_parallel[H] ≤ log(H_size)
    
  W 方向总并行度 ≤ W_size:
    log_parallel[W] ≤ log(W_size)
    
  PE 内部并行度 ≤ Internal_size (通常是固定的):
    log_parallel[Internal] ≤ log(Internal_size)

────────────────────────────────────────────────────────────────────────────────

2. 带宽约束
────────────────────────────────────────────────────────────────────────────────

  对于每个数据类型 t ∈ {Input, Weight, Output}:
  
    总带宽需求 = H贡献 × W贡献 × Internal贡献
    
    log域: log_bw_total[t] = log_bw[t, H] + log_bw[t, W] + log_bw[t, Internal]
    
    约束: log_bw_total[t] ≤ log(BW_limit[t])

────────────────────────────────────────────────────────────────────────────────

3. 计算单元类型约束
────────────────────────────────────────────────────────────────────────────────

  Scalar PE (最灵活):
    • 无额外约束
    • Internal 层级为空 (Internal_size = 1)
  
  Tensor Core:
    • Internal 层级必须包含 Reduction 维度
      Σ_{j∈{R,S,C}} assign[j, Internal] ≥ 1
    
    • H 和 W 只能用 Output 相关维度
      ∀j∈{R,S,C}: assign[j, H] = 0
      ∀j∈{R,S,C}: assign[j, W] = 0
    
    • Internal 并行度 = 固定值 (如 16)
      log_parallel[Internal] = log(16)
  
  Systolic Array:
    • K 必须映射到 H (或其中一个方向)
      assign[K, H] = 1
    
    • C 必须映射到 W (或另一个方向)
      assign[C, W] = 1
    
    • 并行度必须完全利用
      log_parallel[H] = log(H_size)
      log_parallel[W] = log(W_size)

────────────────────────────────────────────────────────────────────────────────

4. Reduction 约束 (如果需要建模顺序对效率的影响)
────────────────────────────────────────────────────────────────────────────────

  如果 PE 互连是简单的 Mesh (近邻更高效):
    
    定义 has_reduction[level] = Σ_{j∈{R,S,C}} assign[j, level]
    
    如果 has_reduction[H] > 0 且 有多个维度在 H:
      鼓励 Reduction 维度在内层 (通过目标函数惩罚)

────────────────────────────────────────────────────────────────────────────────

5. 维度大小约束
────────────────────────────────────────────────────────────────────────────────

  每个维度的所有因子乘积 = 维度大小:
    ∀j: ∏_level ∏_i (div[j][i])^{xb[j,level,i]} = dim_size[j]
    
  Log域:
    ∀j: Σ_level Σ_i xb[j, level, i] × log(div[j][i]) = log(dim_size[j])

================================================================================
                          Part 3: 目标函数
================================================================================

目标: 最小化 总延迟 或 总能耗

1. 延迟模型
────────────────────────────────────────────────────────────────────────────────

  总延迟 = 计算延迟 + 内存延迟 + Reduction延迟
  
  计算延迟:
    = Total_MACs / (H_parallel × W_parallel × Internal_parallel)
    
    log域: log_compute = log(Total_MACs) - log_parallel[H] - log_parallel[W] - log_parallel[Internal]
  
  内存延迟 (带宽受限时):
    = max_t(Data_size[t] / BW[t])
    
  Reduction延迟 (如果 Reduction 维度在 H 或 W):
    Tensor Core: 0 (内部完成)
    Systolic:    H_size cycles (pipeline)
    Scalar:      log2(reduction_parallel) × reduction_latency

────────────────────────────────────────────────────────────────────────────────

2. 能耗模型 (可选)
────────────────────────────────────────────────────────────────────────────────

  总能耗 = 计算能耗 + 数据移动能耗
  
  数据移动能耗:
    = Σ_t (访问次数[t] × 单次访问能耗[t])
    
  访问次数与 reuse (temporal loop) 相关

================================================================================
                          Part 4: 与现有代码集成
================================================================================

需要修改/添加的文件:

1. src/pim_optimizer/arch/pim_arch.py
   - 添加 ComputeUnit 类
   
   @dataclass
   class ComputeUnit:
       type: str = "scalar"  # "scalar", "systolic", "tensor_core"
       internal_dims: list = None  # PE内部映射的维度
       internal_size: int = 1  # PE内部并行度
       reduction_type: str = "buffer"  # "tree", "systolic", "buffer"

2. src/pim_optimizer/model/variables.py
   - 添加 assign[j, level] 变量
   - 添加 xb_h[j, i], xb_w[j, i], xb_int[j, i] 变量
   - 添加 order[j1, j2, level] 变量 (可选)

3. src/pim_optimizer/model/constraints.py (新建)
   - build_parallelism_constraints()
   - build_bandwidth_constraints()
   - build_compute_unit_constraints()
   - build_reduction_constraints()

4. src/pim_optimizer/model/objective.py
   - build_latency_objective()
   - build_energy_objective() (可选)

5. src/pim_optimizer/optimizer.py
   - 集成所有约束和目标函数
   - 调用 Gurobi 求解

================================================================================
                          Part 5: 简化版本 (先实现)
================================================================================

先实现一个简化版本，逐步添加复杂功能:

V1 (最简):
  • 只有 H 和 W 两个层级 (无 Internal)
  • 不考虑维度顺序
  • 只考虑 Scalar PE
  • 目标: 最大化并行度

V2:
  • 添加 Internal 层级
  • 支持 Tensor Core (固定 Internal 配置)
  • 添加带宽约束

V3:
  • 添加维度顺序变量
  • 支持 Systolic Array
  • 添加 Reduction 延迟到目标函数

V4 (完整):
  • 完整的延迟/能耗模型
  • 支持多 workload 调度
  • 支持层融合优化
""")
