#!/usr/bin/env python3
"""
架构配置对比分析

对比 pim_optimizer 的 arch 配置和 UniNDP 的 hbm-pim.yaml
"""

print("="*70)
print("架构配置对比: pim_optimizer vs UniNDP")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                        计算单元配置对比                                  │
├─────────────────────┬────────────────────┬──────────────────────────────┤
│        参数         │   pim_optimizer    │         UniNDP              │
├─────────────────────┼────────────────────┼──────────────────────────────┤
│  PE Array           │   16 × 16 = 256    │      N/A (不同模型)          │
│  MACs per PE        │        8           │      N/A                     │
│  Total MACs         │   256 × 8 = 2048   │      N/A                     │
├─────────────────────┼────────────────────┼──────────────────────────────┤
│  Channels           │       N/A          │       64                     │
│  Ranks per Channel  │       N/A          │        1                     │
│  Devices per Rank   │       N/A          │        1                     │
│  PUs per Device     │       N/A          │        8                     │
│  Total PUs          │       N/A          │   64×1×1×8 = 512             │
│  PU Parallelism     │       N/A          │   [1,1,16] (16 MACs/PU)      │
│  Total MACs         │       N/A          │   512 × 16 = 8192            │
└─────────────────────┴────────────────────┴──────────────────────────────┘
""")

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                        内存配置对比                                      │
├─────────────────────┬────────────────────┬──────────────────────────────┤
│        参数         │   pim_optimizer    │         UniNDP              │
├─────────────────────┼────────────────────┼──────────────────────────────┤
│  数据精度           │      8 bit         │       16 bit                 │
│  DRAM Rows          │       N/A          │      16384                   │
│  DRAM Columns       │       N/A          │        32                    │
│  Column Width       │     256 bit        │      256 bit                 │
│  Bank Groups        │       N/A          │         4                    │
│  Banks per Group    │       4            │         4                    │
│  Row Buffer Size    │    1024 bytes      │       N/A                    │
│  PU Buffer Size     │       N/A          │    2048 bytes (256B×8?)      │
└─────────────────────┴────────────────────┴──────────────────────────────┘
""")

print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                       DRAM 时序对比                                      │
├─────────────────────┬────────────────────┬──────────────────────────────┤
│        参数         │   pim_optimizer    │         UniNDP              │
├─────────────────────┼────────────────────┼──────────────────────────────┤
│  RL (Read Latency)  │     25 cycles      │      20 cycles               │
│  WL (Write Latency) │     25 cycles      │       8 cycles               │
│  tRCDRD             │     14 cycles      │      14 cycles     ✓         │
│  tRCDWR             │     14 cycles      │      10 cycles               │
│  tRP                │     14 cycles      │      14 cycles     ✓         │
│  tCCDL              │      2 cycles      │       4 cycles               │
│  BL (Burst Length)  │         4          │         4          ✓         │
└─────────────────────┴────────────────────┴──────────────────────────────┘
""")

print("""
⚠️  关键差异分析
─────────────────────────────────────────────────────────────────────────

1. 【计算模型完全不同】
   
   pim_optimizer: PE Array 模型
   - 16×16 PE array = 256 PEs
   - 每个 PE 有 8 个 MACs
   - 总共 2048 MACs
   
   UniNDP: Channel/Rank/Device/PU 层次模型
   - 64 Channels × 1 Rank × 1 Device × 8 PUs = 512 PUs
   - 每个 PU 有 16 个并行 MACs ([1,1,16])
   - 总共 8192 MACs
   
   ❌ 计算能力差异: 8192 / 2048 = 4x

2. 【数据精度不同】
   
   pim_optimizer: 8 bit
   UniNDP: 16 bit
   
   ❌ 这会影响内存带宽和存储需求计算

3. 【时序参数部分不同】
   
   Read Latency: 25 vs 20 cycles
   Write Latency: 25 vs 8 cycles
   tCCDL: 2 vs 4 cycles
   
   ⚠️ 这会影响延迟计算

4. 【架构层次不同】
   
   pim_optimizer: 4 级存储层次 (PE Buffer → Global Buffer → Row Buffer → DRAM)
   UniNDP: Channel/Rank/Device/PU 硬件层次
   
   ❌ 完全不同的建模方式
""")

print("""
📌 结论
─────────────────────────────────────────────────────────────────────────

当前 pim_optimizer 和 UniNDP 的架构配置 **不对齐**:

1. 计算能力差 4 倍 (2048 vs 8192 MACs)
2. 数据精度差 2 倍 (8 vs 16 bit)
3. 架构建模方式完全不同

这意味着之前的验证结果可能不准确!

要进行有意义的验证，需要:
1. 修改 pim_optimizer 的 arch.yaml 以匹配 UniNDP
2. 或者创建一个专门的 UniNDP 架构配置文件
""")

print("="*70)
