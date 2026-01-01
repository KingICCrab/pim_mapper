"""
找到公式了!

ResNet-L1 观察:
- Switches per p = [14, 14, ...] 对大多数 p
- 关键: 14 = 2 × 7 = W_unique_configs × R_l2 ?

让我验证这个规律
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
import math

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def compute_switches_per_p(name, workload, dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """计算每个 P 循环内的 switches."""
    
    P_l3 = dram[DIM_P]
    Q_l3 = dram[DIM_Q]
    R_l2 = l2[DIM_R]
    S_l2 = l2[DIM_S]
    
    P_tile = buf[DIM_P]
    Q_tile = buf[DIM_Q]
    R_tile = buf[DIM_R]
    S_tile = buf[DIM_S]
    
    H_per_tile = (P_tile - 1) * stride_h + R_tile * dilation_h
    W_per_tile = (Q_tile - 1) * stride_w + S_tile * dilation_w
    
    H_step = P_tile * stride_h
    W_step = Q_tile * stride_w
    R_step = R_tile * dilation_h
    S_step = S_tile * dilation_w
    
    H_in = workload.input_size['H']
    W_in = workload.input_size['W']
    
    num_h_blocks = (H_in + block_h - 1) // block_h
    num_w_blocks = (W_in + block_w - 1) // block_w
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Loops: P={P_l3}, Q={Q_l3}, R={R_l2}, S={S_l2}")
    print(f"Step: H={H_step}, W={W_step}, R={R_step}, S={S_step}")
    print(f"Blocks: {num_h_blocks} × {num_w_blocks}")
    
    # 计算每个 P 循环内的 switches
    switches_per_p = []
    total_global = 0
    prev_row_global = -1
    
    for p in range(P_l3):
        prev_row_local = prev_row_global  # 继承上一个 P 的状态
        count_local = 0
        for q in range(Q_l3):
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                    w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                    
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            if row != prev_row_local:
                                count_local += 1
                                total_global += 1
                                prev_row_local = row
                                prev_row_global = row
        
        switches_per_p.append(count_local)
    
    print(f"Switches per p: {switches_per_p[:10]}...")
    print(f"Unique values: {sorted(set(switches_per_p))}")
    print(f"Total: {total_global}")
    
    # 统计 W 方向 unique configs
    w_unique_configs = set()
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_per_tile - 1
            w_blocks = tuple(range(w_start // block_w, w_end // block_w + 1))
            w_unique_configs.add(w_blocks)
    
    print(f"\nW unique configs: {len(w_unique_configs)}")
    print(f"  {w_unique_configs}")
    
    # 统计每个 W config 包含几个 blocks
    w_blocks_per_config = [len(cfg) for cfg in w_unique_configs]
    print(f"W blocks per config: {w_blocks_per_config}")
    
    # 统计 H 方向 unique configs (在一个 P 内, 遍历所有 R)
    h_unique_configs_per_p = []
    for p in range(P_l3):
        h_configs = set()
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            h_blocks = tuple(range(h_start // block_h, h_end // block_h + 1))
            h_configs.add(h_blocks)
        h_unique_configs_per_p.append(len(h_configs))
    
    print(f"\nH unique configs per p: {h_unique_configs_per_p[:10]}...")
    print(f"Unique values: {sorted(set(h_unique_configs_per_p))}")
    
    # 尝试公式: switches_per_p[i] = H_configs[i] × sum(W_blocks_per_config)
    # 或者: switches_per_p[i] = H_configs[i] × W_unique_configs × something
    
    print("\n验证公式:")
    for i in range(min(5, P_l3)):
        h_configs = set()
        for r in range(R_l2):
            h_start = i * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            h_blocks = tuple(range(h_start // block_h, h_end // block_h + 1))
            h_configs.add(h_blocks)
        
        # 计算每个 H config 跨多少 blocks
        h_blocks_total = sum(len(cfg) for cfg in h_configs)
        
        # 预测 = H_blocks_total × W_blocks_total / 某种因子
        w_blocks_total = sum(len(cfg) for cfg in w_unique_configs)
        
        predict_1 = h_blocks_total * len(w_unique_configs)
        predict_2 = len(h_configs) * w_blocks_total
        predict_3 = h_blocks_total * w_blocks_total
        
        print(f"  p={i}: actual={switches_per_p[i]}, H_blk={h_blocks_total}, H_cfg={len(h_configs)}, "
              f"predict1={predict_1}, predict2={predict_2}, predict3={predict_3}")
    
    return switches_per_p, total_global


def main():
    test_workloads = [
        ("ResNet-L1", ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)),
        ("ResNet-L2", ConvWorkload(name="ResNet-L2", R=3, S=3, P=56, Q=56, C=64, K=64, N=1)),
        ("ResNet-L3", ConvWorkload(name="ResNet-L3", R=1, S=1, P=56, Q=56, C=64, K=256, N=1)),
        ("VGG-L1", ConvWorkload(name="VGG-L1", R=3, S=3, P=224, Q=224, C=3, K=64, N=1)),
    ]
    
    optimizer = PIMOptimizer()
    
    for name, workload in test_workloads:
        result = optimizer.optimize([workload], objective="latency")
        mapping = result.mappings[0]
        
        dram = {d: 1 for d in range(7)}
        if 3 in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[3]:
                    for d, bound in mapping.loop_bounds[3][key].items():
                        dram[d] *= bound
        
        l2 = {d: 1 for d in range(7)}
        if 2 in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[2]:
                    for d, bound in mapping.loop_bounds[2][key].items():
                        l2[d] *= bound
        
        buf = {d: 1 for d in range(7)}
        for level in [0, 1]:
            if level not in mapping.loop_bounds:
                continue
            level_bounds = mapping.loop_bounds[level]
            if level == 0:
                for key in ['H', 'W', 'Internal', 'temporal']:
                    if key in level_bounds:
                        for d, bound in level_bounds[key].items():
                            buf[d] *= bound
            else:
                for key in ['spatial', 'temporal']:
                    if key in level_bounds:
                        for d, bound in level_bounds[key].items():
                            buf[d] *= bound
        
        block_h = mapping.tile_info.get('block_h', 31)
        block_w = mapping.tile_info.get('block_w', 31)
        stride_h = workload.stride[0] if hasattr(workload, 'stride') else 1
        stride_w = workload.stride[1] if hasattr(workload, 'stride') else 1
        dilation_h = workload.dilation[0] if hasattr(workload, 'dilation') else 1
        dilation_w = workload.dilation[1] if hasattr(workload, 'dilation') else 1
        
        compute_switches_per_p(name, workload, dram, l2, buf, block_h, block_w,
                              stride_h, stride_w, dilation_h, dilation_w)


if __name__ == "__main__":
    main()
