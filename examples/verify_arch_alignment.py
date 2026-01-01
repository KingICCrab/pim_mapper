#!/usr/bin/env python3
"""
验证 UniNDP 配置是否与 pim_optimizer 架构对齐
"""

import yaml

print("="*70)
print("架构对齐验证: pim_optimizer vs UniNDP (aligned config)")
print("="*70)

# 加载 pim_optimizer 配置
with open('/Users/haochenzhao/Projects/pim_optimizer/examples/configs/arch.yaml', 'r') as f:
    pim_config = yaml.safe_load(f)

# 加载对齐后的 UniNDP 配置
with open('/Users/haochenzhao/Projects/UniNDP/config/pim-optimizer-aligned.yaml', 'r') as f:
    unindp_config = yaml.safe_load(f)

print("\n【计算能力对比】")
print("-" * 50)

# pim_optimizer
pe_h = pim_config['architecture']['pe_array']['dim_h']
pe_w = pim_config['architecture']['pe_array']['dim_w']
macs_per_pe = pim_config['architecture']['pe_array']['num_macs']
total_pes = pe_h * pe_w
pim_total_macs = total_pes * macs_per_pe

print(f"pim_optimizer:")
print(f"  PE Array: {pe_h} × {pe_w} = {total_pes} PEs")
print(f"  MACs/PE: {macs_per_pe}")
print(f"  总 MACs: {pim_total_macs}")

# UniNDP
num_pus = unindp_config['de_pu'][0]
macs_per_pu = unindp_config['de_pu_w'][2]
channels = unindp_config['ch']
unindp_total_macs = num_pus * macs_per_pu * channels

print(f"\nUniNDP (aligned):")
print(f"  Channels: {channels}")
print(f"  PUs/Channel: {num_pus}")
print(f"  MACs/PU: {macs_per_pu}")
print(f"  总 MACs: {unindp_total_macs}")

if pim_total_macs == unindp_total_macs:
    print(f"\n✅ 计算能力匹配: {pim_total_macs} MACs")
else:
    print(f"\n❌ 计算能力不匹配: {pim_total_macs} vs {unindp_total_macs}")

print("\n【数据精度对比】")
print("-" * 50)
pim_precision = pim_config['architecture']['dram_timings']['data_pr']
unindp_precision = unindp_config['data_pr']
print(f"pim_optimizer: {pim_precision} bit")
print(f"UniNDP: {unindp_precision} bit")
if pim_precision == unindp_precision:
    print("✅ 数据精度匹配")
else:
    print("❌ 数据精度不匹配")

print("\n【Bank 配置对比】")
print("-" * 50)
# 找 pim_optimizer 的 bank 数
pim_banks = None
for mem in pim_config['architecture']['memory_hierarchy']:
    if mem['name'] == 'LocalDRAM' and 'num_banks' in mem:
        pim_banks = mem['num_banks']
        break

unindp_banks = unindp_config['bg'] * unindp_config['ba']
print(f"pim_optimizer: {pim_banks} banks")
print(f"UniNDP: {unindp_config['bg']} BG × {unindp_config['ba']} BA = {unindp_banks} banks")
if pim_banks == unindp_banks:
    print("✅ Bank 数量匹配")
else:
    print("❌ Bank 数量不匹配")

print("\n【DRAM 时序对比】")
print("-" * 50)
timings = ['RL', 'WL', 'tRCDRD', 'tRCDWR', 'tRP', 'tCCDL', 'BL']
all_match = True
for t in timings:
    pim_val = pim_config['architecture']['dram_timings'].get(t, 'N/A')
    unindp_val = unindp_config.get(t, 'N/A')
    match = "✅" if pim_val == unindp_val else "❌"
    if pim_val != unindp_val:
        all_match = False
    print(f"  {t:10s}: pim={pim_val:6} | unindp={unindp_val:6} {match}")

print("\n" + "="*70)
print("对齐总结")
print("="*70)
checks = [
    ("计算能力", pim_total_macs == unindp_total_macs),
    ("数据精度", pim_precision == unindp_precision),
    ("Bank数量", pim_banks == unindp_banks),
    ("时序参数", all_match),
]

all_pass = True
for name, passed in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {name}: {status}")
    if not passed:
        all_pass = False

if all_pass:
    print("\n🎉 所有配置已对齐！可以进行有效验证。")
else:
    print("\n⚠️ 部分配置未对齐，需要进一步调整。")
