"""
Golden Model: Cycle-accurate DRAM simulator for PIM Optimizer verification.

This module provides a ground-truth simulator based on:
- UniNDP: Cycle-accurate bank state machine simulation (PRIMARY)
- Ramulator/OptiPIM: Timing constraint modeling

The Golden Model serves two purposes:
1. Verify the correctness of cost model formulas
2. Verify the optimality of ILP solutions

Unlike analytical formulas (which are estimations similar to ILP), 
the simulator provides exact results by modeling every memory access.

Architecture:
                    ┌─────────────────────────┐
                    │    ILP Optimizer        │
                    │  (pim_optimizer)        │
                    └───────────┬─────────────┘
                                │ mapping
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                      Golden Model                              │
├───────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐        ┌─────────────────────┐          │
│  │  PIM Cost Model │        │   UniNDP Wrapper    │          │
│  │  (analytical)   │◄──────►│   (simulation)      │          │
│  │  - Fast ~5% err │validate│  - Ground truth     │          │
│  └─────────────────┘        └─────────────────────┘          │
└───────────────────────────────────────────────────────────────┘

Usage:
    from src.golden_model import PIMCostModel, UniNDPSimulator
    
    # Quick estimation (analytical)
    model = PIMCostModel()
    result = model.estimate_mvm(M=5000, K=5000)
    print(f"Estimated cycles: {result.total_cycles}")
    
    # Ground truth simulation
    sim = UniNDPSimulator()
    sim_result = sim.simulate_gemm(M=5000, K=5000)
    print(f"Simulated cycles: {sim_result.cycles}")
"""

# Primary: Validated cost model and UniNDP wrapper
from .pim_cost_model import (
    PIMCostModel,
    PIMArchConfig, 
    TilingConfig,
    WorkloadSpec,
    CostResult,
    DataLayout,
    estimate_pim_cycles,
)

from .unindp_wrapper import (
    UniNDPSimulator,
    UniNDPResult,
)

# Bridge: ILP to UniNDP conversion
from .unindp_bridge import (
    UniNDPBridge,
    ILPMapping,
    UniNDPStrategy,
    ilp_to_unindp_strategy,
    verify_ilp_with_unindp,
)

# Legacy: Original simulator components (kept for compatibility)
from .simulator import Simulator, SimulatorConfig, SimulationResult, simulate_mapping
from .dram import DRAMBank, BankState, DRAMTiming
from .access_trace import (
    AccessTrace, 
    AccessType, 
    AccessPatternGenerator,
    MemoryAccess,
    analyze_row_crossing,
)
from .verification import (
    ILPVerifier,
    VerificationResult,
    generate_report,
    quick_verify,
)

__all__ = [
    # Primary: Validated models
    'PIMCostModel',
    'PIMArchConfig',
    'TilingConfig', 
    'WorkloadSpec',
    'CostResult',
    'DataLayout',
    'estimate_pim_cycles',
    'UniNDPSimulator',
    'UniNDPResult',
    # Bridge: ILP → UniNDP
    'UniNDPBridge',
    'ILPMapping',
    'UniNDPStrategy',
    'ilp_to_unindp_strategy',
    'verify_ilp_with_unindp',
    # Legacy: Simulator
    'Simulator',
    'SimulatorConfig',
    'SimulationResult',
    'simulate_mapping',
    # Legacy: DRAM model
    'DRAMBank',
    'BankState',
    'DRAMTiming',
    # Legacy: Access trace
    'AccessTrace',
    'AccessType',
    'AccessPatternGenerator',
    'MemoryAccess',
    'analyze_row_crossing',
    # Legacy: Verification
    'ILPVerifier',
    'VerificationResult',
    'generate_report',
    'quick_verify',
]
