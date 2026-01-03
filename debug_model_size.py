
import sys
import gurobipy as gp
import numpy as np
from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

class DebugOptimizer(PIMOptimizer):
    def optimize(self, workloads, **kwargs):
        self.workloads = workloads
        self.model = gp.Model("PIM_Optimizer")
        
        # 1. Create variables
        from pim_optimizer.model import create_decision_variables
        self.vars = create_decision_variables(
            self.model, 
            self.arch, 
            workloads,
            fix_permutations=kwargs.get('fix_permutations', False),
            fix_bypass=kwargs.get('fix_bypass', False),
            optimize_bypass=kwargs.get('optimize_bypass', False)
        )
        
        # 2. Create expressions
        from pim_optimizer.model.expressions import build_buffer_utilization, build_memory_expressions
        from pim_optimizer.model.objective import build_compute_cycles
        
        buf_util, buf_util_log, tile_info = build_buffer_utilization(self.model, self.vars, self.arch, workloads)
        mem_writes_inst, mem_writes, mem_reads_inst, mem_reads = build_memory_expressions(self.model, self.vars, self.arch, workloads, buf_util_log)
        
        # Calculate scale factors
        MAX_BOUND = 1e4
        macs_scale_factors = [MAX_BOUND / (1.02 * w.macs) for w in workloads]
        macs_scale_factor_logs = np.array([np.log(sf) for sf in macs_scale_factors])
        
        compute_cycles, _ = build_compute_cycles(self.model, self.vars, self.arch, workloads, macs_scale_factors, macs_scale_factor_logs)

        # 3. Add constraints
        from pim_optimizer.model import add_basic_constraints, add_buffer_constraints, add_pe_parallelism_constraints, add_compute_unit_constraints
        add_basic_constraints(self.model, self.vars, self.arch, workloads)
        add_buffer_constraints(self.model, self.vars, self.arch, workloads, buf_util, buf_util_log)
        add_pe_parallelism_constraints(self.model, self.vars, self.arch, workloads)
        add_compute_unit_constraints(self.model, self.vars, self.arch, workloads)
        
        # 4. Add objective
        from pim_optimizer.model.objective import build_latency_objective, set_objective
        
        mem_reads_vars = {}
        for w in range(len(workloads)):
            for m in range(1, self.arch.num_mems):
                for t in range(3):
                    if (w, m, t) in mem_reads:
                        mem_reads_vars[w, m, t] = self.model.addVar(name=f"mem_reads_{w}_{m}_{t}")
        
        lat_vars, total_latency = build_latency_objective(self.model, self.vars, self.arch, workloads, compute_cycles, mem_reads_vars, macs_scale_factors, enable_row_activation=False)
        set_objective(self.model, total_latency, "latency")
            
        self.model.update()
        print(f"Model Stats:")
        print(f"  Variables: {self.model.NumVars}")
        print(f"  Constraints: {self.model.NumConstrs}")
        
        return None

workload = ConvWorkload(
    name="Tiny_Test",
    P=4, Q=4, C=16, K=16, R=1, S=1, stride=(1,1), dilation=(1,1)
)

opt = DebugOptimizer()
opt.optimize([workload], enable_row_activation=False)
