import gurobipy as gp
from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload import ConvWorkload
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

def test_ilp_lookup_pruned():
    # Ensure table exists
    if not os.path.exists("row_activation_cost_table.json"):
        print("Table not found!")
        return

    # Create workload matching the table generation
    wl = ConvWorkload(name="TestLayer", N=1, C=64, K=64, P=56, Q=56, R=3, S=3, stride=(1,1))
    
    optimizer = PIMOptimizer(verbose=True)
    optimizer.time_limit = 5.0
    
    print("Starting optimization...")
    try:
        result = optimizer.optimize([wl], enable_row_activation=True)
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
    except Exception as e:
        print(f"Other Error: {e}")
    
    model = optimizer.model
    
    # Check for z_lut variables
    z_vars = [v for v in model.getVars() if "z_lut" in v.VarName]
    print(f"Found {len(z_vars)} z_lut variables (Pruned).")
    
    # Check constraints count
    constrs = model.getConstrs()
    print(f"Total constraints: {len(constrs)}")
    
    if len(z_vars) <= 50:
        print("Success: Pruning effective (<= 50 variables).")
    else:
        print(f"Warning: Pruning might not be aggressive enough ({len(z_vars)} vars).")

if __name__ == "__main__":
    test_ilp_lookup_pruned()
