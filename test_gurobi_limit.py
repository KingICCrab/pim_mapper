
import gurobipy as gp

try:
    m = gp.Model("test")
    m.setParam("NumericFocus", 1)
    m.setParam("IntFeasTol", 1e-6)
    m.setParam("MIPFocus", 1)
    # Add 2001 variables
    x = m.addVars(2001, vtype=gp.GRB.BINARY)
    m.update()
    print(f"Model has {m.NumVars} variables")
    m.optimize()
    print("Optimization successful")
except gp.GurobiError as e:
    print(f"Gurobi Error: {e}")
except Exception as e:
    print(f"Error: {e}")
