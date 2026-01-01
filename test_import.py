import sys
import os
sys.path.insert(0, os.getcwd())
try:
    from validation.dram_v2.core.workload import Workload
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    print(sys.path)
