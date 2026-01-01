"""
Utility functions for PIM optimizer.
"""

import os
import sys
import contextlib
from typing import Iterator


@contextlib.contextmanager
def suppress_gurobi_logger() -> Iterator[None]:
    """
    Context manager to suppress Gurobi output.
    
    Usage:
        with suppress_gurobi_logger():
            model.optimize()
    """
    import gurobipy as gp
    
    # Create a temporary environment with no output
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        yield


@contextlib.contextmanager 
def redirect_stdout_to_null() -> Iterator[None]:
    """
    Redirect stdout to null device.
    
    Useful for suppressing print statements from libraries.
    """
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def create_gurobi_model(
    name: str = "PIMOptimizer",
    verbose: bool = False,
    time_limit: float = None,
    mip_gap: float = None,
    threads: int = None,
) -> "gp.Model":
    """
    Create a Gurobi model with standard settings.
    
    Args:
        name: Model name
        verbose: Whether to enable Gurobi output
        time_limit: Solver time limit in seconds
        mip_gap: MIP optimality gap tolerance
        threads: Number of threads to use
        
    Returns:
        Configured Gurobi model
    """
    import gurobipy as gp
    
    model = gp.Model(name)
    
    if not verbose:
        model.setParam("OutputFlag", 0)
    
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    
    if mip_gap is not None:
        model.setParam("MIPGap", mip_gap)
    
    if threads is not None:
        model.setParam("Threads", threads)
    
    # Default settings for numerical stability
    model.setParam("NumericFocus", 1)
    model.setParam("IntFeasTol", 1e-6)
    model.setParam("FeasibilityTol", 1e-6)
    
    return model


def get_divisors(n: int, include_one: bool = True) -> list[int]:
    """
    Get all divisors of a number.
    
    Args:
        n: The number to factorize
        include_one: Whether to include 1 as a divisor
        
    Returns:
        List of divisors in ascending order
    """
    if n <= 0:
        return [1] if include_one else []
    
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    
    if not include_one and 1 in divisors:
        divisors.remove(1)
    
    return divisors


def get_prime_factors(n: int) -> list[int]:
    """
    Get prime factorization of a number.
    
    Args:
        n: Number to factorize
        
    Returns:
        List of prime factors (with repetition)
    """
    if n <= 1:
        return []
    
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    return factors


def compute_gcd(a: int, b: int) -> int:
    """Compute greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


def compute_lcm(a: int, b: int) -> int:
    """Compute least common multiple."""
    return a * b // compute_gcd(a, b)


def format_number(n: float, precision: int = 2) -> str:
    """
    Format a number for display.
    
    Uses scientific notation for very large/small numbers.
    """
    if abs(n) >= 1e6 or (abs(n) < 1e-3 and n != 0):
        return f"{n:.{precision}e}"
    else:
        return f"{n:.{precision}f}"


def estimate_macs(workload) -> int:
    """
    Estimate MACs for a convolution workload.
    
    MACs = R × S × P × Q × C × K × N
    """
    bounds = workload.bounds
    macs = 1
    for b in bounds:
        macs *= b
    return macs


def validate_mapping(mapping, workload) -> list[str]:
    """
    Validate a mapping against a workload.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check dimension factorization
    for dim_idx, dim_bound in enumerate(workload.bounds):
        total = 1
        for m in mapping.loop_bounds:
            for s in ("spatial", "temporal"):
                if s in mapping.loop_bounds[m]:
                    total *= mapping.loop_bounds[m][s].get(dim_idx, 1)
        
        if total != dim_bound:
            errors.append(
                f"Dimension {dim_idx}: product of factors ({total}) != "
                f"problem size ({dim_bound})"
            )
    
    return errors


class Timer:
    """Simple timer for profiling."""
    
    def __init__(self):
        self.times = {}
        self._starts = {}
    
    def start(self, name: str):
        """Start timing a section."""
        import time
        self._starts[name] = time.perf_counter()
    
    def stop(self, name: str):
        """Stop timing a section."""
        import time
        if name in self._starts:
            elapsed = time.perf_counter() - self._starts[name]
            if name not in self.times:
                self.times[name] = 0.0
            self.times[name] += elapsed
            del self._starts[name]
    
    def report(self) -> str:
        """Generate timing report."""
        lines = ["Timing Report:"]
        for name, elapsed in sorted(self.times.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {elapsed:.3f}s")
        return "\n".join(lines)
