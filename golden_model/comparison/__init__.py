"""
Comparison module for golden model verification.

This module provides:
- Comparison between ILP results and analytical formulas
- Comparison between ILP results and exhaustive search
- Detailed reporting of discrepancies
"""

from .compare import (
    verify_cost_model,
    verify_optimality,
    compare_with_ilp,
    run_verification_suite,
)

from .report import (
    generate_report,
    print_verification_summary,
    export_to_csv,
)

__all__ = [
    'verify_cost_model',
    'verify_optimality',
    'compare_with_ilp',
    'run_verification_suite',
    'generate_report',
    'print_verification_summary',
    'export_to_csv',
]
