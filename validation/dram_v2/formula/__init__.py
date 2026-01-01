"""
Row activation formula module.

This module contains the simplified row activation formulas used in the ILP model.
"""

from .row_activation_formula import (
    FormulaConfig,
    compute_input_row_switches_formula,
    compute_weight_row_switches_formula,
    compute_output_row_switches_formula,
    compute_total_row_switches_formula,
)

__all__ = [
    'FormulaConfig',
    'compute_input_row_switches_formula',
    'compute_weight_row_switches_formula',
    'compute_output_row_switches_formula',
    'compute_total_row_switches_formula',
]
