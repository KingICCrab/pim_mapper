"""
DRAM Bank and Timing Model for Golden Model Simulator.

Based on:
- UniNDP sim/bank.py: Bank state machine (IDLE/ROWOPEN)
- Ramulator node.h: Timing constraint tracking

This provides cycle-accurate simulation of DRAM bank behavior:
- Row buffer state tracking (open row, idle)
- Row hit vs row miss detection
- DRAM timing constraint enforcement
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


class BankState(Enum):
    """Bank state machine states."""
    IDLE = 0      # No row is open
    ROWOPEN = 1   # A row is currently open in the row buffer


@dataclass
class DRAMTiming:
    """
    DRAM timing parameters (in cycles).
    
    Based on typical DDR4/HBM parameters, following UniNDP convention.
    All values are in memory clock cycles.
    """
    # Row Access
    tRCD: int = 14      # Row-to-Column Delay (ACT to RD/WR)
    tRP: int = 14       # Row Precharge time (PRE to ACT)
    tRAS: int = 32      # Row Access Strobe (ACT to PRE minimum)
    tRC: int = 46       # Row Cycle time (ACT to ACT same bank) = tRAS + tRP
    
    # Column Access
    tCCD_L: int = 4     # Column-to-Column Delay (same bank group)
    tCCD_S: int = 4     # Column-to-Column Delay (different bank group)
    tBL: int = 4        # Burst Length (data transfer time)
    
    # Read/Write Latency
    tCL: int = 14       # CAS Latency (RD command to data out)
    tCWL: int = 12      # CAS Write Latency (WR command to data in)
    
    # Read-Write Turnaround
    tRTW: int = 8       # Read to Write turnaround
    tWTR_L: int = 8     # Write to Read turnaround (same bank group)
    tWTR_S: int = 4     # Write to Read turnaround (different bank group)
    
    # Write Recovery
    tWR: int = 15       # Write Recovery time
    
    @property
    def row_hit_latency(self) -> int:
        """Latency for row buffer hit (just column access)."""
        return self.tCL + self.tBL
    
    @property
    def row_miss_latency(self) -> int:
        """Latency for row buffer miss (precharge + activate + column access)."""
        return self.tRP + self.tRCD + self.tCL + self.tBL
    
    @property
    def row_empty_latency(self) -> int:
        """Latency when bank is idle (activate + column access)."""
        return self.tRCD + self.tCL + self.tBL


@dataclass 
class BankTimingState:
    """
    Tracks the next available cycle for each operation type.
    
    Based on UniNDP's np_bankstate array:
    - nxt_act: Next cycle ACT can be issued
    - nxt_pre: Next cycle PRE can be issued
    - nxt_read: Next cycle RD can be issued
    - nxt_write: Next cycle WR can be issued
    """
    nxt_act: int = 0
    nxt_pre: int = 0
    nxt_read: int = 0
    nxt_write: int = 0


class DRAMBank:
    """
    Cycle-accurate DRAM bank model.
    
    Simulates a single DRAM bank with:
    - Row buffer state machine (IDLE/ROWOPEN)
    - Open row tracking
    - Timing constraint enforcement
    
    Based on UniNDP sim/bank.py implementation.
    """
    
    def __init__(self, bank_id: int, timing: Optional[DRAMTiming] = None):
        """
        Initialize a DRAM bank.
        
        Args:
            bank_id: Unique identifier for this bank
            timing: DRAM timing parameters (uses defaults if not provided)
        """
        self.bank_id = bank_id
        self.timing = timing or DRAMTiming()
        
        # State
        self.state = BankState.IDLE
        self.open_row: Optional[int] = None
        
        # Timing state tracking
        self.timing_state = BankTimingState()
        
        # Statistics
        self.stats = {
            'row_hits': 0,
            'row_misses': 0,
            'row_empty_access': 0,
            'total_cycles': 0,
        }
    
    def check_access(self, target_row: int, is_write: bool = False) -> Tuple[int, int]:
        """
        Check timing for accessing a row.
        
        Returns:
            Tuple of (earliest_issue_cycle, access_latency)
            
        Based on UniNDP bank.check_inst() logic.
        """
        if self.state == BankState.IDLE:
            # Bank is idle, need ACT command
            first_cmd_delay = self.timing_state.nxt_act
            following_cmd_delay = self.timing.tRCD
            access_type = 'empty'
        else:
            # Bank has open row
            if self.open_row == target_row:
                # ROW HIT - can directly read/write
                if is_write:
                    first_cmd_delay = self.timing_state.nxt_write
                else:
                    first_cmd_delay = self.timing_state.nxt_read
                following_cmd_delay = 0
                access_type = 'hit'
            else:
                # ROW MISS - need PRE + ACT
                first_cmd_delay = self.timing_state.nxt_pre
                following_cmd_delay = self.timing.tRP + self.timing.tRCD
                access_type = 'miss'
        
        return first_cmd_delay, following_cmd_delay, access_type
    
    def issue_access(self, current_cycle: int, target_row: int, 
                     is_write: bool = False) -> Tuple[int, str]:
        """
        Issue an access to this bank.
        
        Args:
            current_cycle: Current simulation cycle
            target_row: Row address to access
            is_write: True for write, False for read
            
        Returns:
            Tuple of (completion_cycle, access_type)
            access_type is one of: 'hit', 'miss', 'empty'
            
        Based on UniNDP bank.issue_inst() logic.
        """
        first_cmd_delay, following_cmd_delay, access_type = self.check_access(
            target_row, is_write
        )
        
        # Determine when access starts and completes
        issue_cycle = max(current_cycle, first_cmd_delay)
        
        if access_type == 'empty':
            # Need ACT command
            act_cycle = issue_cycle
            col_cycle = act_cycle + self.timing.tRCD
        elif access_type == 'miss':
            # Need PRE + ACT commands
            pre_cycle = issue_cycle
            act_cycle = pre_cycle + self.timing.tRP
            col_cycle = act_cycle + self.timing.tRCD
        else:  # hit
            col_cycle = issue_cycle
            act_cycle = None
        
        # Calculate completion cycle
        if is_write:
            completion_cycle = col_cycle + self.timing.tCWL + self.timing.tBL
        else:
            completion_cycle = col_cycle + self.timing.tCL + self.timing.tBL
        
        # Update timing state
        self._update_timing_state(issue_cycle, target_row, is_write, access_type)
        
        # Update state
        self.state = BankState.ROWOPEN
        self.open_row = target_row
        
        # Update statistics
        if access_type == 'hit':
            self.stats['row_hits'] += 1
        elif access_type == 'miss':
            self.stats['row_misses'] += 1
        else:
            self.stats['row_empty_access'] += 1
        
        self.stats['total_cycles'] = max(self.stats['total_cycles'], completion_cycle)
        
        return completion_cycle, access_type
    
    def _update_timing_state(self, issue_cycle: int, target_row: int,
                             is_write: bool, access_type: str) -> None:
        """
        Update timing constraints after issuing an access.
        
        Based on UniNDP bank.issue_inst() timing updates.
        """
        timing = self.timing
        
        if access_type == 'empty':
            # After ACT
            act_cycle = issue_cycle
            self.timing_state.nxt_act = act_cycle + timing.tRC
            self.timing_state.nxt_pre = act_cycle + timing.tRAS
            col_cycle = act_cycle + timing.tRCD
            
        elif access_type == 'miss':
            # After PRE + ACT
            pre_cycle = issue_cycle
            act_cycle = pre_cycle + timing.tRP
            self.timing_state.nxt_act = act_cycle + timing.tRC
            self.timing_state.nxt_pre = act_cycle + timing.tRAS
            col_cycle = act_cycle + timing.tRCD
            
        else:  # hit
            col_cycle = issue_cycle
        
        # After RD/WR command
        if is_write:
            self.timing_state.nxt_write = col_cycle + timing.tCCD_L
            self.timing_state.nxt_read = col_cycle + timing.tCWL + timing.tBL + timing.tWTR_L
        else:
            self.timing_state.nxt_read = col_cycle + timing.tCCD_L
            self.timing_state.nxt_write = col_cycle + timing.tCL + timing.tBL + timing.tRTW
    
    def precharge(self, current_cycle: int) -> int:
        """
        Explicitly close the open row.
        
        Returns the cycle when precharge completes.
        """
        if self.state == BankState.IDLE:
            return current_cycle
        
        pre_cycle = max(current_cycle, self.timing_state.nxt_pre)
        completion_cycle = pre_cycle + self.timing.tRP
        
        # Update state
        self.state = BankState.IDLE
        self.open_row = None
        
        # Update timing
        self.timing_state.nxt_act = completion_cycle
        
        return completion_cycle
    
    def reset(self) -> None:
        """Reset bank to initial state."""
        self.state = BankState.IDLE
        self.open_row = None
        self.timing_state = BankTimingState()
        self.stats = {
            'row_hits': 0,
            'row_misses': 0,
            'row_empty_access': 0,
            'total_cycles': 0,
        }
    
    def get_stats(self) -> dict:
        """Get bank statistics."""
        total_accesses = (self.stats['row_hits'] + self.stats['row_misses'] + 
                        self.stats['row_empty_access'])
        hit_rate = self.stats['row_hits'] / total_accesses if total_accesses > 0 else 0
        
        return {
            **self.stats,
            'total_accesses': total_accesses,
            'row_buffer_hit_rate': hit_rate,
        }


class DRAMBankGroup:
    """
    A group of DRAM banks sharing certain resources.
    
    In modern DRAM (DDR4, HBM), banks are organized into bank groups.
    Banks within the same group have stricter timing constraints.
    """
    
    def __init__(self, group_id: int, num_banks: int, 
                 timing: Optional[DRAMTiming] = None):
        """
        Initialize a bank group.
        
        Args:
            group_id: Unique identifier for this bank group
            num_banks: Number of banks in this group
            timing: DRAM timing parameters
        """
        self.group_id = group_id
        self.timing = timing or DRAMTiming()
        self.banks = [
            DRAMBank(bank_id=group_id * num_banks + i, timing=self.timing)
            for i in range(num_banks)
        ]
    
    def get_bank(self, bank_idx: int) -> DRAMBank:
        """Get a bank by its index within this group."""
        return self.banks[bank_idx]
    
    def reset(self) -> None:
        """Reset all banks."""
        for bank in self.banks:
            bank.reset()
    
    def get_stats(self) -> dict:
        """Get aggregated statistics for all banks."""
        total_stats = {
            'row_hits': 0,
            'row_misses': 0,
            'row_empty_access': 0,
            'total_cycles': 0,
        }
        for bank in self.banks:
            stats = bank.get_stats()
            for key in total_stats:
                if key == 'total_cycles':
                    total_stats[key] = max(total_stats[key], stats[key])
                else:
                    total_stats[key] += stats[key]
        
        total_accesses = (total_stats['row_hits'] + total_stats['row_misses'] + 
                         total_stats['row_empty_access'])
        total_stats['total_accesses'] = total_accesses
        total_stats['row_buffer_hit_rate'] = (
            total_stats['row_hits'] / total_accesses if total_accesses > 0 else 0
        )
        
        return total_stats
