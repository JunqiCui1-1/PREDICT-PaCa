from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Config:
    # modeling & data
    random_state: int = 42
    pi_clip: float = 1e-3
    weight_trunc_pct: float = 0.99
    max_cat_unique: int = 50

    # DRIFT-NAT-X improvements
    margin_thresh: float = 0.01
    overlap_thresh: float = 0.10
    weight_step: float = 0.05
    reg_pref: float = 0.01
    boot_b: int = 300

    # Feasibility
    t_r_default: Dict[str, float] = field(default_factory=lambda: {"FFX": 12.0, "GemCap": 12.0, "CRT": 6.0})
    buffer_r_default: Dict[str, float] = field(default_factory=lambda: {"FFX": 2.0, "GemCap": 2.0, "CRT": 3.0})
    mds_feas_q: float = 0.25
    penalty: float = 0.05
    estimate_durations: bool = True

    # Plots
    n_show_gantt: int = 12

    # Exclusions (by normalized name)
    user_exclude: List[str] = field(default_factory=lambda: ["mean arterial pressure", "hematocrit"])
