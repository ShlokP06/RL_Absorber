"""
compare_controllers.py
======================
Compares PPO-LSTM real-time controller vs three-loop PID controller
on the MEA CCU absorber-stripper process.

Scenarios
---------
1. Step change in G_gas (load increase: 1.0 → 1.8 kg/m²/s)
2. Step change in y_CO2_in (composition change: 0.10 → 0.18)
3. Combined disturbance (both simultaneously)

Output
------
- Rich matplotlib dashboard saved as results/controller_comparison.png
- CSV of all simulation data saved as results/comparison_data.csv
- Console statistics table

Usage
-----
    python compare_controllers.py
    python compare_controllers.py --scenario 1
    python compare_controllers.py --no-plot
"""

import argparse
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.surrogate import SurrogatePredictor
from src.simulation import max_safe_L, flood_fraction

# ── Colours ───────────────────────────────────────────────────────────────────
C_RL  = "#2196F3"   # blue
C_PID = "#F44336"   # red
C_DIST= "#FF9800"   # orange
C_BG  = "#F8F9FA"
C_GRID= "#E0E0E0"
C_LINE= "#333333"


# ═════════════════════════════════════════════════════════════════════════════
# PID CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PIDController:
    """
    Independent PID loop for one controlled variable.
    Uses anti-windup on the integral term.
    """
    Kp: float
    Ki: float
    Kd: float
    setpoint: float
    out_lo: float
    out_hi: float
    integral: float = 0.0
    prev_error: float = 0.0
    integral_limit: float = 10.0

    def step(self, measurement: float) -> float:
        error   = self.setpoint - measurement
        self.integral = np.clip(
            self.integral + error, -self.integral_limit, self.integral_limit)
        derivative    = error - self.prev_error
        self.prev_error = error
        out = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return float(np.clip(out, self.out_lo, self.out_hi))

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0


class ThreeLoopPID:
    """
    Three independent PID loops controlling L_liq, alpha_lean, T_L_in.
    T_ic is held at a fixed nominal value (PIDs don't optimise intercooling).

    Tuning philosophy:
      L_liq     — fast loop, responds to capture rate error
      alpha_lean — slow loop, responds to energy and capture jointly
      T_L_in    — medium loop, responds to capture rate error
    """

    # Nominal operating point
    L_NOM     = 5.0
    AL_NOM    = 0.27
    T_NOM     = 40.0
    IC_NOM    = 38.0

    # Capture rate setpoint
    CAP_SP    = 90.0

    def __init__(self):
        # L_liq PID: increases solvent flow when capture drops
        self.pid_L = PIDController(
            Kp=0.08, Ki=0.015, Kd=0.02,
            setpoint=self.CAP_SP,
            out_lo=2.0, out_hi=12.0,
            integral_limit=15.0,
        )
        # alpha_lean PID: leans solvent when capture is low
        self.pid_al = PIDController(
            Kp=-0.002, Ki=-0.0005, Kd=-0.0005,
            setpoint=self.CAP_SP,
            out_lo=0.18, out_hi=0.38,
            integral_limit=10.0,
        )
        # T_L_in PID: cools lean solvent when capture drops
        self.pid_T = PIDController(
            Kp=-0.15, Ki=-0.03, Kd=-0.05,
            setpoint=self.CAP_SP,
            out_lo=30.0, out_hi=55.0,
            integral_limit=20.0,
        )

        # State
        self.L_liq      = self.L_NOM
        self.alpha_lean = self.AL_NOM
        self.T_L_in     = self.T_NOM
        self.T_ic       = self.IC_NOM

        # First-order actuator lag (same as RL agent for fair comparison)
        self.TAU_L  = 3.0
        self.TAU_AL = 5.0
        self.TAU_T  = 2.0

        # Actual values (lag-filtered)
        self.L_act  = self.L_NOM
        self.al_act = self.AL_NOM
        self.T_act  = self.T_NOM

    def reset(self):
        self.pid_L.reset()
        self.pid_al.reset()
        self.pid_T.reset()
        self.L_liq = self.L_NOM; self.alpha_lean = self.AL_NOM
        self.T_L_in = self.T_NOM; self.T_ic = self.IC_NOM
        self.L_act = self.L_NOM; self.al_act = self.AL_NOM
        self.T_act = self.T_NOM

    def step(self, capture: float, G_gas: float) -> Dict:
        # PID commands
        L_cmd  = self.pid_L.step(capture)
        al_cmd = self.pid_al.step(capture)
        T_cmd  = self.pid_T.step(capture)

        # Hard flood constraint (same as RL)
        T_K   = self.T_act + 273.15
        L_max = max_safe_L(G_gas, T_K, self.al_act, limit=0.79)
        L_cmd = float(np.clip(L_cmd, 2.0, min(L_max, 12.0)))

        # Actuator lag
        self.L_act  += (1/self.TAU_L)  * (L_cmd  - self.L_act)
        self.al_act += (1/self.TAU_AL) * (al_cmd - self.al_act)
        self.T_act  += (1/self.TAU_T)  * (T_cmd  - self.T_act)

        self.L_act  = float(np.clip(self.L_act,  2.0,  12.0))
        self.al_act = float(np.clip(self.al_act, 0.18, 0.38))
        self.T_act  = float(np.clip(self.T_act,  30.0, 55.0))

        return dict(L_liq=self.L_act, alpha_lean=self.al_act,
                    T_L_in=self.T_act, T_ic=self.T_ic)


# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_scenario(surrogate: SurrogatePredictor,
                 rl_model,
                 scenario: Dict,
                 n_steps: int = 60,
                 lam: float = 0.05) -> Dict:
    """
    Run one disturbance scenario for both controllers.
    Returns dict of time-series arrays for plotting.
    """
    from src.env import CCUEnv, CTRL, TAU
    import gymnasium as gym

    disturbance_step = scenario["disturbance_step"]
    G_init  = scenario["G_init"]
    G_final = scenario["G_final"]
    y_init  = scenario["y_init"]
    y_final = scenario["y_final"]

    # ── Initial steady-state ──────────────────────────────────────────────────
    L0  = 5.0; al0 = 0.27; T0 = 40.0; ic0 = 38.0
    r0  = surrogate.predict(G_init, L0, y_init, T0, al0, ic0)
    cap0 = r0["capture_rate"]
    eng0 = r0["E_specific_GJ"]

    # ── Storage ───────────────────────────────────────────────────────────────
    def empty(): return np.zeros(n_steps)
    data = {
        "G_gas"       : empty(), "y_CO2"      : empty(),
        "rl_capture"  : empty(), "rl_energy"  : empty(),
        "rl_L"        : empty(), "rl_alpha"   : empty(),
        "rl_T"        : empty(), "rl_Tic"     : empty(),
        "rl_flood"    : empty(),
        "pid_capture" : empty(), "pid_energy" : empty(),
        "pid_L"       : empty(), "pid_alpha"  : empty(),
        "pid_T"       : empty(), "pid_flood"  : empty(),
    }

    # ── RL setup ──────────────────────────────────────────────────────────────
    rl_env = CCUEnv(
        model_path="models/surrogate/model.pt",
        scaler_path="models/surrogate/scalers.pkl",
        max_steps=n_steps + 5,
        lambda_range=(lam, lam),
        step_prob=0.0,
        obs_noise=False,
        domain_rand=False,
        continue_prob=0.0,
        curriculum_phase=0,
    )
    # Reset with fixed seed then override state
    rl_env.reset(seed=0)
    rl_env.G = G_init;  rl_env.G_mean = G_init
    rl_env.y = y_init;  rl_env.y_mean = y_init
    rl_env.L_cmd = L0;  rl_env.L_act  = L0
    rl_env.al_cmd= al0; rl_env.al_act = al0
    rl_env.T_cmd = T0;  rl_env.T_act  = T0
    rl_env.ic_cmd= ic0; rl_env.ic_act = ic0
    rl_env.lam   = lam
    rl_env.cap, rl_env.eng = cap0, eng0
    rl_env.prev_cap = cap0; rl_env.prev_eng = eng0
    rl_env.cap_int  = 0.0;  rl_env.eng_int  = 0.0
    rl_env.below    = cap0 < 90.0
    rl_env.t        = 0
    obs = rl_env._obs()
    lstm_state = None

    # ── PID setup ─────────────────────────────────────────────────────────────
    pid = ThreeLoopPID()
    pid.reset()
    pid_cap = cap0; pid_eng = eng0

    # ── Simulation loop ───────────────────────────────────────────────────────
    for t in range(n_steps):
        # Apply disturbance
        G = G_final if t >= disturbance_step else G_init
        y = y_final if t >= disturbance_step else y_init

        data["G_gas"][t] = G
        data["y_CO2"][t] = y

        # ── RL step ───────────────────────────────────────────────────────────
        # Override disturbances directly — bypass OU update
        rl_env.G = G; rl_env.G_mean = G
        rl_env.y = y; rl_env.y_mean = y
        rl_env.G_trend = 0.0; rl_env.y_trend = 0.0
        action, lstm_state = rl_model.predict(
            obs, state=lstm_state, deterministic=True)
        obs, _, _, _, info_rl = rl_env.step(action)
        # After step, re-override in case OU moved it
        rl_env.G = G; rl_env.y = y

        data["rl_capture"][t] = info_rl["capture_rate"]
        data["rl_energy"][t]  = info_rl["E_specific_GJ"]
        data["rl_L"][t]       = info_rl["L_liq"]
        data["rl_alpha"][t]   = info_rl["alpha_lean"]
        data["rl_T"][t]       = info_rl["T_L_in_C"]
        data["rl_Tic"][t]     = info_rl["T_ic_C"]
        data["rl_flood"][t]   = info_rl["flood_fraction"]

        # ── PID step ──────────────────────────────────────────────────────────
        pid_ctrl = pid.step(pid_cap, G)
        pid_r    = surrogate.predict(
            G, pid_ctrl["L_liq"], y,
            pid_ctrl["T_L_in"], pid_ctrl["alpha_lean"], pid_ctrl["T_ic"])
        pid_cap = pid_r["capture_rate"]
        pid_eng = pid_r["E_specific_GJ"]

        data["pid_capture"][t] = pid_cap
        data["pid_energy"][t]  = pid_eng
        data["pid_L"][t]       = pid_ctrl["L_liq"]
        data["pid_alpha"][t]   = pid_ctrl["alpha_lean"]
        data["pid_T"][t]       = pid_ctrl["T_L_in"]
        data["pid_flood"][t]   = flood_fraction(
            G, pid_ctrl["L_liq"], pid_ctrl["T_L_in"]+273.15,
            pid_ctrl["alpha_lean"])

    data["disturbance_step"] = disturbance_step
    data["cap0"]  = cap0
    data["eng0"]  = eng0
    return data


# ═════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_stats(data: Dict, disturbance_step: int) -> Dict:
    """Compute comparison statistics for both controllers."""
    ds = disturbance_step

    def recovery_time(arr, target, after):
        """Steps after disturbance until arr stays above target."""
        post = arr[after:]
        for i in range(len(post) - 5):
            if all(post[i:i+5] >= target):
                return i
        return len(post)

    def integral_error(arr, target, after):
        """Integrated absolute error after disturbance."""
        return float(np.sum(np.abs(arr[after:] - target)))

    stats = {}
    for name in ["rl", "pid"]:
        cap = data[f"{name}_capture"]
        eng = data[f"{name}_energy"]
        stats[name] = {
            "mean_cap_pre"  : float(cap[:ds].mean()),
            "mean_cap_post" : float(cap[ds:].mean()),
            "min_cap"       : float(cap[ds:].min()),
            "mean_eng_post" : float(eng[ds:].mean()),
            "recovery_time" : recovery_time(cap, 85.0, ds),
            "iae_capture"   : integral_error(cap, 90.0, ds),
            "iae_energy"    : integral_error(eng, data["eng0"], ds),
            "pct_above_85"  : float((cap[ds:] >= 85.0).mean() * 100),
            "pct_above_90"  : float((cap[ds:] >= 90.0).mean() * 100),
            "max_flood"     : float(data[f"{name}_flood"][ds:].max()),
        }
    return stats


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def plot_dashboard(data: Dict, stats: Dict,
                   scenario_name: str, out_path: Path):
    """
    Full 8-panel comparison dashboard.
    """
    t    = np.arange(len(data["G_gas"]))
    ds   = data["disturbance_step"]
    cap0 = data["cap0"]
    eng0 = data["eng0"]

    fig = plt.figure(figsize=(20, 14), facecolor=C_BG)
    fig.suptitle(
        f"PPO-LSTM vs PID Controller — MEA CO₂ Capture\n"
        f"Scenario: {scenario_name}",
        fontsize=16, fontweight="bold", color=C_LINE, y=0.98
    )

    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        hspace=0.45, wspace=0.35,
        top=0.93, bottom=0.07, left=0.07, right=0.97
    )

    def style_ax(ax, title, ylabel, ylim=None):
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=C_LINE, pad=6)
        ax.set_ylabel(ylabel, fontsize=9, color=C_LINE)
        ax.set_xlabel("Timestep", fontsize=9, color=C_LINE)
        ax.set_facecolor(C_BG)
        ax.grid(True, color=C_GRID, linewidth=0.8, linestyle="--")
        ax.tick_params(colors=C_LINE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        if ylim:
            ax.set_ylim(ylim)
        ax.axvline(ds, color=C_DIST, linewidth=1.5, linestyle=":",
                   label="Disturbance")

    # ── Panel 1: Disturbance ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, data["G_gas"], color=C_DIST, lw=2, label="G_gas")
    ax1.plot(t, data["y_CO2"] * 10, color="#9C27B0",
             lw=2, linestyle="--", label="y_CO₂ ×10")
    style_ax(ax1, "Disturbances", "Value")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.axvline(ds, color=C_DIST, lw=1.5, ls=":")

    # ── Panel 2: Capture rate ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:3])
    ax2.fill_between(t, 85, 100, alpha=0.08, color="green", label="Target zone")
    ax2.plot(t, data["rl_capture"],  color=C_RL,  lw=2.5, label="PPO-LSTM")
    ax2.plot(t, data["pid_capture"], color=C_PID, lw=2.5,
             linestyle="--", label="PID")
    ax2.axhline(90, color="green", lw=1, ls="--", alpha=0.6, label="90% target")
    ax2.axhline(85, color="green", lw=0.8, ls=":", alpha=0.4)
    style_ax(ax2, "CO₂ Capture Rate", "Capture Rate (%)", ylim=(20, 105))
    ax2.legend(fontsize=9, loc="lower right")

    # ── Panel 3: Energy ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(t, data["rl_energy"],  color=C_RL,  lw=2.5, label="PPO-LSTM")
    ax3.plot(t, data["pid_energy"], color=C_PID, lw=2.5, ls="--", label="PID")
    ax3.axhline(eng0, color="grey", lw=1, ls=":", alpha=0.6,
                label=f"Initial {eng0:.2f}")
    style_ax(ax3, "Specific Energy", "E (GJ/tonne CO₂)")
    ax3.legend(fontsize=8)

    # ── Panel 4: L_liq ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, data["rl_L"],   color=C_RL,  lw=2, label="PPO-LSTM")
    ax4.plot(t, data["pid_L"],  color=C_PID, lw=2, ls="--", label="PID")
    style_ax(ax4, "Solvent Flow Rate (L_liq)", "kg/m²/s")
    ax4.legend(fontsize=8)

    # ── Panel 5: alpha_lean ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, data["rl_alpha"],  color=C_RL,  lw=2, label="PPO-LSTM")
    ax5.plot(t, data["pid_alpha"], color=C_PID, lw=2, ls="--", label="PID")
    style_ax(ax5, "Lean Loading (α_lean)", "mol CO₂/mol MEA")
    ax5.legend(fontsize=8)

    # ── Panel 6: T_L_in ───────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, data["rl_T"],  color=C_RL,  lw=2, label="PPO-LSTM")
    ax6.plot(t, data["pid_T"], color=C_PID, lw=2, ls="--", label="PID")
    style_ax(ax6, "Lean Solvent Temp (T_L_in)", "°C")
    ax6.legend(fontsize=8)

    # ── Panel 7: T_ic (RL only) ───────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.plot(t, data["rl_Tic"], color=C_RL, lw=2, label="PPO-LSTM")
    ax7.axhline(38.0, color=C_PID, lw=1.5, ls="--", alpha=0.7,
                label="PID (fixed)")
    style_ax(ax7, "Intercooler Temp (T_ic) [RL only]", "°C")
    ax7.legend(fontsize=8)

    # ── Panel 8: Flood fraction ───────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 0])
    ax8.plot(t, data["rl_flood"],  color=C_RL,  lw=2, label="PPO-LSTM")
    ax8.plot(t, data["pid_flood"], color=C_PID, lw=2, ls="--", label="PID")
    ax8.axhline(0.79, color="red", lw=1.5, ls="--", alpha=0.8,
                label="Flood limit (0.79)")
    style_ax(ax8, "Flooding Fraction", "Fraction of flood velocity",
             ylim=(0, 1.1))
    ax8.legend(fontsize=8)

    # ── Panel 9: Capture error histogram ─────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 1])
    post_rl  = data["rl_capture"][ds:]
    post_pid = data["pid_capture"][ds:]
    ax9.hist(post_rl,  bins=15, alpha=0.7, color=C_RL,
             label="PPO-LSTM", density=True)
    ax9.hist(post_pid, bins=15, alpha=0.7, color=C_PID,
             label="PID", density=True)
    ax9.axvline(90, color="green", lw=1.5, ls="--", label="90% target")
    ax9.set_title("Post-Disturbance Capture Distribution",
                  fontsize=11, fontweight="bold", color=C_LINE)
    ax9.set_xlabel("Capture Rate (%)", fontsize=9)
    ax9.set_ylabel("Density", fontsize=9)
    ax9.set_facecolor(C_BG)
    ax9.legend(fontsize=8)
    ax9.grid(True, color=C_GRID, lw=0.8, ls="--")

    # ── Panel 10: Cumulative energy ───────────────────────────────────────────
    ax10 = fig.add_subplot(gs[2, 2])
    cum_rl  = np.cumsum(data["rl_energy"][ds:])
    cum_pid = np.cumsum(data["pid_energy"][ds:])
    t_post  = np.arange(len(cum_rl))
    ax10.plot(t_post, cum_rl,  color=C_RL,  lw=2.5, label="PPO-LSTM")
    ax10.plot(t_post, cum_pid, color=C_PID, lw=2.5, ls="--", label="PID")
    ax10.fill_between(t_post, cum_rl, cum_pid,
                      where=cum_rl < cum_pid, alpha=0.15, color=C_RL,
                      label="RL energy saving")
    ax10.set_title("Cumulative Energy (post-disturbance)",
                   fontsize=11, fontweight="bold", color=C_LINE)
    ax10.set_xlabel("Steps after disturbance", fontsize=9)
    ax10.set_ylabel("Cumulative GJ/t", fontsize=9)
    ax10.set_facecolor(C_BG)
    ax10.legend(fontsize=8)
    ax10.grid(True, color=C_GRID, lw=0.8, ls="--")

    # ── Panel 11: Statistics table ────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[2:4, 3])
    ax11.set_facecolor(C_BG)
    ax11.axis("off")

    rl_s  = stats["rl"]
    pid_s = stats["pid"]

    rows = [
        ["Metric", "PPO-LSTM", "PID", "Winner"],
        ["Mean capture (post)", f"{rl_s['mean_cap_post']:.1f}%",
         f"{pid_s['mean_cap_post']:.1f}%",
         "RL" if rl_s["mean_cap_post"] > pid_s["mean_cap_post"] else "PID"],
        ["Min capture (post)", f"{rl_s['min_cap']:.1f}%",
         f"{pid_s['min_cap']:.1f}%",
         "RL" if rl_s["min_cap"] > pid_s["min_cap"] else "PID"],
        ["Recovery time (steps)", str(rl_s["recovery_time"]),
         str(pid_s["recovery_time"]),
         "RL" if rl_s["recovery_time"] < pid_s["recovery_time"] else "PID"],
        ["≥85% capture", f"{rl_s['pct_above_85']:.0f}%",
         f"{pid_s['pct_above_85']:.0f}%",
         "RL" if rl_s["pct_above_85"] > pid_s["pct_above_85"] else "PID"],
        ["≥90% capture", f"{rl_s['pct_above_90']:.0f}%",
         f"{pid_s['pct_above_90']:.0f}%",
         "RL" if rl_s["pct_above_90"] > pid_s["pct_above_90"] else "PID"],
        ["Mean energy (post)", f"{rl_s['mean_eng_post']:.3f}",
         f"{pid_s['mean_eng_post']:.3f}",
         "RL" if rl_s["mean_eng_post"] < pid_s["mean_eng_post"] else "PID"],
        ["IAE capture", f"{rl_s['iae_capture']:.1f}",
         f"{pid_s['iae_capture']:.1f}",
         "RL" if rl_s["iae_capture"] < pid_s["iae_capture"] else "PID"],
        ["Max flood fraction", f"{rl_s['max_flood']:.3f}",
         f"{pid_s['max_flood']:.3f}",
         "RL" if rl_s["max_flood"] < pid_s["max_flood"] else "PID"],
    ]

    col_colours = []
    for i, row in enumerate(rows):
        if i == 0:
            col_colours.append(["#37474F"] * 4)
        else:
            winner = row[3]
            w_col  = C_RL if winner == "RL" else C_PID
            col_colours.append(
                [C_BG, C_RL+"22", C_PID+"22", w_col+"55"])

    tbl = ax11.table(
        cellText=rows,
        cellLoc="center",
        loc="center",
        cellColours=col_colours,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C_GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold", color="white")

    ax11.set_title("Performance Statistics",
                   fontsize=11, fontweight="bold", color=C_LINE, pad=10)

    # ── Panel 12: Key messages ────────────────────────────────────────────────
    ax12 = fig.add_subplot(gs[3, 0:3])
    ax12.set_facecolor("#E3F2FD")
    ax12.axis("off")

    rl_wins  = sum(1 for r in rows[1:] if r[3] == "RL")
    pid_wins = sum(1 for r in rows[1:] if r[3] == "PID")

    energy_save = pid_s["mean_eng_post"] - rl_s["mean_eng_post"]
    cap_diff    = rl_s["mean_cap_post"]  - pid_s["mean_cap_post"]
    rec_diff    = pid_s["recovery_time"] - rl_s["recovery_time"]

    messages = [
        f"🏆  PPO-LSTM wins {rl_wins}/{len(rows)-1} metrics vs PID",
        f"⚡  Energy saving: {energy_save:+.3f} GJ/tonne CO₂  "
        f"({'RL saves more' if energy_save > 0 else 'PID saves more'})",
        f"📈  Capture advantage: {cap_diff:+.1f}% mean post-disturbance",
        f"⏱   Recovery: PPO-LSTM recovers {abs(rec_diff)} steps "
        f"{'faster' if rec_diff > 0 else 'slower'} than PID",
        f"🔒  PPO-LSTM max flood fraction: {rl_s['max_flood']:.3f}  "
        f"(hard constraint active)",
        f"🎛   PPO-LSTM uses 4 control variables; PID uses 3 "
        f"(T_ic optimised by RL only)",
    ]

    for i, msg in enumerate(messages):
        ax12.text(0.02, 0.85 - i * 0.15, msg,
                  transform=ax12.transAxes,
                  fontsize=10, color=C_LINE,
                  verticalalignment="top")

    ax12.set_title("Key Findings",
                   fontsize=11, fontweight="bold", color=C_LINE,
                   loc="left", pad=6)
    for spine in ax12.spines.values():
        spine.set_edgecolor("#90CAF9")
        spine.set_linewidth(1.5)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    print(f"  Dashboard saved → {out_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    1: {
        "name"             : "Gas Flow Rate Step Change (G: 0.8 → 1.25 kg/m²/s)",
        "G_init"           : 0.8,
        "G_final"          : 1.25,
        "y_init"           : 0.13,
        "y_final"          : 0.13,
        "disturbance_step" : 15,
    },
    2: {
        "name"             : "Flue Gas Composition Step Change (y_CO₂: 0.08 → 0.16)",
        "G_init"           : 1.0,
        "G_final"          : 1.0,
        "y_init"           : 0.08,
        "y_final"          : 0.16,
        "disturbance_step" : 15,
    },
    3: {
        "name"             : "Combined Disturbance (G: 0.8→1.25  &  y_CO₂: 0.08→0.16)",
        "G_init"           : 0.8,
        "G_final"          : 1.25,
        "y_init"           : 0.08,
        "y_final"          : 0.16,
        "disturbance_step" : 15,
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="models/rl/best/best_model.zip")
    p.add_argument("--model-path",  default="models/surrogate/model.pt")
    p.add_argument("--scaler-path", default="models/surrogate/scalers.pkl")
    p.add_argument("--scenario",    type=int, default=0,
                   help="0=all, 1=G_gas step, 2=y_CO2 step, 3=combined")
    p.add_argument("--steps",       type=int, default=60)
    p.add_argument("--lam",         type=float, default=0.05)
    p.add_argument("--no-plot",     action="store_true")
    args = p.parse_args()

    Path("results").mkdir(exist_ok=True)

    print("Loading surrogate and RL model...")
    surrogate = SurrogatePredictor(args.model_path, args.scaler_path)

    from sb3_contrib import RecurrentPPO
    rl_model = RecurrentPPO.load(args.model)
    print(f"  RL model   : {args.model}")
    print(f"  Lambda     : {args.lam}")
    print(f"  Steps      : {args.steps}")

    scenarios_to_run = ([args.scenario] if args.scenario > 0
                        else list(SCENARIOS.keys()))

    all_data = {}
    for sc_id in scenarios_to_run:
        sc = SCENARIOS[sc_id]
        print(f"\n{'='*60}")
        print(f"  Scenario {sc_id}: {sc['name']}")
        print("="*60)

        data  = run_scenario(surrogate, rl_model, sc,
                             n_steps=args.steps, lam=args.lam)
        stats = compute_stats(data, sc["disturbance_step"])
        all_data[sc_id] = (data, stats, sc["name"])

        # Console stats
        print(f"\n  {'Metric':<25}  {'PPO-LSTM':>10}  {'PID':>10}")
        print("  " + "-"*50)
        rl_s, pid_s = stats["rl"], stats["pid"]
        rows = [
            ("Mean capture (post %)",   rl_s["mean_cap_post"],  pid_s["mean_cap_post"],  True),
            ("Min capture (post %)",    rl_s["min_cap"],         pid_s["min_cap"],         True),
            ("Recovery time (steps)",   rl_s["recovery_time"],   pid_s["recovery_time"],   False),
            ("≥90% capture (%)",        rl_s["pct_above_90"],    pid_s["pct_above_90"],    True),
            ("Mean energy (GJ/t)",      rl_s["mean_eng_post"],   pid_s["mean_eng_post"],   False),
            ("IAE capture",             rl_s["iae_capture"],     pid_s["iae_capture"],     False),
            ("Max flood fraction",      rl_s["max_flood"],       pid_s["max_flood"],       False),
        ]
        for label, rl_v, pid_v, higher_better in rows:
            rl_wins = (rl_v > pid_v) if higher_better else (rl_v < pid_v)
            mark = "✓" if rl_wins else " "
            print(f"  {label:<25}  {rl_v:>10.3f}  {pid_v:>10.3f}  {mark}")

        if not args.no_plot:
            out = Path(f"results/comparison_scenario_{sc_id}.png")
            plot_dashboard(data, stats, sc["name"], out)

    # Save combined CSV
    frames = []
    for sc_id, (data, stats, name) in all_data.items():
        t = np.arange(len(data["G_gas"]))
        df = pd.DataFrame({
            "scenario"    : sc_id,
            "step"        : t,
            "G_gas"       : data["G_gas"],
            "y_CO2"       : data["y_CO2"],
            "rl_capture"  : data["rl_capture"],
            "rl_energy"   : data["rl_energy"],
            "rl_L"        : data["rl_L"],
            "rl_alpha"    : data["rl_alpha"],
            "rl_T"        : data["rl_T"],
            "rl_Tic"      : data["rl_Tic"],
            "rl_flood"    : data["rl_flood"],
            "pid_capture" : data["pid_capture"],
            "pid_energy"  : data["pid_energy"],
            "pid_L"       : data["pid_L"],
            "pid_alpha"   : data["pid_alpha"],
            "pid_T"       : data["pid_T"],
            "pid_flood"   : data["pid_flood"],
        })
        frames.append(df)
    pd.concat(frames).to_csv("results/comparison_data.csv", index=False)
    print("\n  All data saved → results/comparison_data.csv")


if __name__ == "__main__":
    main()