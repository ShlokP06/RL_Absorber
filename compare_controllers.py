"""
compare_controllers.py
======================
Compares PPO-LSTM real-time controller vs three-loop PID controller
on the MEA CCU absorber-stripper process.

Scenarios
---------
1. Step change in G_gas (load increase: 0.8 -> 1.25 kg/m2/s)
2. Step change in y_CO2_in (composition change: 0.08 -> 0.16)
3. Combined disturbance (both simultaneously)

Output
------
- Rich matplotlib dashboard saved as results/comparison_scenario_N.png
- CSV of all simulation data saved as results/comparison_data.csv
- Console statistics table

Usage
-----
    python compare_controllers.py
    python compare_controllers.py --scenario 1
    python compare_controllers.py --no-plot
    python compare_controllers.py --lam-max 0.10 --lam 0.05
"""

import argparse
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import pandas as pd

from src.surrogate import SurrogatePredictor
from src.simulation import max_safe_L, flood_fraction

log = logging.getLogger(__name__)

# ── Colours ───────────────────────────────────────────────────────────────────
C_RL   = "#2196F3"   # blue
C_PID  = "#F44336"   # red
C_DIST = "#FF9800"   # orange
C_BG   = "#F8F9FA"
C_GRID = "#E0E0E0"
C_LINE = "#333333"


# =============================================================================
# PID CONTROLLER
# =============================================================================

@dataclass
class PIDController:
    """
    Independent PID loop for one controlled variable.
    Uses anti-windup on the integral term.
    Output = bias + Kp*error + Ki*integral + Kd*derivative,
    where bias is the nominal operating point.
    """
    Kp: float
    Ki: float
    Kd: float
    setpoint: float
    out_lo: float
    out_hi: float
    bias: float = 0.0
    integral: float = 0.0
    prev_error: float = 0.0
    integral_limit: float = 10.0

    def step(self, measurement: float) -> float:
        error = self.setpoint - measurement
        self.integral = np.clip(
            self.integral + error, -self.integral_limit, self.integral_limit)
        derivative = error - self.prev_error
        self.prev_error = error
        out = self.bias + self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return float(np.clip(out, self.out_lo, self.out_hi))

    def reset(self) -> None:
        self.integral   = 0.0
        self.prev_error = 0.0


class ThreeLoopPID:
    """
    Three independent PID loops controlling L_liq, alpha_lean, T_L_in.
    T_ic is held at a fixed nominal value (PIDs do not optimise intercooling).

    Tuning philosophy:
      L_liq      -- fast loop, responds to capture rate error
      alpha_lean -- slow loop, responds to energy and capture jointly
      T_L_in     -- medium loop, responds to capture rate error
    """

    # Nominal operating point
    L_NOM  = 5.0
    AL_NOM = 0.27
    T_NOM  = 40.0
    IC_NOM = 38.0

    # Capture rate setpoint
    CAP_SP = 90.0

    def __init__(self) -> None:
        # L_liq PID: increases solvent flow when capture drops
        self.pid_L = PIDController(
            Kp=0.40, Ki=0.06, Kd=0.10,
            setpoint=self.CAP_SP,
            out_lo=2.0, out_hi=12.0,
            bias=self.L_NOM,
            integral_limit=20.0,
        )
        # alpha_lean PID: leans solvent when capture is low
        self.pid_al = PIDController(
            Kp=-0.008, Ki=-0.002, Kd=-0.002,
            setpoint=self.CAP_SP,
            out_lo=0.18, out_hi=0.38,
            bias=self.AL_NOM,
            integral_limit=15.0,
        )
        # T_L_in PID: cools lean solvent when capture drops
        self.pid_T = PIDController(
            Kp=-0.60, Ki=-0.10, Kd=-0.15,
            setpoint=self.CAP_SP,
            out_lo=30.0, out_hi=55.0,
            bias=self.T_NOM,
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

    def reset(self) -> None:
        self.pid_L.reset()
        self.pid_al.reset()
        self.pid_T.reset()
        self.L_liq      = self.L_NOM
        self.alpha_lean = self.AL_NOM
        self.T_L_in     = self.T_NOM
        self.T_ic       = self.IC_NOM
        self.L_act      = self.L_NOM
        self.al_act     = self.AL_NOM
        self.T_act      = self.T_NOM

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
        self.L_act  += (1 / self.TAU_L)  * (L_cmd  - self.L_act)
        self.al_act += (1 / self.TAU_AL) * (al_cmd - self.al_act)
        self.T_act  += (1 / self.TAU_T)  * (T_cmd  - self.T_act)

        self.L_act  = float(np.clip(self.L_act,  2.0,  12.0))
        self.al_act = float(np.clip(self.al_act, 0.18, 0.38))
        self.T_act  = float(np.clip(self.T_act,  30.0, 55.0))

        return dict(
            L_liq=self.L_act, alpha_lean=self.al_act,
            T_L_in=self.T_act, T_ic=self.T_ic,
        )

def _normalize_obs(norm_env, obs: np.ndarray) -> np.ndarray:
    """Normalize a raw CCUEnv observation using VecNormalize running stats.

    The RL model was trained with VecNormalize, so raw observations must be
    normalized before prediction. This applies the same clip/standardize
    transform without updating the running statistics.

    Args:
        norm_env: Loaded VecNormalize environment, or None for no normalization.
        obs: Raw 17-dim observation array from CCUEnv.

    Returns:
        Normalized observation as float32 array.
    """
    if norm_env is None:
        return obs
    obs_rms = norm_env.obs_rms
    normalized = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + norm_env.epsilon)
    return np.clip(normalized, -norm_env.clip_obs, norm_env.clip_obs).astype(np.float32)

def run_scenario(
    surrogate: SurrogatePredictor,
    rl_model,
    scenario: Dict,
    n_steps: int = 50,
    lam: float = 0.05,
    lam_max: float = 0.10,
    norm_env=None,
) -> Dict:
    """Run one disturbance scenario for both controllers.

    Args:
        surrogate: Trained surrogate predictor.
        rl_model: Loaded RecurrentPPO model.
        scenario: Dict with G_init, G_final, y_init, y_final, disturbance_step.
        n_steps: Total simulation steps.
        lam: Energy penalty weight for RL agent (goal conditioning).
        lam_max: Upper bound of lambda_range (must match training). Used to
            correctly normalize obs[15] for the policy.
        norm_env: VecNormalize environment for observation normalization, or None.

    Returns:
        Dict of time-series arrays for plotting.
    """
    from src.env import CCUEnv

    disturbance_step = scenario["disturbance_step"]
    G_init  = scenario["G_init"]
    G_final = scenario["G_final"]
    y_init  = scenario["y_init"]
    y_final = scenario["y_final"]

    L0  = 5.0; al0 = 0.27; T0 = 40.0; ic0 = 38.0
    r0   = surrogate.predict(
        G_gas=G_init, L_liq=L0, y_CO2_in=y_init,
        T_L_in_C=T0, alpha_lean=al0, T_ic_C=ic0,
    )
    cap0 = r0["capture_rate"]
    eng0 = r0["E_specific_GJ"]

    def empty() -> np.ndarray:
        return np.zeros(n_steps)

    data: Dict[str, object] = {
        "G_gas"       : empty(), "y_CO2"      : empty(),
        "rl_capture"  : empty(), "rl_energy"  : empty(),
        "rl_L"        : empty(), "rl_alpha"   : empty(),
        "rl_T"        : empty(), "rl_Tic"     : empty(),
        "rl_flood"    : empty(), "rl_reward"  : empty(),
        "pid_capture" : empty(), "pid_energy" : empty(),
        "pid_L"       : empty(), "pid_alpha"  : empty(),
        "pid_T"       : empty(), "pid_Tic"    : empty(),
        "pid_flood"   : empty(),
    }

    rl_env = CCUEnv(
        model_path="models/surrogate/model.pt",
        scaler_path="models/surrogate/scalers.pkl",
        max_steps=n_steps + 5,
        lambda_range=(0.0, lam_max),
        step_prob=0.0,
        obs_noise=False,
        domain_rand=False,
        continue_prob=0.0,
        curriculum_phase=0,
    )
    rl_env.reset(seed=0)
    rl_env.G       = G_init;  rl_env.G_mean = G_init
    rl_env.y       = y_init;  rl_env.y_mean = y_init
    rl_env.L_cmd   = L0;      rl_env.L_act  = L0
    rl_env.al_cmd  = al0;     rl_env.al_act = al0
    rl_env.T_cmd   = T0;      rl_env.T_act  = T0
    rl_env.ic_cmd  = ic0;     rl_env.ic_act = ic0
    rl_env.lam     = lam
    rl_env.cap, rl_env.eng = cap0, eng0
    rl_env.prev_cap = cap0; rl_env.prev_eng = eng0
    rl_env.cap_int  = 0.0;  rl_env.eng_int  = 0.0
    rl_env.t        = 0

    obs = _normalize_obs(norm_env, rl_env._obs())
    lstm_state    = None
    episode_start = np.array([True])
    prev_rl_action = np.zeros(4, np.float32)
    pid = ThreeLoopPID()
    pid.reset()
    pid_cap = cap0
    pid_eng = eng0

    for t in range(n_steps):
        G = G_final if t >= disturbance_step else G_init
        y = y_final if t >= disturbance_step else y_init

        data["G_gas"][t] = G
        data["y_CO2"][t] = y

        rl_env.G = G; rl_env.G_mean = G
        rl_env.y = y; rl_env.y_mean = y
        rl_env.G_trend = 0.0; rl_env.y_trend = 0.0

        action, lstm_state = rl_model.predict(
            obs, state=lstm_state,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start = np.array([False])

        raw_obs, _, _, _, info_rl = rl_env.step(action)
        obs = _normalize_obs(norm_env, raw_obs)
        rl_env.G = G; rl_env.y = y
        cap_n = info_rl["capture_rate"] / 100.0
        above = rl_env.lam_above * max(0.0, info_rl["capture_rate"] - 85.0) / 15.0
        eng_pen = lam * (info_rl["E_specific_GJ"] - 3.5) / 3.0
        da2 = float(np.mean((action - prev_rl_action) ** 2))
        rl_reward = cap_n ** 2 + above - eng_pen - rl_env.lam_smooth * da2
        prev_rl_action = action.copy()
        data["rl_capture"][t] = info_rl["capture_rate"]
        data["rl_energy"][t]  = info_rl["E_specific_GJ"]
        data["rl_L"][t]       = info_rl["L_liq"]
        data["rl_alpha"][t]   = info_rl["alpha_lean"]
        data["rl_T"][t]       = info_rl["T_L_in_C"]
        data["rl_Tic"][t]     = info_rl["T_ic_C"]
        data["rl_flood"][t]   = info_rl["flood_fraction"]
        data["rl_reward"][t]  = rl_reward

        pid_ctrl = pid.step(pid_cap, G)
        pid_r = surrogate.predict(
            G_gas=G,
            L_liq=pid_ctrl["L_liq"],
            y_CO2_in=y,
            T_L_in_C=pid_ctrl["T_L_in"],
            alpha_lean=pid_ctrl["alpha_lean"],
            T_ic_C=pid_ctrl["T_ic"],
        )
        pid_cap = pid_r["capture_rate"]
        pid_eng = pid_r["E_specific_GJ"]

        data["pid_capture"][t] = pid_cap
        data["pid_energy"][t]  = pid_eng
        data["pid_L"][t]       = pid_ctrl["L_liq"]
        data["pid_alpha"][t]   = pid_ctrl["alpha_lean"]
        data["pid_T"][t]       = pid_ctrl["T_L_in"]
        data["pid_Tic"][t]     = pid_ctrl["T_ic"]
        data["pid_flood"][t]   = flood_fraction(
            G, pid_ctrl["L_liq"],
            pid_ctrl["T_L_in"] + 273.15,
            pid_ctrl["alpha_lean"],
        )

    data["disturbance_step"] = disturbance_step
    data["cap0"]  = cap0
    data["eng0"]  = eng0
    return data

def compute_stats(data: Dict, disturbance_step: int) -> Dict:
    """Compute comparison statistics for both controllers.

    Args:
        data: Time-series dict from run_scenario().
        disturbance_step: Step index at which disturbance was applied.

    Returns:
        Nested dict: stats["rl"] and stats["pid"] each containing scalar metrics.
    """
    ds = disturbance_step

    def recovery_time(arr: np.ndarray, target: float, after: int) -> int:
        """Steps after disturbance until arr stays above target for 5 steps."""
        post = arr[after:]
        for i in range(len(post) - 5):
            if all(post[i:i + 5] >= target):
                return i
        return len(post)

    def integral_absolute_error(arr: np.ndarray, target: float, after: int) -> float:
        """Integrated absolute error after disturbance."""
        return float(np.sum(np.abs(arr[after:] - target)))

    stats = {}
    for name in ["rl", "pid"]:
        cap = data[f"{name}_capture"]
        eng = data[f"{name}_energy"]
        stats[name] = {
            "mean_cap_pre"   : float(cap[:ds].mean()),
            "mean_cap_post"  : float(cap[ds:].mean()),
            "std_cap_post"   : float(cap[ds:].std()),
            "min_cap"        : float(cap[ds:].min()),
            "mean_eng_post"  : float(eng[ds:].mean()),
            "std_eng_post"   : float(eng[ds:].std()),
            "recovery_time"  : recovery_time(cap, 85.0, ds),
            "iae_capture"    : integral_absolute_error(cap, 90.0, ds),
            "iae_energy"     : integral_absolute_error(eng, data["eng0"], ds),
            "pct_above_85"   : float((cap[ds:] >= 85.0).mean() * 100),
            "pct_above_90"   : float((cap[ds:] >= 90.0).mean() * 100),
            "max_flood"      : float(data[f"{name}_flood"][ds:].max()),
            "mean_flood"     : float(data[f"{name}_flood"].mean()),
            "pareto_score"   : float(cap[ds:].mean() - 5.0 * eng[ds:].mean()),
        }
    return stats

def plot_dashboard(
    data: Dict,
    stats: Dict,
    scenario_name: str,
    out_path: Path,
) -> plt.Figure:
    """Full 16-panel comparison dashboard.

    Layout:
        Row 0: Disturbances | Capture rate (x2) | Specific energy
        Row 1: L_liq | alpha_lean | T_L_in | T_ic
        Row 2: Flood fraction | Capture histogram | Energy histogram | Pareto scatter
        Row 3: Cumulative energy | Capture deviation | IAE bar chart | Stats table (x1)
        Row 4: Key findings (x3) | (stats table continued)
    """
    t    = np.arange(len(data["G_gas"]))
    ds   = data["disturbance_step"]
    cap0 = data["cap0"]
    eng0 = data["eng0"]

    fig = plt.figure(figsize=(22, 18), facecolor=C_BG)
    fig.suptitle(
        f"PPO-LSTM vs PID Controller -- MEA CO2 Capture\n"
        f"Scenario: {scenario_name}",
        fontsize=15, fontweight="bold", color=C_LINE, y=0.99,
    )

    gs = gridspec.GridSpec(
        5, 4, figure=fig,
        hspace=0.50, wspace=0.35,
        top=0.94, bottom=0.05, left=0.07, right=0.97,
    )

    def style_ax(
        ax: plt.Axes,
        title: str,
        ylabel: str,
        ylim: Optional[tuple] = None,
        add_vline: bool = True,
    ) -> None:
        ax.set_title(title, fontsize=10, fontweight="bold", color=C_LINE, pad=5)
        ax.set_ylabel(ylabel, fontsize=8, color=C_LINE)
        ax.set_xlabel("Timestep", fontsize=8, color=C_LINE)
        ax.set_facecolor(C_BG)
        ax.grid(True, color=C_GRID, linewidth=0.7, linestyle="--")
        ax.tick_params(colors=C_LINE, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        if ylim:
            ax.set_ylim(ylim)
        if add_vline:
            ax.axvline(ds, color=C_DIST, linewidth=1.5, linestyle=":",
                       label="Disturbance")

    # ── Panel 1: Disturbances (row 0, col 0) ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, data["G_gas"], color=C_DIST, lw=2, label="G_gas")
    ax1.plot(t, data["y_CO2"] * 10, color="#9C27B0",
             lw=2, linestyle="--", label="y_CO2 x10")
    style_ax(ax1, "Feed Disturbances", "Value")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.axvline(ds, color=C_DIST, lw=1.5, ls=":")

    # ── Panel 2: Capture rate (row 0, col 1-2) ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1:3])
    ax2.fill_between(t, 85, 100, alpha=0.08, color="green", label="Target zone (85%+)")
    ax2.plot(t, data["rl_capture"],  color=C_RL,  lw=2.5, label="PPO-LSTM")
    ax2.plot(t, data["pid_capture"], color=C_PID, lw=2.5,
             linestyle="--", label="PID")
    ax2.axhline(90, color="green", lw=1.2, ls="--", alpha=0.7, label="90% setpoint")
    ax2.axhline(85, color="green", lw=0.8, ls=":",  alpha=0.4)
    style_ax(ax2, "CO2 Capture Rate", "Capture Rate (%)", ylim=(20, 105))
    ax2.legend(fontsize=8, loc="lower right")

    # ── Panel 3: Specific energy (row 0, col 3) ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(t, data["rl_energy"],  color=C_RL,  lw=2.5, label="PPO-LSTM")
    ax3.plot(t, data["pid_energy"], color=C_PID, lw=2.5, ls="--", label="PID")
    ax3.axhline(eng0, color="grey", lw=1, ls=":", alpha=0.6,
                label=f"Init {eng0:.2f}")
    style_ax(ax3, "Specific Energy", "E (GJ/tonne CO2)")
    ax3.legend(fontsize=7)

    # ── Panel 4: L_liq (row 1, col 0) ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, data["rl_L"],  color=C_RL,  lw=2, label="PPO-LSTM")
    ax4.plot(t, data["pid_L"], color=C_PID, lw=2, ls="--", label="PID")
    ax4.axhline(2.0,  color="grey", lw=0.8, ls=":", alpha=0.5)
    ax4.axhline(12.0, color="grey", lw=0.8, ls=":", alpha=0.5)
    style_ax(ax4, "Solvent Flow (L_liq)", "kg/m2/s", ylim=(1.5, 12.5))
    ax4.legend(fontsize=7)

    # ── Panel 5: alpha_lean (row 1, col 1) ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, data["rl_alpha"],  color=C_RL,  lw=2, label="PPO-LSTM")
    ax5.plot(t, data["pid_alpha"], color=C_PID, lw=2, ls="--", label="PID")
    style_ax(ax5, "Lean Loading (alpha_lean)", "mol CO2/mol MEA")
    ax5.legend(fontsize=7)

    # ── Panel 6: T_L_in (row 1, col 2) ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, data["rl_T"],  color=C_RL,  lw=2, label="PPO-LSTM")
    ax6.plot(t, data["pid_T"], color=C_PID, lw=2, ls="--", label="PID")
    style_ax(ax6, "Lean Solvent Temp (T_L_in)", "deg C")
    ax6.legend(fontsize=7)

    # ── Panel 7: T_ic (row 1, col 3) ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.plot(t, data["rl_Tic"],  color=C_RL, lw=2, label="PPO-LSTM")
    ax7.plot(t, data["pid_Tic"], color=C_PID, lw=1.5, ls="--",
             alpha=0.7, label=f"PID (fixed {ThreeLoopPID.IC_NOM:.0f} C)")
    style_ax(ax7, "Intercooler Temp (T_ic)", "deg C")
    ax7.legend(fontsize=7)

    # ── Panel 8: Flood fraction (row 2, col 0) ───────────────────────────────
    ax8 = fig.add_subplot(gs[2, 0])
    ax8.fill_between(t, 0.75, 0.79, alpha=0.12, color="red", label="Danger zone")
    ax8.plot(t, data["rl_flood"],  color=C_RL,  lw=2, label="PPO-LSTM")
    ax8.plot(t, data["pid_flood"], color=C_PID, lw=2, ls="--", label="PID")
    ax8.axhline(0.79, color="red", lw=1.5, ls="--", alpha=0.8, label="Flood limit")
    ax8.axhline(0.75, color="orange", lw=1, ls=":", alpha=0.6)
    style_ax(ax8, "Flooding Fraction", "Fraction of flood velocity", ylim=(0, 1.05))
    ax8.legend(fontsize=7)
    ax9 = fig.add_subplot(gs[2, 1])
    post_rl  = data["rl_capture"][ds:]
    post_pid = data["pid_capture"][ds:]
    bins = np.linspace(
        min(post_rl.min(), post_pid.min()) - 2,
        max(post_rl.max(), post_pid.max()) + 2,
        18,
    )
    ax9.hist(post_rl,  bins=bins, alpha=0.65, color=C_RL,  label="PPO-LSTM", density=True)
    ax9.hist(post_pid, bins=bins, alpha=0.65, color=C_PID, label="PID",       density=True)
    ax9.axvline(90, color="green", lw=1.5, ls="--", label="90% setpoint")
    ax9.axvline(post_rl.mean(),  color=C_RL,  lw=1.5, ls="-", alpha=0.8)
    ax9.axvline(post_pid.mean(), color=C_PID, lw=1.5, ls="-", alpha=0.8)
    ax9.set_title("Capture Distribution (post-dist.)", fontsize=10,
                  fontweight="bold", color=C_LINE)
    ax9.set_xlabel("Capture Rate (%)", fontsize=8)
    ax9.set_ylabel("Density", fontsize=8)
    ax9.set_facecolor(C_BG)
    ax9.legend(fontsize=7)
    ax9.grid(True, color=C_GRID, lw=0.7, ls="--")

    ax10 = fig.add_subplot(gs[2, 2])
    eng_rl  = data["rl_energy"][ds:]
    eng_pid = data["pid_energy"][ds:]
    ebins = np.linspace(
        min(eng_rl.min(), eng_pid.min()) - 0.1,
        max(eng_rl.max(), eng_pid.max()) + 0.1,
        18,
    )
    ax10.hist(eng_rl,  bins=ebins, alpha=0.65, color=C_RL,  label="PPO-LSTM", density=True)
    ax10.hist(eng_pid, bins=ebins, alpha=0.65, color=C_PID, label="PID",       density=True)
    ax10.axvline(eng0, color="grey", lw=1.5, ls="--", label=f"Init {eng0:.2f}")
    ax10.axvline(eng_rl.mean(),  color=C_RL,  lw=1.5, ls="-", alpha=0.8)
    ax10.axvline(eng_pid.mean(), color=C_PID, lw=1.5, ls="-", alpha=0.8)
    ax10.set_title("Energy Distribution (post-dist.)", fontsize=10,
                   fontweight="bold", color=C_LINE)
    ax10.set_xlabel("Specific Energy (GJ/tonne CO2)", fontsize=8)
    ax10.set_ylabel("Density", fontsize=8)
    ax10.set_facecolor(C_BG)
    ax10.legend(fontsize=7)
    ax10.grid(True, color=C_GRID, lw=0.7, ls="--")

    ax11 = fig.add_subplot(gs[2, 3])
    pre_mask  = t < ds
    post_mask = t >= ds
    ax11.scatter(data["pid_energy"][pre_mask],  data["pid_capture"][pre_mask],
                 c=C_PID, alpha=0.3, s=20, marker="s", label="PID (pre)")
    ax11.scatter(data["pid_energy"][post_mask], data["pid_capture"][post_mask],
                 c=C_PID, alpha=0.7, s=30, marker="s", label="PID (post)")
    ax11.scatter(data["rl_energy"][pre_mask],   data["rl_capture"][pre_mask],
                 c=C_RL,  alpha=0.3, s=20, marker="o", label="RL (pre)")
    ax11.scatter(data["rl_energy"][post_mask],  data["rl_capture"][post_mask],
                 c=C_RL,  alpha=0.7, s=30, marker="o", label="RL (post)")
    ax11.axhline(90, color="green", lw=1, ls="--", alpha=0.6)
    ax11.set_xlabel("Energy (GJ/tonne CO2)", fontsize=8)
    ax11.set_ylabel("Capture Rate (%)", fontsize=8)
    ax11.set_title("Pareto: Capture vs Energy", fontsize=10,
                   fontweight="bold", color=C_LINE)
    ax11.set_facecolor(C_BG)
    ax11.legend(fontsize=6, loc="lower right", ncol=2)
    ax11.grid(True, color=C_GRID, lw=0.7, ls="--")
    for spine in ax11.spines.values():
        spine.set_edgecolor(C_GRID)

    ax12 = fig.add_subplot(gs[3, 0])
    cum_rl  = np.cumsum(data["rl_energy"][ds:])
    cum_pid = np.cumsum(data["pid_energy"][ds:])
    t_post  = np.arange(len(cum_rl))
    ax12.plot(t_post, cum_rl,  color=C_RL,  lw=2.5, label="PPO-LSTM")
    ax12.plot(t_post, cum_pid, color=C_PID, lw=2.5, ls="--", label="PID")
    ax12.fill_between(t_post, cum_rl, cum_pid,
                      where=cum_rl < cum_pid, alpha=0.15, color=C_RL,
                      label="RL energy saving")
    ax12.fill_between(t_post, cum_rl, cum_pid,
                      where=cum_rl > cum_pid, alpha=0.15, color=C_PID,
                      label="PID energy saving")
    ax12.set_title("Cumulative Energy (post-dist.)", fontsize=10,
                   fontweight="bold", color=C_LINE)
    ax12.set_xlabel("Steps after disturbance", fontsize=8)
    ax12.set_ylabel("Cumulative GJ/tonne", fontsize=8)
    ax12.set_facecolor(C_BG)
    ax12.legend(fontsize=7)
    ax12.grid(True, color=C_GRID, lw=0.7, ls="--")
    for spine in ax12.spines.values():
        spine.set_edgecolor(C_GRID)

    ax13 = fig.add_subplot(gs[3, 1])
    dev_rl  = np.abs(data["rl_capture"]  - 90.0)
    dev_pid = np.abs(data["pid_capture"] - 90.0)
    ax13.plot(t, dev_rl,  color=C_RL,  lw=2, label="PPO-LSTM")
    ax13.plot(t, dev_pid, color=C_PID, lw=2, ls="--", label="PID")
    ax13.fill_between(t, dev_rl, dev_pid,
                      where=dev_rl < dev_pid, alpha=0.12, color=C_RL,
                      label="RL closer to SP")
    ax13.axvline(ds, color=C_DIST, lw=1.5, ls=":", label="Disturbance")
    ax13.axhline(0, color="green", lw=0.8, ls="--", alpha=0.4)
    ax13.set_title("|Capture - 90%| Deviation", fontsize=10,
                   fontweight="bold", color=C_LINE)
    ax13.set_xlabel("Timestep", fontsize=8)
    ax13.set_ylabel("Absolute deviation (%)", fontsize=8)
    ax13.set_facecolor(C_BG)
    ax13.legend(fontsize=7)
    ax13.grid(True, color=C_GRID, lw=0.7, ls="--")
    for spine in ax13.spines.values():
        spine.set_edgecolor(C_GRID)

    ax14 = fig.add_subplot(gs[3, 2])
    rl_s  = stats["rl"]
    pid_s = stats["pid"]
    metrics_bar = [
        ("IAE\nCapture", rl_s["iae_capture"],   pid_s["iae_capture"],   True),
        ("IAE\nEnergy",  rl_s["iae_energy"],     pid_s["iae_energy"],    True),
        ("Recov.\nSteps",rl_s["recovery_time"],  pid_s["recovery_time"], True),
        ("Max\nFlood",   rl_s["max_flood"] * 100, pid_s["max_flood"] * 100, True),
    ]
    x_bar  = np.arange(len(metrics_bar))
    width  = 0.35
    labels = [m[0] for m in metrics_bar]
    rl_vals  = [m[1] for m in metrics_bar]
    pid_vals = [m[2] for m in metrics_bar]

    bars_rl  = ax14.bar(x_bar - width / 2, rl_vals,  width, color=C_RL,
                        alpha=0.8, label="PPO-LSTM")
    bars_pid = ax14.bar(x_bar + width / 2, pid_vals, width, color=C_PID,
                        alpha=0.8, label="PID")
    for i, (_, rv, pv, lower_better) in enumerate(metrics_bar):
        winner = rv < pv if lower_better else rv > pv
        if winner:
            ax14.text(i - width / 2, rv * 1.03, "Win",
                      ha="center", fontsize=6, color=C_RL, fontweight="bold")
        else:
            ax14.text(i + width / 2, pv * 1.03, "Win",
                      ha="center", fontsize=6, color=C_PID, fontweight="bold")

    ax14.set_xticks(x_bar)
    ax14.set_xticklabels(labels, fontsize=7)
    ax14.set_title("Key Metric Comparison (lower = better)", fontsize=10,
                   fontweight="bold", color=C_LINE)
    ax14.set_facecolor(C_BG)
    ax14.legend(fontsize=7)
    ax14.grid(True, axis="y", color=C_GRID, lw=0.7, ls="--")
    for spine in ax14.spines.values():
        spine.set_edgecolor(C_GRID)
    ax15 = fig.add_subplot(gs[3:5, 3])
    ax15.set_facecolor(C_BG)
    ax15.axis("off")

    rows_tbl = [
        ["Metric", "PPO-LSTM", "PID", "Winner"],
        ["Mean capture (post)", f"{rl_s['mean_cap_post']:.1f}%",
         f"{pid_s['mean_cap_post']:.1f}%",
         "RL" if rl_s["mean_cap_post"] > pid_s["mean_cap_post"] else "PID"],
        ["Min capture (post)", f"{rl_s['min_cap']:.1f}%",
         f"{pid_s['min_cap']:.1f}%",
         "RL" if rl_s["min_cap"] > pid_s["min_cap"] else "PID"],
        ["Std capture (post)", f"{rl_s['std_cap_post']:.2f}%",
         f"{pid_s['std_cap_post']:.2f}%",
         "RL" if rl_s["std_cap_post"] < pid_s["std_cap_post"] else "PID"],
        ["Recovery (steps)", str(rl_s["recovery_time"]),
         str(pid_s["recovery_time"]),
         "RL" if rl_s["recovery_time"] < pid_s["recovery_time"] else "PID"],
        [">85% capture", f"{rl_s['pct_above_85']:.0f}%",
         f"{pid_s['pct_above_85']:.0f}%",
         "RL" if rl_s["pct_above_85"] > pid_s["pct_above_85"] else "PID"],
        [">90% capture", f"{rl_s['pct_above_90']:.0f}%",
         f"{pid_s['pct_above_90']:.0f}%",
         "RL" if rl_s["pct_above_90"] > pid_s["pct_above_90"] else "PID"],
        ["Mean energy (post)", f"{rl_s['mean_eng_post']:.3f}",
         f"{pid_s['mean_eng_post']:.3f}",
         "RL" if rl_s["mean_eng_post"] < pid_s["mean_eng_post"] else "PID"],
        ["IAE capture", f"{rl_s['iae_capture']:.1f}",
         f"{pid_s['iae_capture']:.1f}",
         "RL" if rl_s["iae_capture"] < pid_s["iae_capture"] else "PID"],
        ["IAE energy", f"{rl_s['iae_energy']:.2f}",
         f"{pid_s['iae_energy']:.2f}",
         "RL" if rl_s["iae_energy"] < pid_s["iae_energy"] else "PID"],
        ["Max flood", f"{rl_s['max_flood']:.3f}",
         f"{pid_s['max_flood']:.3f}",
         "RL" if rl_s["max_flood"] < pid_s["max_flood"] else "PID"],
    ]

    col_colours = []
    for i, row in enumerate(rows_tbl):
        if i == 0:
            col_colours.append(["#37474F"] * 4)
        else:
            winner = row[3]
            w_col  = C_RL if winner == "RL" else C_PID
            col_colours.append([C_BG, C_RL + "22", C_PID + "22", w_col + "55"])

    tbl = ax15.table(
        cellText=rows_tbl,
        cellLoc="center",
        loc="center",
        cellColours=col_colours,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.45)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C_GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold", color="white")

    ax15.set_title("Performance Statistics", fontsize=10,
                   fontweight="bold", color=C_LINE, pad=8)

    ax16 = fig.add_subplot(gs[4, 0:3])
    ax16.set_facecolor("#E3F2FD")
    ax16.axis("off")

    rl_wins  = sum(1 for r in rows_tbl[1:] if r[3] == "RL")
    pid_wins = sum(1 for r in rows_tbl[1:] if r[3] == "PID")

    energy_save = pid_s["mean_eng_post"] - rl_s["mean_eng_post"]
    cap_diff    = rl_s["mean_cap_post"]  - pid_s["mean_cap_post"]
    rec_diff    = pid_s["recovery_time"] - rl_s["recovery_time"]
    iae_diff    = pid_s["iae_capture"]   - rl_s["iae_capture"]

    messages = [
        f"PPO-LSTM wins {rl_wins}/{len(rows_tbl)-1} metrics vs PID  "
        f"({'RL dominant' if rl_wins > pid_wins else 'PID competitive'})",
        f"Energy: {energy_save:+.3f} GJ/tonne CO2  "
        f"({'RL saves energy' if energy_save > 0 else 'PID saves energy'})",
        f"Capture: {cap_diff:+.1f}% mean advantage for "
        f"{'PPO-LSTM' if cap_diff > 0 else 'PID'} post-disturbance",
        f"Recovery: PPO-LSTM recovers {abs(rec_diff)} steps "
        f"{'faster' if rec_diff > 0 else 'slower'} than PID  "
        f"(IAE diff: {iae_diff:+.1f})",
        f"Safety: PPO-LSTM max flood {rl_s['max_flood']:.3f} | "
        f"PID max flood {pid_s['max_flood']:.3f}  (hard limit: 0.79)",
        f"4-DoF advantage: PPO-LSTM optimises T_ic; PID holds it fixed at "
        f"{ThreeLoopPID.IC_NOM:.0f} C",
    ]

    for i, msg in enumerate(messages):
        ax16.text(0.01, 0.88 - i * 0.16, msg,
                  transform=ax16.transAxes,
                  fontsize=9, color=C_LINE,
                  verticalalignment="top")

    ax16.set_title("Key Findings",
                   fontsize=10, fontweight="bold", color=C_LINE,
                   loc="left", pad=5)
    for spine in ax16.spines.values():
        spine.set_edgecolor("#90CAF9")
        spine.set_linewidth(1.5)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    log.info("Dashboard saved -> %s", out_path)
    print(f"  Dashboard saved -> {out_path}")
    return fig

SCENARIOS: Dict[int, Dict] = {
    1: {
        "name"             : "Gas Flow Rate Step Change (G: 0.8 -> 1.25 kg/m2/s)",
        "G_init"           : 0.8,
        "G_final"          : 1.25,
        "y_init"           : 0.13,
        "y_final"          : 0.13,
        "disturbance_step" : 15,
    },
    2: {
        "name"             : "Flue Gas Composition Step Change (y_CO2: 0.08 -> 0.16)",
        "G_init"           : 1.0,
        "G_final"          : 1.0,
        "y_init"           : 0.08,
        "y_final"          : 0.16,
        "disturbance_step" : 15,
    },
    3: {
        "name"             : "Combined Disturbance (G: 0.8->1.25 & y_CO2: 0.08->0.16)",
        "G_init"           : 0.8,
        "G_final"          : 1.25,
        "y_init"           : 0.08,
        "y_final"          : 0.16,
        "disturbance_step" : 15,
    },
}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Compare PPO-LSTM vs PID on MEA CCU scenarios."
    )
    p.add_argument("--model", default="models/rl/best/best_model.zip",
                   help="Path to trained RecurrentPPO model.")
    p.add_argument("--model-path", default="models/surrogate/model.pt",
                   help="Path to surrogate model weights.")
    p.add_argument("--scaler-path", default="models/surrogate/scalers.pkl",
                   help="Path to surrogate scalers.")
    p.add_argument("--scenario", type=int, default=0,
                   help="0=all scenarios, 1=G_gas step, 2=y_CO2 step, 3=combined.")
    p.add_argument("--steps", type=int, default=60, help="Total simulation steps per scenario.")
    p.add_argument("--lam", type=float, default=0.05, help="Energy penalty weight for RL agent (goal conditioning).")
    p.add_argument("--lam-max", type=float, default=0.10, 
                   help="Training lam_max -- must match --lam-max used in train_rl.py "
                        "(default 0.10). Controls obs[15] normalisation range.")
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib dashboard generation.")
    args = p.parse_args()
    Path("results").mkdir(exist_ok=True)
    log.info("Loading surrogate and RL model...")
    surrogate = SurrogatePredictor(args.model_path, args.scaler_path)
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from src.env import CCUEnv
    vecnorm_path = Path("models/rl/vecnorm.pkl")
    norm_env: Optional[VecNormalize] = None
    if vecnorm_path.exists():
        def _eval_env_factory():
            return Monitor(CCUEnv(
                model_path=args.model_path,
                scaler_path=args.scaler_path,
                max_steps=100,
                lambda_range=(0.0, args.lam_max),
                step_prob=0.0,
                obs_noise=False,
                domain_rand=False,
                curriculum_phase=2,
            ))
        _venv = DummyVecEnv([_eval_env_factory])
        norm_env = VecNormalize.load(str(vecnorm_path), _venv)
        norm_env.training    = False
        norm_env.norm_reward = False
        rl_model = RecurrentPPO.load(args.model, env=norm_env)
        log.info("VecNormalize loaded: %s", vecnorm_path)
    else:
        warnings.warn(
            f"{vecnorm_path} not found -- observations will not be normalised. "
            "Run training to generate vecnorm.pkl for accurate comparison.",
            stacklevel=2,
        )
        rl_model = RecurrentPPO.load(args.model)

    log.info("RL model    : %s", args.model)
    log.info("Lambda      : %s  (lam_max=%s)", args.lam, args.lam_max)
    log.info("Steps       : %s", args.steps)

    scenarios_to_run = (
        [args.scenario] if args.scenario > 0 else list(SCENARIOS.keys())
    )

    all_data: Dict[int, tuple] = {}
    for sc_id in scenarios_to_run:
        sc = SCENARIOS[sc_id]
        print(f"\n{'='*65}")
        print(f"  Scenario {sc_id}: {sc['name']}")
        print("=" * 65)

        data  = run_scenario(
            surrogate, rl_model, sc,
            n_steps=args.steps,
            lam=args.lam,
            lam_max=args.lam_max,
            norm_env=norm_env,
        )
        stats = compute_stats(data, sc["disturbance_step"])
        all_data[sc_id] = (data, stats, sc["name"])

        # Console stats table
        print(f"\n  {'Metric':<26}  {'PPO-LSTM':>10}  {'PID':>10}")
        print("  " + "-" * 52)
        rl_s, pid_s = stats["rl"], stats["pid"]
        console_rows = [
            ("Mean capture (post %)",  rl_s["mean_cap_post"],  pid_s["mean_cap_post"],  True),
            ("Min capture (post %)",   rl_s["min_cap"],         pid_s["min_cap"],         True),
            ("Std capture (post %)",   rl_s["std_cap_post"],    pid_s["std_cap_post"],    False),
            ("Recovery time (steps)",  rl_s["recovery_time"],   pid_s["recovery_time"],   False),
            (">90% capture (%)",       rl_s["pct_above_90"],    pid_s["pct_above_90"],    True),
            ("Mean energy (GJ/t)",     rl_s["mean_eng_post"],   pid_s["mean_eng_post"],   False),
            ("IAE capture",            rl_s["iae_capture"],     pid_s["iae_capture"],     False),
            ("IAE energy",             rl_s["iae_energy"],      pid_s["iae_energy"],      False),
            ("Max flood fraction",     rl_s["max_flood"],       pid_s["max_flood"],       False),
            ("Pareto score",           rl_s["pareto_score"],    pid_s["pareto_score"],    True),
        ]
        for label, rl_v, pid_v, higher_better in console_rows:
            rl_wins = (rl_v > pid_v) if higher_better else (rl_v < pid_v)
            mark = "RL" if rl_wins else "  "
            print(f"  {label:<26}  {rl_v:>10.3f}  {pid_v:>10.3f}  {mark}")

        if not args.no_plot:
            out = Path(f"results/comparison_scenario_{sc_id}.png")
            plot_dashboard(data, stats, sc["name"], out)

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
            "rl_reward"   : data["rl_reward"],
            "pid_capture" : data["pid_capture"],
            "pid_energy"  : data["pid_energy"],
            "pid_L"       : data["pid_L"],
            "pid_alpha"   : data["pid_alpha"],
            "pid_T"       : data["pid_T"],
            "pid_Tic"     : data["pid_Tic"],
            "pid_flood"   : data["pid_flood"],
        })
        frames.append(df)

    pd.concat(frames, ignore_index=True).to_csv(
        "results/comparison_data.csv", index=False)
    print("\n  All data saved -> results/comparison_data.csv")


if __name__ == "__main__":
    main()
