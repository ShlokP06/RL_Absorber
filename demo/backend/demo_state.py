"""
DemoState — single source of truth for the live RL demo.

Manages:
  - RL environment (CCUEnv via DummyVecEnv + VecNormalize)
  - RecurrentPPO model + LSTM hidden states
  - PIDSimulator running in parallel on the same disturbances
  - Manual disturbance overrides, frozen-agent mode, lambda live-tuning
  - History buffer for frontend replay / initial load
  - Impact counters (CO2, kWh, $, trees)
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from typing import Optional

# ── Project root on sys.path ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from src.env import CCUEnv
from src.surrogate import SurrogatePredictor
from pid import PIDSimulator  # noqa: E402 — same directory


# --------------------------------------------------------------------------- #
# DemoState                                                                     #
# --------------------------------------------------------------------------- #

class DemoState:

    STEP_INTERVAL: float = 0.5   # seconds per sim tick
    MAX_HISTORY:   int   = 120   # 60 s of data @ 500 ms

    # Impact scale: assume 100 m² column, ~0.3 kg CO2 /s captured at nominal
    _CO2_SCALE: float = 0.3 * 0.5 / 1000  # kg/s × dt → tonnes per step
    _KWH_SCALE: float = 0.3 * 0.5 * 277.78  # GJ/t × t/step → kWh/step

    def __init__(self, config: dict) -> None:
        self._config = config
        self._load_models(config)
        self.reset()

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_models(self, config: dict) -> None:
        root = _PROJECT_ROOT

        surrogate_path = str(root / config["surrogate_path"])
        scalers_path   = str(root / config["scalers_path"])
        model_path     = str(root / config["model_path"])
        vecnorm_path   = str(root / config["vecnorm_path"])

        # Surrogate shared by RL env and PID sim
        self.surrogate = SurrogatePredictor(surrogate_path, scalers_path)

        # RL environment wrapped in VecNormalize (eval mode)
        def _make_env() -> CCUEnv:
            return CCUEnv(
                model_path=surrogate_path,
                scaler_path=scalers_path,
                max_steps=100_000,        # effectively infinite episodes
                lambda_range=(0.0, 0.20),
                step_prob=0.0,            # user controls disturbances
                actuator_lag=True,
                obs_noise=False,
                domain_rand=False,
                continue_prob=0.0,
                curriculum_phase=2,
            )

        venv = DummyVecEnv([_make_env])
        self.rl_vec: VecNormalize = VecNormalize.load(vecnorm_path, venv)
        self.rl_vec.training    = False
        self.rl_vec.norm_reward = False

        # RecurrentPPO model (no env binding needed — spaces match)
        self.model = RecurrentPPO.load(model_path, device="cpu")

        # PID baseline
        self.pid_sim = PIDSimulator(self.surrogate)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _raw(self) -> CCUEnv:
        """Unwrap to the underlying CCUEnv."""
        return self.rl_vec.venv.envs[0]

    def _apply_overrides(self) -> None:
        """Force disturbance and lambda onto the raw env if overrides are set."""
        raw = self._raw()
        if self._manual_dist is not None:
            G, y = self._manual_dist
            raw.G = G;      raw.G_mean = G
            raw.y = y;      raw.y_mean = y
        if raw.lam is not None:
            raw.lam = self.lambda_energy

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset both simulations and all counters."""
        self.obs: np.ndarray = self.rl_vec.reset()
        self.lstm_states                   = None
        self.episode_starts: np.ndarray   = np.ones((1,), dtype=bool)
        self._frozen_lstm: Optional[tuple] = None

        self.pid_sim.reset()

        self.frozen:        bool  = False
        self._manual_dist:  Optional[tuple] = None
        self.lambda_energy: float = 0.05

        # EMA action smoothing — prevents high-frequency oscillation when agent
        # is near constraint boundaries or at high λ (edge of training distribution)
        self._smooth_action = np.zeros((1, 4), dtype=np.float32)

        # Apply initial lambda
        raw = self._raw()
        if raw.lam is not None:
            raw.lam = self.lambda_energy

        self.history: list = []
        self.t:       int  = 0

        # Impact accumulators
        self._co2_rl   = 0.0
        self._co2_pid  = 0.0
        self._kwh_rl   = 0.0
        self._kwh_pid  = 0.0

        return self._snapshot()

    def step(self) -> dict:
        """Advance both simulations by one tick and return a state snapshot."""
        self._apply_overrides()
        raw = self._raw()

        # ── RL action ──────────────────────────────────────────────────
        if not self.frozen:
            action, self.lstm_states = self.model.predict(
                self.obs,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=True,
            )
            self.episode_starts[:] = False
        else:
            action = self._frozen_action(raw)
            # LSTM state preserved but not updated when frozen

        # EMA smoothing on the command signal — α=0.55 keeps responsiveness
        # while killing the high-frequency bang-bang near constraint boundaries.
        # Real plants have this naturally (valve rate limiters, DCS setpoint ramps).
        SMOOTH = 0.55
        action = SMOOTH * action + (1.0 - SMOOTH) * self._smooth_action
        self._smooth_action = action.copy()

        # ── Step RL env ────────────────────────────────────────────────
        self.obs, _reward, dones, _infos = self.rl_vec.step(action)

        if dones[0]:
            self.episode_starts[:] = True
            if not self.frozen:
                self.lstm_states = None
            self._smooth_action[:] = 0.0

        raw = self._raw()  # re-fetch after possible reset

        # ── Snap G,y back after OU noise if manual override is active ──
        # env._step_dist() adds OU noise even when G_mean is pinned, so
        # we re-clamp here to keep the displayed value stable.
        if self._manual_dist is not None:
            raw.G      = self._manual_dist[0]
            raw.G_mean = self._manual_dist[0]
            raw.y      = self._manual_dist[1]
            raw.y_mean = self._manual_dist[1]

        # ── Sync disturbances → PID sim, step it ───────────────────────
        G_sync = self._manual_dist[0] if self._manual_dist else raw.G
        y_sync = self._manual_dist[1] if self._manual_dist else raw.y
        pid_result = self.pid_sim.step(G_sync, y_sync)

        # ── Build RL result dict ───────────────────────────────────────
        rl_result = {
            "cap":    round(raw.cap,    3),
            "eng":    round(raw.eng,    4),
            "G":      round(raw.G,      4),
            "y":      round(raw.y,      4),
            "L":      round(raw.L_act,  4),
            "al":     round(raw.al_act, 4),
            "T":      round(raw.T_act,  3),
            "ic":     round(raw.ic_act, 3),
            "ff":     round(raw.ff,     4),
            "action": [round(float(x), 4) for x in action[0]],
        }

        # ── Update impact counters ─────────────────────────────────────
        rl_cap_frac  = rl_result["cap"]  / 100.0
        pid_cap_frac = pid_result["cap"] / 100.0
        self._co2_rl  += rl_cap_frac  * self._CO2_SCALE
        self._co2_pid += pid_cap_frac * self._CO2_SCALE

        # Energy comparison is normalised to RL's capture rate:
        # "for each tonne RL captured, how much less energy did it use vs PID?"
        # kwh_delta = (pid_eng - rl_eng) × rl_cap × scale  →  positive = RL cheaper
        self._kwh_rl  += rl_result["eng"]  * rl_cap_frac * self._KWH_SCALE
        self._kwh_pid += pid_result["eng"] * rl_cap_frac * self._KWH_SCALE

        snap = self._snapshot(rl_result, pid_result)
        self.history.append(snap)
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)
        self.t += 1
        return snap

    # ── Frozen-agent fallback ──────────────────────────────────────────

    def _frozen_action(self, raw: CCUEnv) -> np.ndarray:
        """
        Proportional delta action that steers RL env toward PID targets.
        LSTM state is NOT updated while frozen, so RL can recover when unfrozen.
        """
        error = 90.0 - raw.cap
        dL  = 1.0;  dal = 0.02;  dT = 2.5

        L_target  = float(np.clip(5.0  + 0.40  * error, 2.0, 12.0))
        al_target = float(np.clip(0.27 - 0.008 * error, 0.18, 0.38))
        T_target  = float(np.clip(40.0 - 0.60  * error, 30.0, 55.0))

        a0 = float(np.clip((L_target  - raw.L_cmd)  / dL,  -1.0, 1.0))
        a1 = float(np.clip((al_target - raw.al_cmd) / dal, -1.0, 1.0))
        a2 = float(np.clip((T_target  - raw.T_cmd)  / dT,  -1.0, 1.0))
        a3 = 0.0

        return np.array([[a0, a1, a2, a3]], dtype=np.float32)

    # ── Snapshot builder ───────────────────────────────────────────────

    def _snapshot(
        self,
        rl_result: Optional[dict] = None,
        pid_result: Optional[dict] = None,
    ) -> dict:
        if rl_result is None:
            raw = self._raw()
            rl_result = {
                "cap":    round(raw.cap    or 85.0, 3),
                "eng":    round(raw.eng    or 4.0,  4),
                "G":      round(raw.G      or 0.8,  4),
                "y":      round(raw.y      or 0.08, 4),
                "L":      round(raw.L_act  or 5.0,  4),
                "al":     round(raw.al_act or 0.27, 4),
                "T":      round(raw.T_act  or 40.0, 3),
                "ic":     round(raw.ic_act or 38.0, 3),
                "ff":     round(raw.ff     or 0.0,  4),
                "action": [0.0, 0.0, 0.0, 0.0],
            }
        if pid_result is None:
            pid_result = {
                "cap": round(self.pid_sim.cap,    3),
                "eng": round(self.pid_sim.eng,    4),
                "L":   round(self.pid_sim.L_act,  4),
                "al":  round(self.pid_sim.al_act, 4),
                "T":   round(self.pid_sim.T_act,  3),
                "ic":  round(self.pid_sim.ic_act, 3),
            }

        co2_delta = self._co2_rl  - self._co2_pid
        kwh_delta = self._kwh_pid - self._kwh_rl   # positive = RL saves energy

        return {
            "t":             self.t,
            "rl":            rl_result,
            "pid":           pid_result,
            "frozen":        self.frozen,
            "lambda_energy": round(self.lambda_energy, 4),
            "impact": {
                "co2_captured_rl_t":  round(self._co2_rl,  4),
                "co2_captured_pid_t": round(self._co2_pid, 4),
                "co2_delta_t":        round(co2_delta, 4),
                "energy_kwh_saved":   round(kwh_delta, 2),
                "money_saved_usd":    round(kwh_delta * 0.10, 2),
                "trees_equivalent":   round(max(0.0, co2_delta) * 1000.0 / 22.0, 1),
            },
        }

    # ── User controls ──────────────────────────────────────────────────

    def set_disturbance(self, G_gas: float, y_CO2_in: float) -> None:
        self._manual_dist = (
            float(np.clip(G_gas,     0.40, 2.50)),
            float(np.clip(y_CO2_in, 0.04, 0.22)),
        )

    def clear_disturbance(self) -> None:
        self._manual_dist = None

    def attack_plant(self) -> None:
        """Push G_gas to 1.5 and y_CO2_in to 0.20 — the demo 'wow moment'."""
        self.set_disturbance(1.50, 0.20)

    def reset_impact(self) -> None:
        """Zero the impact counters without resetting the simulation."""
        self._co2_rl  = 0.0
        self._co2_pid = 0.0
        self._kwh_rl  = 0.0
        self._kwh_pid = 0.0

    def set_lambda(self, lam: float) -> None:
        self.lambda_energy = float(np.clip(lam, 0.0, 0.20))
        raw = self._raw()
        if raw.lam is not None:
            raw.lam = self.lambda_energy

    def freeze_agent(self, frozen: bool) -> None:
        if not self.frozen and frozen:
            self._frozen_lstm = self.lstm_states
        elif self.frozen and not frozen:
            self.lstm_states = self._frozen_lstm
            self._frozen_lstm = None
            self.episode_starts[:] = False
            self._smooth_action[:] = 0.0  # clear buffer so RL starts fresh
        self.frozen = frozen

    def get_snapshot(self) -> dict:
        return self.history[-1] if self.history else self._snapshot()

    def get_history(self) -> list:
        return list(self.history)
