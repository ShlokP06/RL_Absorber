"""
CCU Real-Time Control Environment
==================================
Gymnasium environment for PPO-LSTM Pareto-optimal control.

Framing B: quasi-steady-state surrogate + LSTM policy + actuator lag +
OU disturbances provide approximate dynamic controller behaviour valid
when disturbance timescales (15-30 min) exceed column settling times (5-10 min).

Improvements vs baseline
------------------------
[2] Intercooling as 4th control variable (T_ic_C, 25-50 °C).
    Applied at absorber midpoint; cools liquid → better VLE driving force
    in lower column → more CO2 absorbed per unit solvent.

[3] Hard constraint projection on L_liq.
    Before applying L_cmd, max_safe_L(G, T_K, alpha) is called and the
    command is clipped to that value. Guarantees flood_fraction < 0.79
    regardless of agent action — no reward tuning required.

Observation (17-dim, normalised [-1, 1])
-----------------------------------------
[0-1]   G_gas, y_CO2_in           disturbance state
[2-3]   G_trend, y_trend          OU drift direction (feedforward)
[4-7]   L_act, al_act, T_act, T_ic_act   actual controllable values
[8]     L_cmd                     commanded L (lag signal)
[9]     capture                   (cap - 50) / 50  → [-1, 1] over [0, 100] %
[10]    energy                    (eng - 5) / 5    → [-1, 1] over [0, 10] GJ/t
[11-12] d_capture, d_energy       rate of change (derivative signal)
[13]    cap_integral              accumulated capture deficit below 90% (leaky)
[14]    flood_proximity           constraint headroom = 1 - ff/0.79
[15]    lambda_energy             Pareto weight (goal conditioning)
[16]    T_ic_cmd                  commanded intercooler (lag signal)

Action (4-dim, [-1, 1]): delta commands on L_liq, alpha_lean, T_L_in, T_ic
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .surrogate import SurrogatePredictor, X_BOUNDS
from .simulation import max_safe_L, flood_fraction

CTRL = {
    "L_liq":    (2.0,  12.0),
    "alpha":    (0.18,  0.38),
    "T_L_in":   (30.0,  55.0),
    "T_ic":     (25.0,  50.0),
}
TAU  = {"L_liq": 3.0, "alpha": 5.0, "T_L_in": 2.0, "T_ic": 4.0}
OU   = {"theta": 0.08, "sigma": 0.015}
NOISE = {"capture": 0.5, "energy": 0.03}
DRAND = 0.03


class CCUEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self,
                 model_path      = "models/surrogate/model.pt",
                 scaler_path     = "models/surrogate/scalers.pkl",
                 max_steps       = 120,
                 lambda_range    = (0.0, 0.20),
                 lam_smooth      = 0.030,
                 lam_integral    = 0.15,
                 lam_energy_int  = 0.08,
                 lam_above       = 0.10,
                 lam_over        = 0.05,
                 lam_flood       = 0.15,
                 step_prob       = 0.04,
                 actuator_lag    = True,
                 obs_noise       = True,
                 domain_rand     = True,
                 continue_prob   = 0.30,
                 curriculum_phase= 2):

        super().__init__()
        self.surrogate    = SurrogatePredictor(model_path, scaler_path)
        self.max_steps    = max_steps
        self.lam_range    = lambda_range
        self.lam_smooth   = lam_smooth
        self.lam_I        = lam_integral
        self.lam_Ie       = lam_energy_int
        self.lam_above    = lam_above
        self.lam_over     = lam_over
        self.lam_fl       = lam_flood
        self.step_prob    = step_prob
        self.act_lag      = actuator_lag
        self.obs_noise    = obs_noise
        self.domain_rand  = domain_rand
        self.cont_prob    = continue_prob
        self.phase        = curriculum_phase

        self.G_lo,  self.G_hi  = X_BOUNDS["G_gas_kg_m2s"]
        self.y_lo,  self.y_hi  = X_BOUNDS["y_CO2_in"]
        self.L_lo,  self.L_hi  = CTRL["L_liq"]
        self.al_lo, self.al_hi = CTRL["alpha"]
        self.T_lo,  self.T_hi  = CTRL["T_L_in"]
        self.ic_lo, self.ic_hi = CTRL["T_ic"]

        self.dL  = 0.10 * (self.L_hi  - self.L_lo)
        self.dal = 0.10 * (self.al_hi - self.al_lo)
        self.dT  = 0.10 * (self.T_hi  - self.T_lo)
        self.dic = 0.10 * (self.ic_hi - self.ic_lo)

        self.observation_space = spaces.Box(
            -np.ones(17, np.float32), np.ones(17, np.float32), dtype=np.float32)
        self.action_space = spaces.Box(
            -np.ones(4,  np.float32), np.ones(4,  np.float32), dtype=np.float32)

        self._carry = None
        self._zero_state()

    # ── State helpers ─────────────────────────────────────────────────────────

    def _zero_state(self):
        self.G = self.y = self.G_mean = self.y_mean = None
        self.G_trend = self.y_trend = 0.0
        self.L_cmd = self.al_cmd = self.T_cmd = self.ic_cmd = None
        self.L_act = self.al_act = self.T_act = self.ic_act = None
        self.cap = self.eng = self.prev_cap = self.prev_eng = None
        self.cap_int = self.eng_int = 0.0
        self.ff = 0.0
        self.lam = None
        self.drand = 1.0
        self.prev_act = np.zeros(4, np.float32)
        self.t = 0

    def set_phase(self, phase: int) -> None:
        self.phase = int(np.clip(phase, 0, 2))

    def _n01(self, v, lo, hi):
        return float(np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0))

    def _nsym(self, v, s):
        return float(np.clip(v / (s + 1e-8), -1.0, 1.0))

    # ── Hard constraint projection ─────────────────────────────────────────────

    def _project_L(self, L_candidate):
        """
        Clamp L_candidate to max_safe_L given current G_gas and actuator actuals.
        Guarantees flood_fraction < 0.79 regardless of agent action.
        Called before updating L_cmd.
        """
        T_K   = self.T_act + 273.15 if self.T_act is not None else 313.15
        alpha = self.al_act          if self.al_act is not None else 0.27
        L_max = max_safe_L(self.G, T_K, alpha, limit=0.79)
        return float(np.clip(L_candidate, self.L_lo, min(L_max, self.L_hi)))

    # ── Disturbances ──────────────────────────────────────────────────────────

    def _init_dist(self, carry=False):
        rng = self.np_random
        if not carry:
            # Cap G_gas start state at 1.30 — above this the column is always
            # flooded at minimum liquid, making start states unrecoverable.
            # The agent still encounters high G_gas via OU drift and step changes.
            G_start_max = min(self.G_hi, 1.30)
            self.G_mean = float(rng.uniform(self.G_lo, G_start_max))
            self.y_mean = float(rng.uniform(self.y_lo, self.y_hi))
            self.G, self.y = self.G_mean, self.y_mean
        self.G_trend = self.y_trend = 0.0

    def _step_dist(self):
        if self.phase == 0:
            return
        rng = self.np_random
        th, sg = OU["theta"], OU["sigma"]
        dG = th*(self.G_mean - self.G) + sg*(self.G_hi-self.G_lo)*float(rng.normal())
        dy = th*(self.y_mean - self.y) + sg*(self.y_hi-self.y_lo)*float(rng.normal())
        self.G_trend = self._nsym(dG, 0.05*(self.G_hi-self.G_lo))
        self.y_trend = self._nsym(dy, 0.05*(self.y_hi-self.y_lo))
        self.G = float(np.clip(self.G + dG, self.G_lo, self.G_hi))
        self.y = float(np.clip(self.y + dy, self.y_lo, self.y_hi))
        if self.phase == 2 and float(rng.random()) < self.step_prob:
            self.G_mean = float(rng.uniform(self.G_lo, self.G_hi))
            self.y_mean = float(rng.uniform(self.y_lo, self.y_hi))
            self.G, self.y = self.G_mean, self.y_mean

    def _step_act(self):
        if not self.act_lag:
            self.L_act  = self.L_cmd
            self.al_act = self.al_cmd
            self.T_act  = self.T_cmd
            self.ic_act = self.ic_cmd
        else:
            self.L_act  += (1/TAU["L_liq"])  * (self.L_cmd  - self.L_act)
            self.al_act += (1/TAU["alpha"])   * (self.al_cmd - self.al_act)
            self.T_act  += (1/TAU["T_L_in"]) * (self.T_cmd  - self.T_act)
            self.ic_act += (1/TAU["T_ic"])    * (self.ic_cmd - self.ic_act)
        self.L_act  = float(np.clip(self.L_act,  self.L_lo,  self.L_hi))
        self.al_act = float(np.clip(self.al_act, self.al_lo, self.al_hi))
        self.T_act  = float(np.clip(self.T_act,  self.T_lo,  self.T_hi))
        self.ic_act = float(np.clip(self.ic_act, self.ic_lo, self.ic_hi))
        self.L_act = self._project_L(self.L_act)

    def _query(self):
        r = self.surrogate.predict(
            G_gas=self.G, L_liq=self.L_act, y_CO2_in=self.y,
            T_L_in_C=self.T_act, alpha_lean=self.al_act, T_ic_C=self.ic_act,
        )
        cap = float(np.clip(r["capture_rate"] * self.drand, 0.0, 100.0))
        eng = float(np.clip(r["E_specific_GJ"] * self.drand, 0.5, 50.0))
        if self.obs_noise:
            cap = float(np.clip(cap + self.np_random.normal(0, NOISE["capture"]),
                                0.0, 100.0))
            eng = float(np.clip(eng + self.np_random.normal(0, NOISE["energy"]),
                                0.5, 50.0))
        self.ff = flood_fraction(self.G, self.L_act, self.T_act + 273.15, self.al_act)
        return cap, eng

    # ── Observation ───────────────────────────────────────────────────────────

    def _obs(self):
        return np.clip(np.array([
            self._n01(self.G,      self.G_lo,  self.G_hi),    # [0]
            self._n01(self.y,      self.y_lo,  self.y_hi),    # [1]
            self.G_trend,                                       # [2]
            self.y_trend,                                       # [3]
            self._n01(self.L_act,  self.L_lo,  self.L_hi),    # [4]
            self._n01(self.al_act, self.al_lo, self.al_hi),   # [5]
            self._n01(self.T_act,  self.T_lo,  self.T_hi),    # [6]
            self._n01(self.ic_act, self.ic_lo, self.ic_hi),   # [7]
            self._n01(self.L_cmd,  self.L_lo,  self.L_hi),    # [8]  lag signal
            self._nsym(self.cap - 50.0, 50.0),                 # [9]  [-1,1] over [0,100]%
            self._nsym(self.eng - 5.0,   5.0),                 # [10] [-1,1] over [0,10] GJ/t
            self._nsym(self.cap - self.prev_cap, 10.0),        # [11]
            self._nsym(self.eng - self.prev_eng,  2.0),        # [12]
            self._nsym(self.cap_int, 5.0),                     # [13]
            float(np.clip(1.0 - self.ff / 0.79, -1.0, 1.0)),  # [14] headroom
            self._n01(self.lam, self.lam_range[0], self.lam_range[1]),  # [15]
            self._n01(self.ic_cmd, self.ic_lo, self.ic_hi),   # [16] ic lag signal
        ], dtype=np.float32), -1.0, 1.0)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _reward(self, action):
        # Squared action change — stronger penalty on large jumps
        da2 = float(np.mean((action - self.prev_act) ** 2))

        # Shaped capture: quadratic amplifies difference between 75% and 95%
        cap_n = self.cap / 100.0
        cap_reward = cap_n ** 2

        # Above-target bonus: scales linearly from 85% → 95%, then plateaus.
        # At 85%: 0, at 90%: lam_above/2, at 95%+: lam_above (full bonus).
        # Capped at 95% so the agent targets 90-95%, not 100%.
        above = self.lam_above * min(max(0.0, self.cap - 85.0), 10.0) / 10.0

        # Over-capture penalty: discourages pushing above 95% unnecessarily.
        # Gentle slope — doesn't override the capture incentive below 95%.
        over_pen = self.lam_over * max(0.0, self.cap - 95.0) / 5.0

        # Energy penalty — normalised to [0, ~1] over typical operating range
        eng_pen = self.lam * (self.eng - 3.5) / 3.0

        # Flood soft penalty (narrower zone; hard constraint does the safety work)
        fl_pen = (self.lam_fl * min((self.ff - 0.75) / 0.05, 2.0)
                  if self.ff > 0.75 else 0.0)

        r = (cap_reward
             + above
             - over_pen
             - eng_pen
             - self.lam_smooth * da2
             - self.lam_I  * max(self.cap_int, 0.0)
             - self.lam_Ie * max(self.eng_int,  0.0)
             - fl_pen)

        info = dict(
            capture_rate=self.cap, E_specific_GJ=self.eng,
            G_gas=self.G, y_CO2_in=self.y,
            L_liq=self.L_act, alpha_lean=self.al_act,
            T_L_in_C=self.T_act, T_ic_C=self.ic_act,
            flood_fraction=self.ff, lambda_energy=self.lam, phase=self.phase,
        )
        return float(r), info

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = self.np_random

        carry = self._carry is not None and float(rng.random()) < self.cont_prob
        if carry:
            s = self._carry
            self.G, self.y          = s["G"],     s["y"]
            self.G_mean, self.y_mean= s["G_mean"],s["y_mean"]
            self.L_cmd,  self.al_cmd= s["L_cmd"], s["al_cmd"]
            self.T_cmd,  self.ic_cmd= s["T_cmd"], s["ic_cmd"]
            self.L_act,  self.al_act= s["L_act"], s["al_act"]
            self.T_act,  self.ic_act= s["T_act"], s["ic_act"]
            self.G_trend = self.y_trend = 0.0
        else:
            # Initialise disturbances FIRST so G is available for _project_L
            self._init_dist(carry=False)
            self.L_cmd  = float(rng.uniform(self.L_lo,  self.L_hi))
            self.al_cmd = float(rng.uniform(self.al_lo, self.al_hi))
            self.T_cmd  = float(rng.uniform(self.T_lo,  self.T_hi))
            self.ic_cmd = float(rng.uniform(self.ic_lo, self.ic_hi))
            self.L_cmd  = self._project_L(self.L_cmd)
            self.L_act  = self.L_cmd;  self.al_act = self.al_cmd
            self.T_act  = self.T_cmd;  self.ic_act = self.ic_cmd

        self.lam   = float(rng.uniform(*self.lam_range))
        self.drand = float(np.clip(1.0 + rng.normal(0, DRAND), 0.90, 1.10)) \
                     if self.domain_rand else 1.0
        self.cap, self.eng = self._query()
        self.prev_cap, self.prev_eng = self.cap, self.eng
        self.cap_int = self.eng_int = 0.0
        self.prev_act = np.zeros(4, np.float32)
        self.t = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.array(action, np.float32)

        # 1. Disturbances advance
        self._step_dist()

        # 2. Commands — L_liq projected through hard flood constraint
        L_raw   = self.L_cmd  + float(action[0]) * self.dL
        self.L_cmd  = self._project_L(L_raw)           # HARD CONSTRAINT [3]
        self.al_cmd = float(np.clip(self.al_cmd + float(action[1])*self.dal,
                                    self.al_lo, self.al_hi))
        self.T_cmd  = float(np.clip(self.T_cmd  + float(action[2])*self.dT,
                                    self.T_lo,  self.T_hi))
        self.ic_cmd = float(np.clip(self.ic_cmd + float(action[3])*self.dic,
                                    self.ic_lo, self.ic_hi))

        # 3. Actuator lag
        self._step_act()

        # 4. Surrogate
        self.prev_cap, self.prev_eng = self.cap, self.eng
        self.cap, self.eng = self._query()

        # 5. Controller signals (exponential decay instead of constant)
        self.cap_int = float(np.clip(
            self.cap_int * 0.95 + max(0.0, 90.0 - self.cap) / 100.0, 0.0, 5.0))
        self.eng_int = float(np.clip(
            self.eng_int * 0.95 + max(0.0, self.eng - 5.0) / 10.0,  0.0, 5.0))

        # 6. Reward
        reward, info = self._reward(action)
        self.prev_act = action.copy()

        # 7. Carry state
        self._carry = dict(
            G=self.G, y=self.y, G_mean=self.G_mean, y_mean=self.y_mean,
            L_cmd=self.L_cmd, al_cmd=self.al_cmd,
            T_cmd=self.T_cmd, ic_cmd=self.ic_cmd,
            L_act=self.L_act, al_act=self.al_act,
            T_act=self.T_act, ic_act=self.ic_act,
        )
        self.t += 1
        return self._obs(), reward, False, self.t >= self.max_steps, info

    def state_dict(self):
        return dict(
            G_gas=self.G, y_CO2_in=self.y,
            G_trend=self.G_trend, y_trend=self.y_trend,
            L_liq=self.L_act, alpha_lean=self.al_act,
            T_L_in_C=self.T_act, T_ic_C=self.ic_act,
            capture_rate=self.cap, E_specific_GJ=self.eng,
            flood_fraction=self.ff, lambda_energy=self.lam,
            capture_integral=self.cap_int, phase=self.phase, step=self.t,
        )

    def render(self):
        pass