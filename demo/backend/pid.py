"""
PID controller and PID-based simulator for the MEA absorber demo.

Three independent PID loops (L_liq, alpha_lean, T_L_in) driving the surrogate
directly. Intentionally sluggish under large G_gas spikes — the RL agent's
"wow moment".
"""

from __future__ import annotations

import numpy as np


class PIDController:
    """Discrete-time PID with anti-windup integral clamping."""

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        setpoint: float,
        out_lo: float,
        out_hi: float,
        bias: float = 0.0,
        integral_limit: float = 10.0,
    ) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.out_lo = out_lo
        self.out_hi = out_hi
        self.bias = bias
        self.integral_limit = integral_limit
        self.integral: float = 0.0
        self.prev_error: float = 0.0

    def step(self, measurement: float) -> float:
        error = self.setpoint - measurement
        self.integral = float(
            np.clip(self.integral + error, -self.integral_limit, self.integral_limit)
        )
        derivative = error - self.prev_error
        self.prev_error = error
        raw = (
            self.bias
            + self.Kp * error
            + self.Ki * self.integral
            + self.Kd * derivative
        )
        return float(np.clip(raw, self.out_lo, self.out_hi))

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0


class PIDSimulator:
    """
    Three-loop PID (L_liq, alpha_lean, T_L_in) running on the surrogate.

    Shares the same surrogate predictor and flood constraint as the RL env.
    alpha_lean and T_L_in have negative gains because:
      - Lower alpha_lean → fresher solvent → more capacity
      - Lower T_L_in   → better VLE driving force
    """

    CAP_SP: float = 90.0   # capture rate setpoint [%]
    IC_NOM: float = 38.0   # intercooler temp fixed [°C]

    def __init__(self, surrogate) -> None:
        self.surrogate = surrogate
        self._make_controllers()
        self._init_state()

    def _make_controllers(self) -> None:
        self.pid_L = PIDController(
            Kp=0.40, Ki=0.06, Kd=0.10,
            setpoint=self.CAP_SP,
            out_lo=2.0, out_hi=12.0,
            bias=5.0, integral_limit=20.0,
        )
        self.pid_al = PIDController(
            Kp=-0.008, Ki=-0.002, Kd=-0.002,
            setpoint=self.CAP_SP,
            out_lo=0.18, out_hi=0.38,
            bias=0.27, integral_limit=15.0,
        )
        self.pid_T = PIDController(
            Kp=-0.60, Ki=-0.10, Kd=-0.15,
            setpoint=self.CAP_SP,
            out_lo=30.0, out_hi=55.0,
            bias=40.0, integral_limit=20.0,
        )

    def _init_state(self) -> None:
        self.L_act:  float = 5.0
        self.al_act: float = 0.27
        self.T_act:  float = 40.0
        self.ic_act: float = self.IC_NOM
        self.cap:    float = 85.0
        self.eng:    float = 4.0

    def reset(self) -> None:
        self.pid_L.reset()
        self.pid_al.reset()
        self.pid_T.reset()
        self._init_state()

    def step(self, G: float, y: float) -> dict:
        """Advance one simulation step with disturbances G_gas, y_CO2_in."""
        # Lazy import avoids circular imports at module load time
        from src.simulation import max_safe_L  # noqa: PLC0415

        L_cmd  = self.pid_L.step(self.cap)
        al_cmd = self.pid_al.step(self.cap)
        T_cmd  = self.pid_T.step(self.cap)

        # First-order actuator lag (same time constants as RL env)
        self.L_act  += (1.0 / 3.0) * (L_cmd  - self.L_act)
        self.al_act += (1.0 / 5.0) * (al_cmd - self.al_act)
        self.T_act  += (1.0 / 2.0) * (T_cmd  - self.T_act)
        self.L_act  = float(np.clip(self.L_act,  2.0, 12.0))
        self.al_act = float(np.clip(self.al_act, 0.18, 0.38))
        self.T_act  = float(np.clip(self.T_act,  30.0, 55.0))

        T_K   = self.T_act + 273.15
        L_max = max_safe_L(G, T_K, self.al_act, limit=0.79)
        self.L_act = float(np.clip(self.L_act, 2.0, min(L_max, 12.0)))

        result = self.surrogate.predict(
            G_gas=G, L_liq=self.L_act, y_CO2_in=y,
            T_L_in_C=self.T_act, alpha_lean=self.al_act,
            T_ic_C=self.ic_act,
        )
        self.cap = result["capture_rate"]
        self.eng = result["E_specific_GJ"]

        return {
            "cap": round(self.cap, 3),
            "eng": round(self.eng, 4),
            "L":   round(self.L_act,  4),
            "al":  round(self.al_act, 4),
            "T":   round(self.T_act,  3),
            "ic":  round(self.ic_act, 3),
        }
