"""
MEA Post-Combustion CO2 Capture — Physics Simulation
=====================================================
Rate-based absorber-stripper model for 30 wt% MEA solvent.

References
----------
[1] Jou et al. (1995)        Can. J. Chem. Eng. 73, 140
[2] Weiland et al. (1998)    J. Chem. Eng. Data 43, 378
[3] Versteeg et al. (1996)   Chem. Eng. Comm. 144, 113
[4] Bravo & Fair (1982)      Ind. Eng. Chem. PDD 21, 162
[5] DeCoursey (1974)         Chem. Eng. Sci. 29, 1867
[6] Abu-Zahra et al. (2007)  Int. J. GHG Control 1, 37
[7] Billet & Schultes (1999) Chem. Eng. RD 77, 498
"""

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
R      = 8.314
MEA_MW = 61.08e-3
H2O_MW = 18.015e-3
w_MEA  = 0.30
a_p    = 250.0
eps    = 0.97
d_h    = 4 * eps / a_p
H_col  = 15.0
P_tot  = 101.325e3
W_COMP = 0.35         # GJ/tonne CO2 compression to 110 bar [6]
N_STGS = 60
J_IC   = N_STGS // 2  # intercooler stage (column midpoint)

# VLE — Kent-Eisenberg fitted to Jou et al. (1995) [1]
_pts = [(0.25, 313.15, 0.30), (0.40, 313.15, 3.50),
        (0.25, 393.15, 40.0), (0.45, 393.15, 200.0)]
_A   = np.array([[1, a, 1/T, a/T] for a, T, _ in _pts])
_b   = np.array([np.log10(p) for _, _, p in _pts])
_C   = np.linalg.solve(_A, _b)


# ── Physical properties ───────────────────────────────────────────────────────

def density(T_C, alpha):
    return 1060.0 - 0.57 * (T_C - 25.0) + 25.0 * alpha

def viscosity(T_K, alpha):
    return float(np.clip(
        np.exp(2121.0 / T_K - 6.40) * (1.0 + 0.50 * alpha) * 1e-3, 5e-4, 1.5e-2))

def diffusivity_CO2_liq(T_K, mu):
    x   = (w_MEA / MEA_MW) / (w_MEA / MEA_MW + (1 - w_MEA) / H2O_MW)
    M   = x * MEA_MW * 1e3 + (1 - x) * H2O_MW * 1e3
    phi = 2.6 * (1 - x) + 1.5 * x
    return float(np.clip(
        7.4e-8 * np.sqrt(phi * M) * T_K / (mu * 1e3 * 34.0**0.6) * 1e-4, 1e-11, 1e-8))

def diffusivity_MEA_liq(T_K, mu):
    return float(np.clip(1e-9 * (313.0 / T_K) * (1.5e-3 / mu), 1e-11, 1e-8))

def diffusivity_CO2_gas(T_K):
    return 1.6e-5 * (T_K / 273.15) ** 1.75

def free_MEA(alpha, T_K):
    return float(np.clip(1.0 - 2.0 * alpha, 0.05, 1.0) *
                 w_MEA * density(T_K - 273.15, alpha) / MEA_MW)

def flue_gas(y_CO2):
    """Returns (y_H2O, y_inert, M_mix) from combustion correlation."""
    y_H2O   = float(np.clip(0.05 + 0.35 * y_CO2, 0.04, 0.15))
    y_inert = float(np.clip(1.0 - y_CO2 - y_H2O, 0.60, 0.92))
    M_mix   = y_CO2 * 0.044 + y_H2O * 0.018 + y_inert * 0.028
    return y_H2O, y_inert, M_mix


# ── VLE ───────────────────────────────────────────────────────────────────────

def p_star(alpha, T_K):
    alpha = float(np.clip(alpha, 0.10, 0.57))
    return float(np.clip(
        10.0 ** (_C[0] + _C[1]*alpha + (_C[2] + _C[3]*alpha) / T_K), 1e-6, 1e4))

def H_phys(T_K):
    return 3.4e-2 * np.exp(2044.0 * (1.0 / 298.15 - 1.0 / T_K))


# ── Kinetics ──────────────────────────────────────────────────────────────────

def k2(T_K):
    return 9.77e10 * np.exp(-6975.0 / T_K) / 1000.0


# ── Mass transfer ─────────────────────────────────────────────────────────────

def kL0(L, T_K, alpha):
    """Returns (kL [m/s], D_L [m²/s], D_M [m²/s]). Billet & Schultes [7]."""
    mu  = viscosity(T_K, alpha)
    rho = density(T_K - 273.15, alpha)
    D_L = diffusivity_CO2_liq(T_K, mu)
    D_M = diffusivity_MEA_liq(T_K, mu)
    u_L = L / max(rho, 1.0)
    kL  = float(np.clip(1.334 * np.sqrt(D_L * u_L / d_h), 1e-6, 1e-2))
    return kL, D_L, D_M

def kG(G, T_K, y_CO2):
    """Gas-film MTC [kmol/m²/s/kPa]. Bravo & Fair (1982) [4]."""
    _, _, M_mix = flue_gas(y_CO2)
    D_G   = diffusivity_CO2_gas(T_K)
    rho_g = P_tot * M_mix / (R * T_K)
    u_g   = G / max(rho_g, 1e-3)
    mu_g  = 1.8e-5
    Sh    = 0.407 * max(rho_g * u_g * d_h / mu_g, 0.1)**0.655 * \
            max(mu_g / (rho_g * D_G), 0.1)**(1.0/3.0)
    return float(np.clip(Sh * D_G / d_h / (R * T_K), 1e-8, 1e-3))

def enhancement(T_K, alpha, D_L, kL_, D_M):
    """E and Ha. DeCoursey (1974) [5]."""
    Ha    = float(np.clip(
        np.sqrt(max(k2(T_K) * free_MEA(alpha, T_K) * D_L, 0.0)) / max(kL_, 1e-15),
        0.01, 1000.0))
    C_i   = max(p_star(alpha, T_K) / H_phys(T_K), 0.05)
    E_inf = float(np.clip(
        1.0 + D_M * free_MEA(alpha, T_K) / (2.0 * D_L * C_i), 2.0, 2000.0))
    Ei    = max(E_inf - 1.0, 1.0)
    disc  = Ha**4 / (4.0 * Ei**2) + E_inf * Ha**2 / Ei + 1.0
    E     = -Ha**2 / (2.0 * Ei) + np.sqrt(max(disc, 0.0))
    return float(np.clip(E, 1.0, E_inf)), Ha


# ── Flooding ──────────────────────────────────────────────────────────────────

def flood_fraction(G, L, T_K, alpha):
    """Fractional approach to flooding. Billet & Schultes (1999) [7]."""
    rho_L = density(T_K - 273.15, alpha)
    rho_g = P_tot * 0.030 / (R * T_K)
    u_g   = G / max(rho_g, 0.1)
    u_L   = L / max(rho_L, 1.0)
    u_fl  = 0.135 * np.sqrt((rho_L - rho_g) / rho_g * 9.81 * d_h)
    liq_f = max(1.0 - 0.3 * (u_L / 0.01), 0.5)
    return float(np.clip(u_g / max(u_fl * liq_f, 1e-6), 0.0, 3.0))


def max_safe_L(G, T_K, alpha, limit=0.79):
    """
    Maximum L_liq [kg/m²/s] such that flood_fraction < limit.
    Found via bisection. Returns L_lo=2.0 if flooding at minimum liquid.
    Used by the RL environment for hard constraint projection.
    """
    if flood_fraction(G, 2.0, T_K, alpha) >= limit:
        return 2.0
    if flood_fraction(G, 15.0, T_K, alpha) < limit:
        return 15.0
    lo, hi = 2.0, 15.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if flood_fraction(G, mid, T_K, alpha) < limit:
            lo = mid
        else:
            hi = mid
    return float(lo)


# ── T_reb from alpha_lean ─────────────────────────────────────────────────────

def T_reb(alpha_lean):
    """Reboiler temperature [°C]: p*(alpha, T_reb) = 0.5*P_tot."""
    target = 0.5 * P_tot / 1000.0
    lo, hi = 370.0, 430.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if p_star(alpha_lean, mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.clip(0.5 * (lo + hi), 370.0, 425.0)) - 273.15


# ── Absorber ──────────────────────────────────────────────────────────────────

def run_absorber(G, L, y_CO2, T_L_in_K, alpha_lean,
                 T_ic_C=None, N=N_STGS, tol=1e-7):
    """
    Counter-current packed absorber via bisection on y_top.

    Parameters
    ----------
    T_ic_C : float, optional
        Intercooler setpoint [°C] applied at stage J_IC (column midpoint).
        Liquid temperature is reset to min(T_L[J_IC], T_ic_K) at that stage,
        cooling the solvent and improving VLE driving force in the lower column.
        If None, intercooling is disabled (backwards-compatible default).
    """
    _, y_inert, _ = flue_gas(y_CO2)
    F_inert = G * y_inert / 0.028
    F_MEA   = L * w_MEA / MEA_MW
    dz      = H_col / N
    H_abs   = 85.0e3
    T_ic_K  = (T_ic_C + 273.15) if T_ic_C is not None else None

    def alpha_at(y, y_t):
        n  = F_inert * y / max(1 - y, 1e-9)
        nt = F_inert * y_t / max(1 - y_t, 1e-9)
        return float(np.clip(alpha_lean + (n - nt) / max(F_MEA, 1e-9),
                             alpha_lean, 0.54))

    def integrate(y_t, n_inner=3):
        y_t    = float(np.clip(y_t, 1e-9, y_CO2 * 0.9999))
        T_L    = np.full(N + 1, T_L_in_K + 8.0)
        T_L[N] = T_L_in_K
        y      = np.zeros(N + 1)
        Ha_list, E_list = [], []

        for _ in range(n_inner):
            Ha_list, E_list = [], []
            y[0] = y_CO2

            for j in range(N):
                # Intercooling applied at midpoint stage
                if T_ic_K is not None and j == J_IC:
                    T_L[j] = min(T_L[j], T_ic_K)

                y_g  = float(np.clip(y[j], 1e-9, 0.9999))
                Ts   = 0.5 * (T_L[j] + T_L[j + 1])
                al   = alpha_at(y_g, y_t)
                kl, Dl, Dm = kL0(L, Ts, al)
                E, Ha  = enhancement(Ts, al, Dl, kl, Dm)
                Hp     = H_phys(Ts)
                KOG    = 1.0 / (1.0 / max(kG(G, Ts, y_CO2), 1e-15)
                                + 1.0 / max(E * kl / Hp / 1000.0, 1e-15))
                df     = max(y_g * P_tot / 1000.0 - p_star(al, Ts), 0.0)
                n_in   = y_g / max(1 - y_g, 1e-9) * F_inert
                n_out  = max(n_in - KOG * a_p * df * dz * 1000.0, 0.0)
                y[j+1] = n_out / max(F_inert + n_out, 1e-9)
                Ha_list.append(Ha); E_list.append(E)

            T_L[N] = T_L_in_K
            for j in range(N - 1, -1, -1):
                y_j  = float(np.clip(0.5 * (y[j] + y[j+1]), 1e-9, y_CO2))
                al   = alpha_at(y_j, y_t)
                Ts   = T_L[j + 1]
                kl, Dl, Dm = kL0(L, Ts, al)
                E, _   = enhancement(Ts, al, Dl, kl, Dm)
                Hp     = H_phys(Ts)
                KOG    = 1.0 / (1.0 / max(kG(G, Ts, y_CO2), 1e-15)
                                + 1.0 / max(E * kl / Hp / 1000.0, 1e-15))
                df     = max(y_j * P_tot / 1000.0 - p_star(al, Ts), 0.0)
                dT     = H_abs * KOG * a_p * df * dz * 1000.0 / max(L * 3800.0, 1e-9)
                T_L[j] = float(np.clip(T_L[j+1] + dT, T_L_in_K, T_L_in_K + 40.0))
                # Intercooling applied AFTER computing T_L[j] so it persists
                # into the next temperature step (j-1 uses the corrected T_L[j])
                if T_ic_K is not None and j == J_IC:
                    T_L[j] = min(T_L[j], T_ic_K)

        return float(y[N]), T_L, Ha_list, E_list

    def residual(y_t):
        out, _, _, _ = integrate(y_t)
        return out - y_t

    lo, hi = 1e-7, y_CO2 * 0.9999
    fl, fh = residual(lo), residual(hi)
    if fl * fh > 0:
        sol = lo if abs(fl) < abs(fh) else hi
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm  = residual(mid)
            if abs(fm) < tol:
                break
            if fl * fm < 0:
                hi, fh = mid, fm
            else:
                lo, fl = mid, fm
        sol = 0.5 * (lo + hi)

    y_out, T_L_fin, Ha_list, E_list = integrate(sol)
    return dict(
        capture_rate  = float(np.clip((1.0 - y_out / y_CO2) * 100.0, 0.0, 100.0)),
        alpha_rich    = alpha_at(y_CO2, sol),
        T_L_bottom_C  = float(T_L_fin[0] - 273.15),
        y_CO2_out     = float(y_out),
        Ha_avg        = float(np.mean(Ha_list)) if Ha_list else 0.0,
        E_avg         = float(np.mean(E_list))  if E_list  else 1.0,
    )


# ── Stripper ──────────────────────────────────────────────────────────────────

def run_stripper(alpha_rich, alpha_lean, L, T_rich_C, T_lean_in_C):
    """
    Specific reboiler duty [GJ/tonne CO2] including CO2 compression.
    HEX approach temperature scales with sqrt(L/L_nom). [6]
    """
    F_MEA   = L * w_MEA / MEA_MW
    mol_CO2 = F_MEA * max(alpha_rich - alpha_lean, 0.0)
    if mol_CO2 < 1e-6:
        return 999.0
    kg_CO2  = mol_CO2 * 0.044
    T_reb_C = T_reb(alpha_lean)
    dT_app  = 5.0 * np.sqrt(max(L, 0.1) / 5.0)
    T_HEX   = float(np.clip(T_reb_C - dT_app, 60.0, T_reb_C - 2.0))
    dT_sens = max(T_reb_C - T_HEX, 0.0)
    lam_s   = max(2260.0 - 2.5 * (T_reb_C - 100.0), 2100.0) * 1e3
    r_s     = 0.45 * (T_reb_C / 120.0) ** 1.5
    Q_des   = 85.0e3 * mol_CO2
    Q_sens  = L * 3800.0 * dT_sens
    Q_steam = r_s * kg_CO2 * lam_s
    E_reb   = (Q_des + Q_sens + Q_steam) / max(kg_CO2, 1e-9) * 1e-6
    return float(np.clip(E_reb + W_COMP, 0.5, 50.0))


# ── Validity filter ───────────────────────────────────────────────────────────

def is_valid(rec):
    cr = rec["capture_rate"];  ar = rec["alpha_rich"]
    al = rec["alpha_lean"];    Tb = rec["T_L_bottom_C"]
    E  = rec["E_specific_GJ"]; ff = rec.get("flood_fraction", 0.0)
    if ff > 0.80:                  return False, "flooded"
    if not (20.0 < cr <= 100.0):   return False, f"capture={cr:.1f}"
    if not (0.28 < ar < 0.57):     return False, f"alpha_rich={ar:.3f}"
    if ar <= al + 0.005:           return False, "delta_alpha<0.005"
    if not (35.0 < Tb < 90.0):     return False, f"T_bot={Tb:.1f}"
    if not (2.0  < E  < 15.0):     return False, f"E={E:.2f}"
    return True, "ok"
