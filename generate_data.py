"""
Dataset Generation
==================
Generates one batch of MEA CCU simulation data via Latin Hypercube Sampling.

Usage
-----
    python generate_data.py --n 10000 --seed 101 --out data/batch1.csv
    python generate_data.py --n 10000 --seed 404 --out data/batch4.csv --wide
    python generate_data.py --check
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.simulation import (
    run_absorber, run_stripper, flood_fraction, T_reb,
    p_star, density, viscosity, diffusivity_CO2_liq,
    kL0, enhancement, is_valid,
)

BOUNDS_CORE = {
    "G_gas":      (0.60, 2.00),
    "L_liq":      (2.00, 12.00),
    "y_CO2_in":   (0.06, 0.20),
    "T_L_in_C":   (30.0, 55.0),
    "alpha_lean": (0.18, 0.38),
    "T_ic_C":     (25.0, 50.0),
}

BOUNDS_WIDE = {
    "G_gas":      (0.40, 2.50),
    "L_liq":      (1.50, 15.00),
    "y_CO2_in":   (0.04, 0.22),
    "T_L_in_C":   (25.0, 60.0),
    "alpha_lean": (0.15, 0.42),
    "T_ic_C":     (20.0, 55.0),
}


def sanity_check():
    print("=" * 55)
    print("  Sanity Check")
    print("=" * 55)

    print("\n[VLE] Jou et al. (1995):")
    for a, T, tgt in [(0.25, 313.15, 0.30), (0.40, 313.15, 3.50),
                      (0.25, 393.15, 40.0),  (0.45, 393.15, 200.0)]:
        p   = p_star(a, T)
        err = (p - tgt) / tgt * 100
        print(f"  alpha={a} T={T-273:.0f}C  p*={p:.2f}  err={err:+.0f}%  "
              f"{'OK' if abs(err) < 1 else 'FAIL'}")

    print("\n[Intercooling effect] G=1.2  L=5.0  y=0.13  T=40C  alpha=0.27:")
    ab_no = run_absorber(1.2, 5.0, 0.13, 313.15, 0.27, T_ic_C=None)
    ab_ic = run_absorber(1.2, 5.0, 0.13, 313.15, 0.27, T_ic_C=35.0)
    print(f"  No intercooling : capture={ab_no['capture_rate']:.1f}%  "
          f"alpha_rich={ab_no['alpha_rich']:.3f}")
    print(f"  T_ic=35°C       : capture={ab_ic['capture_rate']:.1f}%  "
          f"alpha_rich={ab_ic['alpha_rich']:.3f}")
    print(f"  Gain: {ab_ic['capture_rate']-ab_no['capture_rate']:+.1f}%")

    print("\n[Nominal] G=1.2  L=5.0  y=0.13  T=40C  alpha_lean=0.27  T_ic=40C:")
    ab  = run_absorber(1.2, 5.0, 0.13, 313.15, 0.27, T_ic_C=40.0)
    E_s = run_stripper(ab["alpha_rich"], 0.27, 5.0,
                       T_rich_C=ab["T_L_bottom_C"], T_lean_in_C=40.0)
    ff  = flood_fraction(1.2, 5.0, 313.15, 0.27)
    Tr  = T_reb(0.27)
    print(f"  T_reb        = {Tr:.1f} C")
    print(f"  capture_rate = {ab['capture_rate']:.1f} %   (expect 75-97)")
    print(f"  alpha_rich   = {ab['alpha_rich']:.3f}       (expect 0.38-0.50)")
    print(f"  T_L_bottom   = {ab['T_L_bottom_C']:.1f} C  (expect 50-75)")
    print(f"  E_specific   = {E_s:.3f} GJ/t  (expect 3.5-6.5)")
    print(f"  flood_frac   = {ff:.3f}        (expect <0.80)")
    print()


def generate(n, seed, out, bounds, save_every=5000):
    keys    = list(bounds.keys())
    lo      = np.array([bounds[k][0] for k in keys])
    hi      = np.array([bounds[k][1] for k in keys])
    samples = qmc.scale(qmc.LatinHypercube(d=len(keys), seed=seed).random(n), lo, hi)

    records, n_valid, n_err = [], 0, 0

    for i, row in enumerate(tqdm(samples, desc="Simulating", unit="pt")):
        G, L, y, TL, al, T_ic = row
        try:
            ff  = flood_fraction(G, L, TL + 273.15, al)
            ab  = run_absorber(G, L, y, TL + 273.15, al, T_ic_C=T_ic)
            E_s = run_stripper(ab["alpha_rich"], al, L,
                               T_rich_C=ab["T_L_bottom_C"], T_lean_in_C=TL)
            Tr  = T_reb(al)

            rec = dict(
                G_gas_kg_m2s   = round(G,    5),
                L_liq_kg_m2s   = round(L,    5),
                y_CO2_in       = round(y,    5),
                T_L_in_C       = round(TL,   3),
                alpha_lean     = round(al,   5),
                T_ic_C         = round(T_ic, 3),
                LG_ratio       = round(L / G, 4),
                flood_fraction = round(ff,   4),
                T_reb_C        = round(Tr,   2),
                capture_rate   = round(ab["capture_rate"],    4),
                alpha_rich     = round(ab["alpha_rich"],      5),
                delta_alpha    = round(ab["alpha_rich"] - al, 5),
                T_L_bottom_C   = round(ab["T_L_bottom_C"],   3),
                y_CO2_out      = round(ab["y_CO2_out"],       7),
                E_specific_GJ  = round(E_s, 4),
                Ha_avg         = round(ab["Ha_avg"], 3),
                E_factor_avg   = round(ab["E_avg"],  3),
            )
            ok, reason = is_valid({**rec, "flood_fraction": ff})
            rec["valid"]  = ok
            rec["reason"] = reason
            if ok:
                n_valid += 1
        except Exception as e:
            n_err += 1
            rec = {k: None for k in [
                "G_gas_kg_m2s", "L_liq_kg_m2s", "y_CO2_in", "T_L_in_C",
                "alpha_lean", "T_ic_C", "LG_ratio", "flood_fraction", "T_reb_C",
                "capture_rate", "alpha_rich", "delta_alpha", "T_L_bottom_C",
                "y_CO2_out", "E_specific_GJ", "Ha_avg", "E_factor_avg"]}
            rec.update(G_gas_kg_m2s=G, L_liq_kg_m2s=L, y_CO2_in=y, T_L_in_C=TL,
                       alpha_lean=al, T_ic_C=T_ic, valid=False, reason=str(e)[:100])

        records.append(rec)
        if (i + 1) % save_every == 0:
            pd.DataFrame(records).to_csv(out, index=False)
            tqdm.write(f"  [{i+1}/{n}] valid={n_valid} ({n_valid/(i+1)*100:.1f}%) "
                       f"errors={n_err}")

    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    dv = df[df["valid"] == True].copy()

    print(f"\n{'='*55}")
    print(f"  {len(dv)} valid / {n} attempted ({len(dv)/n*100:.1f}%)  errors={n_err}")
    for col in ["capture_rate", "E_specific_GJ", "alpha_lean", "T_ic_C", "delta_alpha"]:
        v = dv[col]
        print(f"  {col:<20} {v.min():.3f} - {v.max():.3f}  (mean={v.mean():.3f})")
    print(f"  Saved: {out}")
    print("=" * 55)
    return dv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",         type=int,  default=10000)
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--out",       type=str,  default="data/batch.csv")
    p.add_argument("--wide",      action="store_true")
    p.add_argument("--check",     action="store_true")
    p.add_argument("--save-every",type=int,  default=500)
    args = p.parse_args()

    sanity_check()
    if not args.check:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        bounds = BOUNDS_WIDE if args.wide else BOUNDS_CORE
        generate(args.n, args.seed, out, bounds, args.save_every)


if __name__ == "__main__":
    main()
