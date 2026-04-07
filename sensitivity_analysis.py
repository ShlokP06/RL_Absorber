import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from tqdm import tqdm

from src.simulation import (
    run_absorber, run_stripper, flood_fraction, max_safe_L
)

PALETTE = {
    "capture"  : "#1565C0",   # deep blue
    "energy"   : "#C62828",   # deep red
    "flood"    : "#E65100",   # deep orange
    "alpha"    : "#2E7D32",   # deep green
    "neutral"  : "#546E7A",   # slate
    "bg"       : "#FAFAFA",
    "grid"     : "#ECEFF1",
    "text"     : "#212121",
    "positive" : "#1976D2",
    "negative" : "#D32F2F",
}
CMAP_CAPTURE = LinearSegmentedColormap.from_list(
    "cap", ["#B71C1C", "#EF9A9A", "#FAFAFA", "#90CAF9", "#0D47A1"])
CMAP_ENERGY  = LinearSegmentedColormap.from_list(
    "eng", ["#0D47A1", "#90CAF9", "#FAFAFA", "#FFCC80", "#E65100"])

NOMINAL = {
    "G_gas"     : 1.20,
    "L_liq"     : 5.00,
    "y_CO2_in"  : 0.13,
    "T_L_in_C"  : 40.0,
    "alpha_lean": 0.27,
    "T_ic_C"    : 38.0,
}

VARS = {
    "G_gas"     : dict(lo=0.60, hi=1.35, label="Gas flux G  (kg m⁻² s⁻¹)",
                       short="G_gas",   colour="#6A1B9A",
                       desc="Flue gas superficial mass flux"),
    "L_liq"     : dict(lo=2.00, hi=12.0, label="Solvent flux L  (kg m⁻² s⁻¹)",
                       short="L_liq",   colour="#1565C0",
                       desc="Lean solvent circulation rate"),
    "y_CO2_in"  : dict(lo=0.06, hi=0.20, label="CO₂ mole fraction y_in",
                       short="y_CO₂",   colour="#00695C",
                       desc="Flue gas CO₂ inlet concentration"),
    "T_L_in_C"  : dict(lo=30.0, hi=55.0, label="Lean solvent temp T_L_in  (°C)",
                       short="T_L_in",  colour="#E65100",
                       desc="Lean solvent inlet temperature"),
    "alpha_lean": dict(lo=0.18, hi=0.38, label="Lean loading α_lean  (mol/mol)",
                       short="α_lean",  colour="#C62828",
                       desc="CO₂ loading of lean solvent"),
    "T_ic_C"    : dict(lo=25.0, hi=50.0, label="Intercooler temp T_ic  (°C)",
                       short="T_ic",    colour="#0277BD",
                       desc="Mid-column intercooler setpoint"),
}

OUTPUTS = ["capture_rate", "E_specific_GJ", "flood_fraction", "alpha_rich"]
OUT_LABELS = {
    "capture_rate"  : "Capture rate  (%)",
    "E_specific_GJ" : "Specific energy  (GJ t⁻¹ CO₂)",
    "flood_fraction": "Flooding fraction  (—)",
    "alpha_rich"    : "Rich loading α_rich  (mol/mol)",
}

def simulate_point(G=None, L=None, y=None, TL=None, al=None, Tic=None):
    G   = G   if G   is not None else NOMINAL["G_gas"]
    L   = L   if L   is not None else NOMINAL["L_liq"]
    y   = y   if y   is not None else NOMINAL["y_CO2_in"]
    TL  = TL  if TL  is not None else NOMINAL["T_L_in_C"]
    al  = al  if al  is not None else NOMINAL["alpha_lean"]
    Tic = Tic if Tic is not None else NOMINAL["T_ic_C"]
    try:
        ff = flood_fraction(G, L, TL + 273.15, al)
        ab = run_absorber(G, L, y, TL + 273.15, al, T_ic_C=Tic)
        E  = run_stripper(ab["alpha_rich"], al, L,
                          T_rich_C=ab["T_L_bottom_C"], T_lean_in_C=TL)
        return dict(
            capture_rate   = ab["capture_rate"],
            E_specific_GJ  = E,
            flood_fraction = ff,
            alpha_rich     = ab["alpha_rich"],
        )
    except Exception:
        return None

def run_oat(n_points=60):
    results = {}
    nom_result = simulate_point()
    for var, meta in tqdm(VARS.items(), desc="OAT sweep", unit="var"):
        sweep = np.linspace(meta["lo"], meta["hi"], n_points)
        rows  = []
        for val in sweep:
            kwargs = {
                "G":   NOMINAL["G_gas"],
                "L":   NOMINAL["L_liq"],
                "y":   NOMINAL["y_CO2_in"],
                "TL":  NOMINAL["T_L_in_C"],
                "al":  NOMINAL["alpha_lean"],
                "Tic": NOMINAL["T_ic_C"],
            }
            key_map = {"G_gas":"G","L_liq":"L","y_CO2_in":"y",
                       "T_L_in_C":"TL","alpha_lean":"al","T_ic_C":"Tic"}
            kwargs[key_map[var]] = val
            r = simulate_point(**kwargs)
            if r is not None:
                rows.append({var: val, **r})
            else:
                rows.append({var: val, "capture_rate": np.nan,
                             "E_specific_GJ": np.nan,
                             "flood_fraction": np.nan, "alpha_rich": np.nan})
        results[var] = pd.DataFrame(rows)
    return results, nom_result

def plot_oat(oat_results, nom_result, out_path: Path):
    var_list = list(VARS.keys())
    n_vars   = len(var_list)
    fig = plt.figure(figsize=(22, 13), facecolor=PALETTE["bg"])
    fig.suptitle(
        "One-At-a-Time (OAT) Sensitivity Analysis — MEA CO₂ Capture",
        fontsize=15, fontweight="bold", color=PALETTE["text"], y=0.98
    )

    gs = gridspec.GridSpec(3, n_vars, figure=fig,
                           hspace=0.55, wspace=0.38,
                           top=0.93, bottom=0.07,
                           left=0.05, right=0.98)

    row_meta = [
        ("capture_rate",   PALETTE["capture"], OUT_LABELS["capture_rate"],
         (0, 105), [85, 90]),
        ("E_specific_GJ",  PALETTE["energy"],  OUT_LABELS["E_specific_GJ"],
         (2, 12), []),
        ("flood_fraction", PALETTE["flood"],   OUT_LABELS["flood_fraction"],
         (0, 1.3), [0.79]),
    ]
    for row, (out_col, colour, ylabel, ylim, hlines) in enumerate(row_meta):
        for col, var in enumerate(var_list):
            meta = VARS[var]
            ax   = fig.add_subplot(gs[row, col])
            df   = oat_results[var].dropna(subset=[out_col])

            ax.plot(df[var], df[out_col], color=colour,
                    lw=2.0, zorder=3)
            ax.fill_between(df[var], df[out_col],
                            alpha=0.08, color=colour, zorder=2)
            nom_x = NOMINAL[var]
            nom_y = nom_result[out_col]
            ax.axvline(nom_x, color=PALETTE["neutral"],
                       lw=1.2, ls="--", alpha=0.7, zorder=4)
            ax.scatter([nom_x], [nom_y], color=PALETTE["neutral"],
                       s=30, zorder=5, edgecolors="white", linewidths=0.8)
            for h in hlines:
                ax.axhline(h, color="black" if out_col=="flood_fraction"
                           else "green",
                           lw=1.0, ls=":", alpha=0.6)
            ax.set_ylim(ylim)
            ax.set_xlim(meta["lo"], meta["hi"])
            ax.set_facecolor(PALETTE["bg"])
            ax.grid(True, color=PALETTE["grid"], lw=0.7, ls="--", zorder=1)
            ax.tick_params(colors=PALETTE["text"], labelsize=7.5)
            for sp in ax.spines.values():
                sp.set_edgecolor(PALETTE["grid"])
            if row == 0:
                ax.set_title(meta["label"], fontsize=8.5,
                             fontweight="bold", color=meta["colour"], pad=5)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=PALETTE["text"])
            if row == 2:
                ax.set_xlabel(meta["label"].split("(")[0].strip(),
                              fontsize=7.5, color=PALETTE["text"])
    row_labels = ["Capture Rate", "Specific Energy", "Flood Fraction"]
    for i, lbl in enumerate(row_labels):
        fig.text(0.002, 0.80 - i*0.30, lbl,
                 fontsize=9, fontweight="bold",
                 color=row_meta[i][1], rotation=90,
                 va="center", ha="center")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"  OAT plot saved → {out_path}")

PAIRS = [
    ("G_gas", "L_liq",
     "Gas flux vs Solvent flux\n(Flooding boundary & L/G trade-off)"),
    ("L_liq", "alpha_lean",
     "Solvent flux vs Lean loading\n(Energy–capture trade-off surface)"),
    ("T_L_in_C", "T_ic_C",
     "Lean temp vs Intercooler temp\n(Thermal synergy)"),
    ("G_gas", "y_CO2_in",
     "Gas flux vs CO₂ concentration\n(Combined load–composition effect)"),
]

def run_heatmaps(n_grid=35):
    results = []
    for var_x, var_y, title in tqdm(PAIRS, desc="Heatmaps", unit="pair"):
        mx = VARS[var_x]; my = VARS[var_y]
        xv = np.linspace(mx["lo"], mx["hi"], n_grid)
        yv = np.linspace(my["lo"], my["hi"], n_grid)
        cap_g = np.full((n_grid, n_grid), np.nan)
        eng_g = np.full((n_grid, n_grid), np.nan)
        ff_g  = np.full((n_grid, n_grid), np.nan)
        key_map = {"G_gas":"G","L_liq":"L","y_CO2_in":"y",
                   "T_L_in_C":"TL","alpha_lean":"al","T_ic_C":"Tic"}
        for i, xi in enumerate(xv):
            for j, yj in enumerate(yv):
                kwargs = {
                    "G":   NOMINAL["G_gas"],    "L":  NOMINAL["L_liq"],
                    "y":   NOMINAL["y_CO2_in"], "TL": NOMINAL["T_L_in_C"],
                    "al":  NOMINAL["alpha_lean"],"Tic":NOMINAL["T_ic_C"],
                }
                kwargs[key_map[var_x]] = xi
                kwargs[key_map[var_y]] = yj
                r = simulate_point(**kwargs)
                if r is not None:
                    cap_g[j, i] = r["capture_rate"]
                    eng_g[j, i] = r["E_specific_GJ"]
                    ff_g[j, i]  = r["flood_fraction"]
        results.append((var_x, var_y, xv, yv, cap_g, eng_g, ff_g, title))
    return results

def plot_heatmaps(heatmap_data, out_path: Path):
    n_pairs = len(heatmap_data)
    fig, axes = plt.subplots(2, n_pairs, figsize=(22, 10),
                             facecolor=PALETTE["bg"])
    fig.suptitle(
        "Two-Variable Interaction Heatmaps — MEA CO₂ Capture",
        fontsize=15, fontweight="bold", color=PALETTE["text"], y=0.99
    )
    for col, (var_x, var_y, xv, yv, cap_g, eng_g, ff_g, title) \
            in enumerate(heatmap_data):
        mx = VARS[var_x]; my = VARS[var_y]
        for row, (grid, cmap, label, levels) in enumerate([
            (cap_g, CMAP_CAPTURE, "Capture rate  (%)",
             [70, 80, 85, 90, 95]),
            (eng_g, CMAP_ENERGY,  "Specific energy  (GJ t⁻¹)",
             [3.5, 4.0, 4.5, 5.0, 6.0]),
        ]):
            ax  = axes[row, col]
            ax.set_facecolor("#CCCCCC") 
            im = ax.pcolormesh(xv, yv, grid,
                               cmap=cmap, shading="auto",
                               vmin=np.nanpercentile(grid, 2),
                               vmax=np.nanpercentile(grid, 98))
            valid = ~np.isnan(grid)
            if valid.sum() > 9:
                try:
                    cs = ax.contour(xv, yv, grid, levels=levels,
                                    colors="white", linewidths=0.8, alpha=0.7)
                    ax.clabel(cs, fmt="%.0f" if row==0 else "%.1f",
                              fontsize=7, colors="white")
                except Exception:
                    pass
            ax.scatter([NOMINAL[var_x]], [NOMINAL[var_y]],
                       marker="*", s=150, color="yellow",
                       edgecolors="black", linewidths=0.8,
                       zorder=5, label="Nominal")
            if row == 0 and var_x == "G_gas" and var_y == "L_liq":
                L_max_vals = [max_safe_L(g, 313.15, 0.27, limit=0.79)
                              for g in xv]
                ax.plot(xv, L_max_vals, "r--", lw=2, label="Flood limit")
                ax.legend(fontsize=7, loc="upper left",
                          framealpha=0.8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(label, fontsize=7.5, color=PALETTE["text"])
            ax.set_xlabel(mx["label"], fontsize=8, color=mx["colour"])
            ax.set_ylabel(my["label"], fontsize=8, color=my["colour"])
            ax.tick_params(labelsize=7.5)
            for sp in ax.spines.values():
                sp.set_edgecolor(PALETTE["grid"])
            if row == 0:
                ax.set_title(title, fontsize=9, fontweight="bold",
                             color=PALETTE["text"], pad=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"  Heatmap plot saved → {out_path}")

def run_tornado(pct=0.20):
    nom = simulate_point()
    rows = []
    for var, meta in VARS.items():
        nom_val = NOMINAL[var]
        key_map = {"G_gas":"G","L_liq":"L","y_CO2_in":"y",
                   "T_L_in_C":"TL","alpha_lean":"al","T_ic_C":"Tic"}
        k = key_map[var]
        lo_val = np.clip(nom_val * (1 - pct), meta["lo"], meta["hi"])
        hi_val = np.clip(nom_val * (1 + pct), meta["lo"], meta["hi"])
        r_lo = simulate_point(**{k: lo_val})
        r_hi = simulate_point(**{k: hi_val})
        for out in ["capture_rate", "E_specific_GJ"]:
            base = nom[out]
            val_lo = (r_lo[out] - base) if r_lo else np.nan
            val_hi = (r_hi[out] - base) if r_hi else np.nan
            pct_lo = val_lo / base * 100
            pct_hi = val_hi / base * 100
            swing  = abs(pct_hi - pct_lo)
            rows.append(dict(
                variable    = meta["short"],
                description = meta["desc"],
                output      = out,
                colour      = meta["colour"],
                pct_lo      = pct_lo,
                pct_hi      = pct_hi,
                swing       = swing,
                nom_val     = nom_val,
                lo_val      = lo_val,
                hi_val      = hi_val,
            ))
    return pd.DataFrame(rows)

def plot_tornado(tornado_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             facecolor=PALETTE["bg"])
    fig.suptitle(
        f"Tornado Chart — Sensitivity to ±20% Variable Perturbation\n"
        f"Nominal: G=1.2, L=5.0, y=0.13, T=40°C, α=0.27, T_ic=38°C",
        fontsize=13, fontweight="bold", color=PALETTE["text"], y=0.99
    )
    outputs = [
        ("capture_rate",  "CO₂ Capture Rate",     PALETTE["capture"]),
        ("E_specific_GJ", "Specific Energy (GJ/t)", PALETTE["energy"]),
    ]
    for ax, (out_col, out_title, bar_colour) in zip(axes, outputs):
        df = tornado_df[tornado_df["output"] == out_col].copy()
        df = df.sort_values("swing", ascending=True)   # largest at top
        y_pos  = np.arange(len(df))
        labels = df["variable"].tolist()
        for i, (_, row) in enumerate(df.iterrows()):
            lo = row["pct_lo"]
            hi = row["pct_hi"]
            if lo < 0:
                ax.barh(i, lo, left=0, height=0.6,
                        color=PALETTE["negative"], alpha=0.85,
                        edgecolor="white", linewidth=0.5)
            else:
                ax.barh(i, lo, left=0, height=0.6,
                        color=PALETTE["positive"], alpha=0.85,
                        edgecolor="white", linewidth=0.5)
            if hi > 0:
                ax.barh(i, hi, left=0, height=0.6,
                        color=PALETTE["positive"], alpha=0.85,
                        edgecolor="white", linewidth=0.5)
            else:
                ax.barh(i, hi, left=0, height=0.6,
                        color=PALETTE["negative"], alpha=0.85,
                        edgecolor="white", linewidth=0.5)
            offset = max(abs(lo), abs(hi)) * 0.06
            ax.text(lo - offset, i, f"{lo:+.1f}%",
                    va="center", ha="right", fontsize=8.5,
                    color=PALETTE["negative"] if lo < 0 else PALETTE["positive"],
                    fontweight="bold")
            ax.text(hi + offset, i, f"{hi:+.1f}%",
                    va="center", ha="left", fontsize=8.5,
                    color=PALETTE["negative"] if hi < 0 else PALETTE["positive"],
                    fontweight="bold")
            ax.text(ax.get_xlim()[1] * 0.99 if ax.get_xlim()[1] > 0 else 0,
                    i, f"Δ={row['swing']:.1f}%",
                    va="center", ha="right", fontsize=7.5,
                    color=PALETTE["neutral"], style="italic")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10, color=PALETTE["text"],
                           fontweight="bold")
        ax.axvline(0, color=PALETTE["text"], lw=1.2, zorder=5)
        ax.set_xlabel("Change from nominal (%)", fontsize=10,
                      color=PALETTE["text"])
        ax.set_title(out_title, fontsize=12, fontweight="bold",
                     color=bar_colour, pad=10)
        ax.set_facecolor(PALETTE["bg"])
        ax.grid(True, axis="x", color=PALETTE["grid"],
                lw=0.7, ls="--", zorder=1)
        ax.tick_params(colors=PALETTE["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["grid"])
        var_colours = df["colour"].tolist()
        for lbl, clr in zip(ax.get_yticklabels(), var_colours):
            lbl.set_color(clr)
    legend_elements = [
        Patch(facecolor=PALETTE["positive"], alpha=0.85,
              label="+20% perturbation"),
        Patch(facecolor=PALETTE["negative"], alpha=0.85,
              label="−20% perturbation"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"Tornado plot saved : {out_path}")

def print_summary(oat_results, nom_result, tornado_df):
    """Print a clean console summary of key sensitivity findings."""
    print("\n" + "═"*65)
    print("  SENSITIVITY ANALYSIS SUMMARY")
    print("═"*65)
    print(f"\nNominal operating point:")
    for k, v in NOMINAL.items():
        print(f"    {k:<15} = {v}")
    print(f"\nNominal outputs:")
    print(f"Capture rate = {nom_result['capture_rate']:.1f}%")
    print(f"Energy = {nom_result['E_specific_GJ']:.3f} GJ/t")
    print(f"Flood fraction = {nom_result['flood_fraction']:.3f}")
    print(f"Alpha rich = {nom_result['alpha_rich']:.3f}")
    print(f"\nTornado — ranked by impact on capture rate (±20%):")
    cap_df = tornado_df[tornado_df["output"]=="capture_rate"] \
             .sort_values("swing", ascending=False)
    print(f"  {'Variable':<12}  {'−20%':>8}  {'+20%':>8}  {'Swing':>8}")
    print("  " + "-"*44)
    for _, r in cap_df.iterrows():
        print(f"  {r['variable']:<12}  {r['pct_lo']:>+8.1f}%"
              f"  {r['pct_hi']:>+8.1f}%  {r['swing']:>8.1f}%")
    print(f"\n  Tornado — ranked by impact on specific energy (±20%):")
    eng_df = tornado_df[tornado_df["output"]=="E_specific_GJ"] \
             .sort_values("swing", ascending=False)
    print(f"  {'Variable':<12}  {'−20%':>8}  {'+20%':>8}  {'Swing':>8}")
    print("  " + "-"*44)
    for _, r in eng_df.iterrows():
        print(f"  {r['variable']:<12}  {r['pct_lo']:>+8.1f}%"
              f"  {r['pct_hi']:>+8.1f}%  {r['swing']:>8.1f}%")
    print("\n  Key findings:")
    top_cap = cap_df.iloc[0]
    top_eng = eng_df.iloc[0]
    print(f"  • {top_cap['variable']} has the largest impact on capture "
          f"rate (swing={top_cap['swing']:.1f}%)")
    print(f"  • {top_eng['variable']} has the largest impact on energy "
          f"(swing={top_eng['swing']:.1f}%)")
    oat_L = oat_results["L_liq"].dropna()
    above_90 = oat_L[oat_L["capture_rate"] >= 90.0]
    if len(above_90) > 0:
        L_90 = above_90["L_liq"].min()
        print(f"  • Minimum L_liq for ≥90% capture: {L_90:.1f} kg/m²/s "
              f"(nominal: {NOMINAL['L_liq']:.1f})")
    print("═"*65)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points",   type=int,  default=60,
                   help="Points per OAT sweep (default: 60)")
    p.add_argument("--grid",     type=int,  default=35,
                   help="Grid size for heatmaps (default: 35×35)")
    p.add_argument("--tornado-pct", type=float, default=0.20,
                   help="Perturbation fraction for tornado (default: 0.20)")
    p.add_argument("--no-plot",  action="store_true",
                   help="Skip plot generation, console output only")
    args = p.parse_args()
    Path("results").mkdir(exist_ok=True)
    print("MEA CCU Sensitivity Analysis")
    print("="*65)
    print("\n[1/3] Running one-at-a-time (OAT) sweeps...")
    oat_results, nom_result = run_oat(n_points=args.points)
    frames = []
    for var, df in oat_results.items():
        df["swept_variable"] = var
        frames.append(df)
    pd.concat(frames).to_csv("results/sensitivity_data.csv", index=False)
    print(f"  OAT data saved → results/sensitivity_data.csv")
    if not args.no_plot:
        plot_oat(oat_results, nom_result,
                 Path("results/sensitivity_oat.png"))
    print("\n[2/3] Running two-variable interaction heatmaps...")
    heatmap_data = run_heatmaps(n_grid=args.grid)
    if not args.no_plot:
        plot_heatmaps(heatmap_data,
                      Path("results/sensitivity_heatmaps.png"))
    print("\n[3/3] Running tornado analysis...")
    tornado_df = run_tornado(pct=args.tornado_pct)
    tornado_df.to_csv("results/sensitivity_tornado.csv", index=False)
    if not args.no_plot:
        plot_tornado(tornado_df,
                     Path("results/sensitivity_tornado.png"))
    print_summary(oat_results, nom_result, tornado_df)
    print("\n  All outputs saved to results/")
    print("sensitivity_oat.png — OAT sweep curves")
    print("sensitivity_heatmaps.png — interaction heatmaps")
    print("sensitivity_tornado.png — tornado chart")
    print("sensitivity_data.csv — full numerical data")

if __name__ == "__main__":
    main()