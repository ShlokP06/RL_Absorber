import argparse
from pathlib import Path
import pandas as pd

X_COLS = ["G_gas_kg_m2s", "L_liq_kg_m2s", "y_CO2_in",
          "T_L_in_C", "alpha_lean", "T_ic_C"]
Y_COLS = ["capture_rate", "E_specific_GJ", "alpha_rich"]
KEEP   = X_COLS + Y_COLS + [
    "delta_alpha", "T_L_bottom_C", "y_CO2_out",
    "T_reb_C", "flood_fraction", "LG_ratio", "Ha_avg", "E_factor_avg",
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--out",   default="data/ccu_merged.csv")
    args = p.parse_args()

    import glob
    all_files = []
    for pattern in args.files:
        expanded = glob.glob(pattern)
        if expanded:
            all_files.extend(sorted(expanded))
        else:
            all_files.append(pattern)   
    if not all_files:
        print("ERROR: No files found matching the given patterns.")
        return

    frames = []
    for path in all_files:
        df = pd.read_csv(path)
        n_before = len(df)
        if "valid" in df.columns:
            df = df[df["valid"] == True]
        frames.append(df)
        print(f"  {Path(path).name:<30} {n_before:>6} rows  ->  {len(df):>6} valid")

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Total before dedup : {len(merged):,}")
    merged = merged.drop_duplicates(subset=X_COLS).dropna(subset=X_COLS + Y_COLS)
    print(f"  After dedup/dropna : {len(merged):,}")
    cols = [c for c in KEEP if c in merged.columns]
    merged = merged[cols].reset_index(drop=True)
    print(f"\n  Output ranges:")
    for col in X_COLS + Y_COLS:
        v = merged[col]
        print(f"    {col:<20} {v.min():.4f} - {v.max():.4f}  (mean={v.mean():.4f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\n  Saved {len(merged):,} points -> {out}")

if __name__ == "__main__":
    main()