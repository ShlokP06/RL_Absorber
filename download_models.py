"""
download_models.py
==================
Downloads pre-trained model artifacts from the GitHub Release.

Usage
-----
    python download_models.py

Downloads
---------
    models/rl/best/best_model.zip   PPO-LSTM checkpoint (~268 MB)
    models/rl/vecnorm.pkl           VecNormalize statistics
    models/surrogate/model.pt       Neural surrogate weights
    models/surrogate/scalers.pkl    MinMax scalers
"""

import sys
import zipfile
import urllib.request
from pathlib import Path

RELEASE_URL = (
    "https://github.com/ShlokP06/RL_Absorber/releases/download/v1.0/models.zip"
)

DEST = Path("models.zip")


def _progress(block: int, block_size: int, total: int) -> None:
    downloaded = block * block_size
    if total > 0:
        pct = min(downloaded / total * 100, 100)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
        sys.stdout.flush()


def main() -> None:
    models_dir = Path("models")
    if (models_dir / "rl" / "best" / "best_model.zip").exists():
        print("models/ already present — skipping download.")
        return

    print(f"Downloading models from GitHub Release...")
    print(f"  {RELEASE_URL}\n")

    urllib.request.urlretrieve(RELEASE_URL, DEST, reporthook=_progress)
    print()

    print("  Extracting...")
    rl_best_dir = models_dir / "rl" / "best"
    surrogate_dir = models_dir / "surrogate"
    rl_best_dir.mkdir(parents=True, exist_ok=True)
    surrogate_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "best_model.zip": rl_best_dir / "best_model.zip",
        "vecnorm.pkl":    models_dir / "rl" / "vecnorm.pkl",
        "model.pt":       surrogate_dir / "model.pt",
        "scalers.pkl":    surrogate_dir / "scalers.pkl",
    }

    with zipfile.ZipFile(DEST) as z:
        for name, dest_path in file_map.items():
            if name in z.namelist():
                dest_path.write_bytes(z.read(name))

    DEST.unlink()
    print("  Done. models/ is ready.\n")

    # Verify expected files exist
    required = [
        models_dir / "rl" / "best" / "best_model.zip",
        models_dir / "rl" / "vecnorm.pkl",
        models_dir / "surrogate" / "model.pt",
        models_dir / "surrogate" / "scalers.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("WARNING: These expected files are missing after extraction:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("All model files verified.")


if __name__ == "__main__":
    main()
