# Adsorber — RL Control for MEA CO₂ Capture

A reinforcement learning controller for optimizing **post-combustion CO₂ capture** using a monoethanolamine (MEA) absorber-stripper system. A PPO-LSTM agent learns to maximize capture rate while minimizing energy consumption, outperforming a classical PID baseline across all disturbance scenarios.

---

## Results

| Metric | PPO-LSTM | PID | Winner |
|---|---|---|---|
| Mean capture (post-disturbance) | 99.3% | 82.2% | RL |
| Recovery time | 0 steps | 45 steps | RL |
| Time ≥ 90% capture | 100% | 48.9% | RL |
| Mean energy | 3.87 GJ/t | 3.66 GJ/t | PID |
| Max flood fraction | 0.790 | 0.767 | — |

---

## Quick Start

### Option A — Docker (recommended, one command)

```bash
git clone https://github.com/ShlokP06/RL_Adsorption.git
cd RL_Adsorption
docker compose up
```

Models (~270 MB) are downloaded automatically on first run from the GitHub Release.

Open **http://localhost:3000** for the live demo dashboard.

**GPU inference** (requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
USE_CUDA=1 docker compose up
```

---

### Option B — Manual setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download pre-trained models**

```bash
python download_models.py
```

This downloads `models/` (~270 MB) from the GitHub Release. Skip this if you want to train from scratch.

**3. Run the demo backend**

```bash
cd demo/backend
python -m uvicorn main:app --reload --port 8000
```

**4. Run the demo frontend** (separate terminal)

```bash
cd demo/frontend
npm install
npm run dev
```

Open **http://localhost:3000**.

---

## Training from Scratch

### 1. Generate simulation data

Uses Latin Hypercube Sampling over the operating space:

```bash
python generate_data.py --n 10000 --seed 101 --out data/batch1.csv
python generate_data.py --n 10000 --seed 404 --out data/batch2.csv --wide
```

`--wide` expands sampling bounds for robustness at extremes.

### 2. Merge datasets

```bash
python merge_data.py --files "data/batch*.csv" --out data/merged_ccu.csv
```

### 3. Train the surrogate model

Trains a 4-layer MLP targeting R² ≥ 0.99 on all outputs:

```bash
python train_surrogate.py --data data/merged_ccu.csv --width 128
```

Outputs: `models/surrogate/model.pt`, `models/surrogate/scalers.pkl`

### 4. Train the RL controller

```bash
python train_rl.py --timesteps 2000000 --n-envs 8
```

Training uses **curriculum learning** across 3 phases:
- **Phase 0** (0 → 300k steps) — frozen disturbances, random initial conditions
- **Phase 1** (300k → 700k) — Ornstein-Uhlenbeck drift on G_gas and y_CO₂_in
- **Phase 2** (700k → end) — full OU dynamics with random step changes

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/
```

### 5. Evaluate the trained agent

```bash
python train_rl.py --eval-only --model models/rl/best/best_model.zip
```

### 6. Compare RL vs PID

```bash
python compare_controllers.py                # all 3 scenarios
python compare_controllers.py --scenario 1   # step in G_gas
python compare_controllers.py --scenario 2   # step in y_CO₂_in
python compare_controllers.py --scenario 3   # combined disturbance
```

Outputs a 16-panel dashboard PNG and CSV to `results/`.

### 7. Sensitivity analysis

```bash
python sensitivity_analysis.py
```

---

## Project Structure

```
Adsorber/
├── src/
│   ├── simulation.py        # Rate-based absorber-stripper physics
│   ├── surrogate.py         # Neural surrogate (6 inputs → 3 outputs)
│   └── env.py               # Gymnasium RL environment (17-dim obs, 4-dim action)
├── demo/
│   ├── backend/             # FastAPI server (WebSocket + REST)
│   └── frontend/            # React + Vite dashboard
├── models/                  # Downloaded via download_models.py (git-ignored)
├── data/                    # Training data (git-ignored)
├── results/                 # Eval plots and CSVs (git-ignored)
├── generate_data.py         # LHS data generation
├── merge_data.py            # Dataset merging
├── train_surrogate.py       # Surrogate training
├── train_rl.py              # PPO-LSTM training
├── compare_controllers.py   # RL vs PID benchmark
├── sensitivity_analysis.py  # OAT sensitivity sweeps
├── download_models.py       # Fetch models from GitHub Release
├── Dockerfile               # Backend container
└── docker-compose.yml       # Full stack orchestration
```

---

## Architecture

### Surrogate model (`src/surrogate.py`)

| | |
|---|---|
| Inputs | G_gas, L_liq, y_CO₂_in, T_L_in, alpha_lean, T_ic |
| Outputs | capture_rate, E_specific, alpha_rich |
| Architecture | 4-layer MLP, ReLU, width 128 |
| Normalization | MinMax scaling |
| Loss | MSE, early stopping (patience 60) |

### RL environment (`src/env.py`)

- **Observation** (17-dim, normalized to [−1, 1]): gas state, OU trend signals, actual control values, actuator lag states, process outputs + derivatives/integrals, flood headroom, Pareto goal weight
- **Action** (4-dim): delta commands on L_liq, alpha_lean, T_L_in, T_ic
- **Hard constraint**: L_liq projected to `max_safe_L()` guaranteeing `flood_fraction < 0.79`
- **Actuator lag**: first-order lags τ = 2–5 steps per variable
- **Reward**: capture-first Pareto — quadratic capture term + above-target bonus (85→95%) − energy penalty − integral penalties − flood penalty

### PPO-LSTM policy (`train_rl.py`)

| Hyperparameter | Value |
|---|---|
| Algorithm | RecurrentPPO (sb3-contrib) |
| Policy | MlpLstmPolicy |
| LSTM hidden size | 256 units |
| Network | pi=[256,128], vf=[256,128] |
| n_steps | 256 |
| batch_size | 512 |
| Learning rate | 3×10⁻⁴ → 0 (linear decay) |
| gamma | 0.99 |
| clip_range | 0.2 |

### Control variables

| Variable | Range | Actuator lag τ |
|---|---|---|
| L_liq (solvent flowrate) | 2–12 kg/m²/s | 3 steps |
| alpha_lean (lean loading) | 0.18–0.38 mol/mol | 5 steps |
| T_L_in (solvent inlet temp) | 30–55 °C | 2 steps |
| T_ic (intercooling temp) | 25–50 °C | 4 steps |

---

## Demo API Endpoints

Once the backend is running at `http://localhost:8000`:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Reset both simulations |
| POST | `/attack` | Slam G_gas→1.5, y_CO₂→0.20 |
| POST | `/set_disturbance` | `{G_gas, y_CO2_in}` manual override |
| POST | `/clear_disturbance` | Remove manual override |
| POST | `/set_lambda` | `{lambda_energy}` live Pareto tuning |
| POST | `/toggle_controller` | Freeze/unfreeze RL agent |
| GET | `/state` | Current snapshot |
| GET | `/history` | Last 120 snapshots |
| WS | `/stream` | Push snapshot every 500 ms |
| GET | `/docs` | Swagger UI |
