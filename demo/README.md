# The Plant Under Attack — RL vs PID Demo

Interactive live demo comparing a **RecurrentPPO (LSTM)** agent against a **three-loop PID**
controller on a MEA post-combustion CO₂ capture column.

## Architecture

```
demo/
├── backend/        FastAPI + WebSocket server (Python)
│   ├── main.py         REST + WS endpoints
│   ├── demo_state.py   RL env + LSTM state + PID sim
│   └── pid.py          PID controller
└── frontend/       React + Recharts + D3 (Vite)
    └── src/
        ├── App.jsx
        ├── components/
        │   ├── MEAColumn.jsx          D3 temperature heatmap
        │   ├── TimeSeriesChart.jsx    Recharts dual-axis
        │   ├── DisturbanceControls.jsx
        │   ├── AgentActionsPanel.jsx
        │   └── BottomBar.jsx
        └── hooks/useWebSocket.js
```

## Prerequisites

Make sure the Adsorber project is set up:
```bash
# From project root (Adsorber/)
pip install -r requirements.txt              # existing project deps
pip install fastapi uvicorn[standard] websockets python-dotenv
```

The following model files must exist:
- `models/rl/best/best_model.zip`  — trained RecurrentPPO
- `models/rl/vecnorm.pkl`          — VecNormalize stats
- `models/surrogate/model.pt`      — surrogate weights
- `models/surrogate/scalers.pkl`   — MinMax scalers

Node.js ≥ 18 is required for the frontend.

## Running

### 1. Backend (Terminal 1)

```bash
cd demo/backend
pip install -r requirements.txt    # first time only
uvicorn main:app --reload --port 8000
```

Expected output:
```
INFO  Loading surrogate + RL model…
INFO  Models loaded. Starting simulation loop.
INFO  Application startup complete.
```

### 2. Frontend (Terminal 2)

```bash
cd demo/frontend
npm install          # first time only
npm run dev          # starts on http://localhost:3000
```

Open **http://localhost:3000** in your browser.

## Demo Controls

| Control | Effect |
|---|---|
| **⚡ Attack the Plant** | Slams G_gas → 2.5 kg/m²/s and y_CO₂_in → 0.20. RL adapts in seconds; PID struggles. |
| **⏸ Freeze Agent** | Switches RL env to PID control. LSTM state is preserved. When unfrozen, RL resumes from its last memory and recovers. |
| **↺ Reset** | Resets both simulations and all impact counters. |
| **G_gas slider** | Manual gas flux override (0.40–2.50 kg/m²/s). Drag to inject disturbance. |
| **y_CO₂ slider** | Manual inlet CO₂ composition override (0.04–0.22 mol/mol). |
| **Pareto λ slider** | Live-tunes the energy penalty weight in the RL agent's reward (0–0.20). Right = more energy-efficient, left = more capture. |

## WebSocket API

The backend pushes JSON snapshots at 2 Hz (`ws://localhost:8000/stream`):

```json
{
  "t": 143,
  "rl":  { "cap": 91.2, "eng": 3.87, "G": 1.05, "y": 0.091, "L": 6.3, "al": 0.26, "T": 38.1, "ic": 34.5, "ff": 0.52, "action": [0.12, -0.05, -0.08, 0.0] },
  "pid": { "cap": 82.7, "eng": 4.21, "L": 7.1, "al": 0.25, "T": 37.4, "ic": 38.0 },
  "frozen": false,
  "lambda_energy": 0.05,
  "impact": {
    "co2_captured_rl_t": 0.0142,
    "co2_delta_t": 0.0031,
    "energy_kwh_saved": 0.89,
    "money_saved_usd": 0.09,
    "trees_equivalent": 0.14
  }
}
```

## REST API

| Method | Path | Body | Description |
|---|---|---|---|
| POST | `/reset` | — | Reset both sims |
| POST | `/attack` | — | Slam G_gas to max |
| POST | `/set_disturbance` | `{G_gas, y_CO2_in}` | Override disturbances |
| POST | `/clear_disturbance` | — | Remove override |
| POST | `/freeze` | `{frozen: bool}` | Freeze/unfreeze RL agent |
| POST | `/toggle_controller` | — | Toggle freeze state |
| POST | `/set_lambda` | `{lambda_energy}` | Live Pareto tuning |
| GET  | `/state` | — | Current snapshot |
| GET  | `/history` | — | Last 60 s of snapshots |

## Panel Layout

```
┌────────────────────────────────────────────────────────────────┐
│ ⚡ The Plant Under Attack — MEA CO₂ Capture RL vs PID      LIVE │
├─────────────┬──────────────────────┬───────────┬──────────────┤
│ ABSORBER    │ RL vs PID CAPTURE    │DISTURBANCE│ AGENT ACTIONS│
│ COLUMN      │ & ENERGY (live 60s)  │ CONTROLS  │ RL / PID     │
│ (D3 heatmap)│ (Recharts dual-axis) │ (sliders) │ (bar charts) │
├─────────────┴──────────────────────┴───────────┴──────────────┤
│ ⚡ATTACK  ⏸FREEZE  ↺RESET  │ λ slider  │ Pareto scatter      │
├────────────────────────────────────────────────────────────────┤
│  kg CO₂ (RL) │ +Δ CO₂ vs PID │ kWh saved │ $ saved │ 🌲trees │
└────────────────────────────────────────────────────────────────┘
```
