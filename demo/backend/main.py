"""
FastAPI backend for the MEA Absorber RL Demo — "The Plant Under Attack".

Run from demo/backend/:
    uvicorn main:app --reload --port 8000

Endpoints
---------
POST /reset              — reset both sims, return initial snapshot
POST /step               — manual single step (normally auto-stepped)
POST /set_disturbance    — {G_gas, y_CO2_in} override
POST /clear_disturbance  — remove manual override
POST /attack             — slam G_gas→2.5, y_CO2_in→0.20
POST /set_lambda         — {lambda_energy} live Pareto tuning
POST /freeze             — {frozen: bool}
POST /toggle_controller  — toggle frozen state
GET  /state              — current snapshot
GET  /history            — last 120 snapshots
WS   /stream             — push snapshot every 500 ms
"""

from __future__ import annotations

import sys
import os
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Set

# ── sys.path before any src imports ─────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from demo_state import DemoState  # noqa: E402

# --------------------------------------------------------------------------- #
# Logging                                                                       #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("adsorber_demo")

# --------------------------------------------------------------------------- #
# Configuration                                                                 #
# --------------------------------------------------------------------------- #

CONFIG = {
    "model_path":     os.getenv("MODEL_PATH",     "models/rl/best/best_model.zip"),
    "vecnorm_path":   os.getenv("VECNORM_PATH",   "models/rl/vecnorm.pkl"),
    "surrogate_path": os.getenv("SURROGATE_PATH", "models/surrogate/model.pt"),
    "scalers_path":   os.getenv("SCALERS_PATH",   "models/surrogate/scalers.pkl"),
}

STEP_INTERVAL: float = float(os.getenv("STEP_INTERVAL", "0.5"))

# --------------------------------------------------------------------------- #
# Global state (single DemoState instance, guarded by asyncio.Lock)            #
# --------------------------------------------------------------------------- #

demo: DemoState | None = None
ws_clients: Set[WebSocket] = set()
sim_lock = asyncio.Lock()

# --------------------------------------------------------------------------- #
# Background simulation loop                                                    #
# --------------------------------------------------------------------------- #

async def simulation_loop() -> None:
    while True:
        await asyncio.sleep(STEP_INTERVAL)
        if demo is None:
            continue

        async with sim_lock:
            try:
                snapshot = demo.step()
            except Exception:
                logger.exception("Simulation step failed")
                continue

        if not ws_clients:
            continue

        payload = json.dumps(snapshot)
        dead: Set[WebSocket] = set()
        for ws in list(ws_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        ws_clients.difference_update(dead)

# --------------------------------------------------------------------------- #
# App lifecycle                                                                 #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global demo
    logger.info("Loading surrogate + RL model…")
    demo = DemoState(CONFIG)
    logger.info("Models loaded. Starting simulation loop.")
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Simulation loop stopped.")


app = FastAPI(title="Adsorber RL Demo — The Plant Under Attack", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Pydantic request schemas                                                      #
# --------------------------------------------------------------------------- #

class DisturbanceRequest(BaseModel):
    G_gas:    float
    y_CO2_in: float

class LambdaRequest(BaseModel):
    lambda_energy: float

class FreezeRequest(BaseModel):
    frozen: bool

# --------------------------------------------------------------------------- #
# REST endpoints                                                                #
# --------------------------------------------------------------------------- #

@app.post("/reset")
async def reset_env():
    async with sim_lock:
        snap = demo.reset()
    return snap


@app.post("/step")
async def manual_step():
    """Single manual step — mostly for debugging."""
    async with sim_lock:
        snap = demo.step()
    return snap


@app.post("/set_disturbance")
async def set_disturbance(req: DisturbanceRequest):
    async with sim_lock:
        demo.set_disturbance(req.G_gas, req.y_CO2_in)
    return {"status": "ok", "G_gas": req.G_gas, "y_CO2_in": req.y_CO2_in}


@app.post("/clear_disturbance")
async def clear_disturbance():
    async with sim_lock:
        demo.clear_disturbance()
    return {"status": "ok"}


@app.post("/attack")
async def attack_plant():
    """Push G_gas to 1.50 and y_CO2_in to 0.20 — the 'wow moment'."""
    async with sim_lock:
        demo.attack_plant()
    return {"status": "ok", "G_gas": 1.50, "y_CO2_in": 0.20}


@app.post("/reset_impact")
async def reset_impact():
    """Zero impact counters without resetting the simulation."""
    async with sim_lock:
        demo.reset_impact()
    return {"status": "ok"}


@app.post("/set_lambda")
async def set_lambda(req: LambdaRequest):
    async with sim_lock:
        demo.set_lambda(req.lambda_energy)
    return {"status": "ok", "lambda_energy": req.lambda_energy}


@app.post("/freeze")
async def freeze_agent(req: FreezeRequest):
    async with sim_lock:
        demo.freeze_agent(req.frozen)
    return {"status": "ok", "frozen": req.frozen}


@app.post("/toggle_controller")
async def toggle_controller():
    async with sim_lock:
        demo.freeze_agent(not demo.frozen)
        frozen = demo.frozen
    return {"status": "ok", "frozen": frozen}


@app.get("/state")
async def get_state():
    return demo.get_snapshot()


@app.get("/history")
async def get_history():
    return demo.get_history()


@app.get("/health")
async def health():
    return {"status": "ok"}

# --------------------------------------------------------------------------- #
# WebSocket                                                                     #
# --------------------------------------------------------------------------- #

@app.websocket("/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    logger.info("WebSocket connected — %d clients", len(ws_clients))

    try:
        # Send current state immediately on connect
        snap = demo.get_snapshot()
        await ws.send_text(json.dumps(snap))

        # Keep alive — browser will send pings
        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error")
    finally:
        ws_clients.discard(ws)
        logger.info("WebSocket disconnected — %d clients", len(ws_clients))
