"""
Neural Surrogate Model
======================
Feedforward network: 6 steady-state inputs → 3 outputs.

Inputs  : G_gas, L_liq, y_CO2_in, T_L_in_C, alpha_lean, T_ic_C
Outputs : capture_rate (%), E_specific_GJ (GJ/t), alpha_rich
"""

import numpy as np
import torch
import torch.nn as nn
import joblib

X_COLS = ["G_gas_kg_m2s", "L_liq_kg_m2s", "y_CO2_in",
          "T_L_in_C", "alpha_lean", "T_ic_C"]
Y_COLS = ["capture_rate", "E_specific_GJ", "alpha_rich"]

X_BOUNDS = {
    "G_gas_kg_m2s": (0.40, 2.50),
    "L_liq_kg_m2s": (1.50, 15.0),
    "y_CO2_in":     (0.04, 0.22),
    "T_L_in_C":     (25.0, 60.0),
    "alpha_lean":   (0.15, 0.42),
    "T_ic_C":       (20.0, 55.0),
}

N_INPUTS = len(X_COLS)   # 6


class CCUSurrogate(nn.Module):

    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(N_INPUTS,   width),      nn.ReLU(),
            nn.Linear(width,      width),      nn.ReLU(),
            nn.Linear(width,      width // 2), nn.ReLU(),
            nn.Linear(width // 2, 3),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Output layer has no activation — use Xavier instead of Kaiming/ReLU
        nn.init.xavier_uniform_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SurrogatePredictor:
    """Inference wrapper. Auto-detects width from checkpoint."""

    def __init__(self, model_path="models/surrogate/model.pt",
                 scaler_path="models/surrogate/scalers.pkl"):
        self.sx, self.sy = joblib.load(scaler_path)

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
            width = ckpt.get("width", state["net.0.weight"].shape[0])
        else:
            state = ckpt
            width = state["net.0.weight"].shape[0]

        self.model = CCUSurrogate(width=width)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, G_gas, L_liq, y_CO2_in, T_L_in_C, alpha_lean, T_ic_C):
        x = np.array([[G_gas, L_liq, y_CO2_in, T_L_in_C, alpha_lean, T_ic_C]],
                     dtype="float32")
        y = self.sy.inverse_transform(
            self.model(
                torch.tensor(self.sx.transform(x).astype("float32"))
            ).numpy()
        )[0]
        return {
            "capture_rate":  float(np.clip(y[0],  0.0, 100.0)),
            "E_specific_GJ": float(np.clip(y[1],  0.5,  50.0)),
            "alpha_rich":    float(np.clip(y[2],  0.10,  0.60)),
        }
