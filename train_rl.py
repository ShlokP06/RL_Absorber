"""
RL Training — RecurrentPPO (LSTM)
==================================
Trains a recurrent PPO agent on the CCU environment with curriculum learning.

Curriculum phases (auto-advance by timestep):
  Phase 0 (0 → phase1):      Frozen disturbances, random starts.
  Phase 1 (phase1 → phase2): OU drift, no step changes.
  Phase 2 (phase2 → end):    Full OU + step changes. Real-time controller.

Outputs (models/rl/)
---------------------
  ppo_ccu.zip      final model weights
  vecnorm.pkl      VecNormalize statistics (required for deployment)
  best/            best checkpoint by eval reward

Usage
-----
    python train_rl.py
    python train_rl.py --timesteps 1000000 --n-envs 8
    python train_rl.py --eval-only --model models/rl/best/best_model.zip
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback,
)
from sb3_contrib import RecurrentPPO

sys.path.insert(0, str(Path(__file__).parent))
from src.env import CCUEnv

RL_DIR = Path("models/rl")


# ── Learning rate schedule ───────────────────────────────────────────────────

def linear_schedule(initial_value: float):
    """Linear decay from initial_value → 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ── Picklable env factory (required for SubprocVecEnv on Windows) ────────────

def _make_env(**kwargs):
    return Monitor(CCUEnv(**kwargs))


# ── Curriculum callback ──────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):

    def __init__(self, phase1, phase2):
        super().__init__(verbose=0)
        self.phase1, self.phase2 = phase1, phase2
        self._phase = 0

    def _on_step(self):
        t = self.num_timesteps
        if self._phase == 0 and t >= self.phase1:
            self._phase = 1
            self.training_env.env_method("set_phase", 1)
            print(f"\n  Curriculum → Phase 1 (OU drift)  [{t:,} steps]\n")
        elif self._phase == 1 and t >= self.phase2:
            self._phase = 2
            self.training_env.env_method("set_phase", 2)
            print(f"\n  Curriculum → Phase 2 (step changes)  [{t:,} steps]\n")
        return True


# ── Evaluation (handles RecurrentPPO + VecEnv correctly) ─────────────────────

def evaluate(model, env, n_episodes=200):
    """
    Evaluate RecurrentPPO with proper LSTM state tracking.
    Uses all VecEnv workers in parallel for fast evaluation.
    """
    records = []

    # SB3 VecEnv.reset() returns ndarray (not tuple)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    n_envs = env.num_envs
    lstm_states = None
    episode_starts = np.ones(n_envs, dtype=bool)

    while len(records) < n_episodes:
        actions, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts,
            deterministic=True,
        )
        # SB3 VecEnv.step() returns 4 values: (obs, rewards, dones, infos)
        obs, rewards, dones, infos = env.step(actions)
        episode_starts = dones.copy()

        for i in range(n_envs):
            if dones[i] and len(records) < n_episodes:
                info = infos[i]
                rec = {}
                for k, v in info.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        rec[k] = float(v)
                if "episode" in info:
                    rec["ep_reward"] = float(info["episode"]["r"])
                records.append(rec)

    df = pd.DataFrame(records[:n_episodes])
    print("\n" + "=" * 55)
    print("EVALUATION")
    print("=" * 55)
    for col in ["capture_rate", "E_specific_GJ", "ep_reward"]:
        if col in df.columns:
            v = df[col]
            print(f"  {col:<20} mean={v.mean():.3f}  "
                  f"min={v.min():.3f}  max={v.max():.3f}")
    if "capture_rate" in df.columns:
        print(f"\n  ≥85% capture : {(df.capture_rate >= 85).mean() * 100:.1f}%")
        print(f"  ≥90% capture : {(df.capture_rate >= 90).mean() * 100:.1f}%")
    if "flood_fraction" in df.columns:
        print(f"  Flood events  : {(df.flood_fraction > 0.80).mean() * 100:.1f}%")
    print("=" * 55)
    return df


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    RL_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("\n  WARNING: CUDA not available — training on CPU (slow).")
        print("  Install PyTorch+CUDA: pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121\n")

    # DummyVecEnv is default: single-process, no pickling issues, stable on
    # all platforms. SubprocVecEnv available via --subproc for heavier envs.
    VecCls = SubprocVecEnv if args.subproc else DummyVecEnv

    env_kwargs = dict(
        model_path      = args.model_path,
        scaler_path     = args.scaler_path,
        max_steps       = args.max_steps,
        lambda_range    = (args.lam_min, args.lam_max),
        lam_smooth      = args.lam_smooth,
        lam_integral    = args.lam_I,
        lam_energy_int  = args.lam_Ie,
        lam_above       = args.lam_above,
        lam_flood       = args.lam_flood,
        step_prob       = args.step_prob,
        actuator_lag    = True,
        obs_noise       = True,
        domain_rand     = True,
        continue_prob   = 0.30,
        curriculum_phase= 0,
    )

    train_env = VecNormalize(
        make_vec_env(partial(_make_env, **env_kwargs),
                     n_envs=args.n_envs, vec_env_cls=VecCls),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
    )

    # Eval envs: always DummyVecEnv (lightweight, no subprocess overhead)
    eval_kwargs = {
        **env_kwargs,
        "step_prob": 0.0,
        "obs_noise": False,
        "domain_rand": False,
        "continue_prob": 0.0,
        "curriculum_phase": 2,
    }
    eval_env = VecNormalize(
        make_vec_env(partial(_make_env, **eval_kwargs),
                     n_envs=args.eval_envs, vec_env_cls=DummyVecEnv),
        norm_obs=True, norm_reward=False, clip_obs=10.0,
    )

    # batch_size must evenly divide n_envs * n_steps
    buffer_size = args.n_envs * args.n_steps
    batch_size = min(args.batch_size, buffer_size)
    while buffer_size % batch_size != 0:
        batch_size -= 1

    callbacks = [
        CurriculumCallback(args.phase1, args.phase2),
        EvalCallback(
            eval_env,
            best_model_save_path=str(RL_DIR / "best"),
            log_path="logs",
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(RL_DIR / "checkpoints"),
            name_prefix="ppo_ccu",
            verbose=1,
        ),
    ]

    model = RecurrentPPO(
        policy          = "MlpLstmPolicy",
        env             = train_env,
        learning_rate   = linear_schedule(args.lr),
        n_steps         = args.n_steps,
        batch_size      = batch_size,
        n_epochs        = args.n_epochs,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(
            lstm_hidden_size = args.lstm_hidden,
            n_lstm_layers    = args.lstm_layers,
            net_arch         = dict(pi=[256, 128], vf=[256, 128]),
        ),
        device          = device,
        tensorboard_log = "logs/",
        verbose         = 1,
        seed            = 42,
    )

    n_p = sum(p.numel() for p in model.policy.parameters())
    print(f"\n  RecurrentPPO  params={n_p:,}  device={device}"
          f"  envs={args.n_envs}  n_steps={args.n_steps}  batch={batch_size}")
    print(f"  LSTM: hidden={args.lstm_hidden}  layers={args.lstm_layers}"
          f"  timesteps={args.timesteps:,}\n")

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=True)
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed / 60:.1f} min")

    model.save(str(RL_DIR / "ppo_ccu"))
    train_env.save(str(RL_DIR / "vecnorm.pkl"))
    print(f"  Saved: {RL_DIR}/ppo_ccu.zip  +  vecnorm.pkl")

    df = evaluate(model, eval_env, n_episodes=300)
    df.to_csv("results/eval_results.csv", index=False)
    print("  Saved: results/eval_results.csv")


# ── Eval only ────────────────────────────────────────────────────────────────

def eval_only(args):
    eval_kwargs = dict(
        model_path=args.model_path, scaler_path=args.scaler_path,
        max_steps=args.max_steps,
        lambda_range=(args.lam_min, args.lam_max),
        step_prob=0.0, obs_noise=False, domain_rand=False,
        continue_prob=0.0, curriculum_phase=2,
    )
    venv = DummyVecEnv([partial(_make_env, **eval_kwargs)])

    vecnorm_path = Path(args.vecnorm)
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), venv)
        env.training = False
        env.norm_reward = False
        print(f"  Loaded VecNormalize stats: {vecnorm_path}")
    else:
        print(f"  WARNING: {vecnorm_path} not found — using unnormalized env")
        env = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = RecurrentPPO.load(args.model, env=env)
    print(f"  Loaded: {args.model}")
    df = evaluate(model, env, n_episodes=500)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/eval_results.csv", index=False)
    print("  Saved: results/eval_results.csv")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train RecurrentPPO on CCU env")

    # Paths
    p.add_argument("--model-path",  default="models/surrogate/model.pt")
    p.add_argument("--scaler-path", default="models/surrogate/scalers.pkl")

    # Environment
    p.add_argument("--max-steps",   type=int,   default=120)
    p.add_argument("--lam-min",     type=float, default=0.0)
    p.add_argument("--lam-max",     type=float, default=0.20)
    p.add_argument("--lam-smooth",  type=float, default=0.030)
    p.add_argument("--lam-I",       type=float, default=0.15)
    p.add_argument("--lam-Ie",      type=float, default=0.08)
    p.add_argument("--lam-above",   type=float, default=0.10)
    p.add_argument("--lam-flood",   type=float, default=0.15)
    p.add_argument("--step-prob",   type=float, default=0.04)

    # Curriculum
    p.add_argument("--phase1",      type=int,   default=200_000)
    p.add_argument("--phase2",      type=int,   default=600_000)

    # Training
    p.add_argument("--timesteps",   type=int,   default=2_000_000)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--n-envs",      type=int,   default=16)
    p.add_argument("--n-steps",     type=int,   default=256,
                   help="Steps per env per rollout (>= max_steps for full episodes)")
    p.add_argument("--batch-size",  type=int,   default=512,
                   help="Mini-batch size (auto-adjusted to divide buffer)")
    p.add_argument("--n-epochs",    type=int,   default=10)
    p.add_argument("--lstm-hidden", type=int,   default=256)
    p.add_argument("--lstm-layers", type=int,   default=1)
    p.add_argument("--subproc",     action="store_true",
                   help="Use SubprocVecEnv instead of DummyVecEnv")

    # Evaluation
    p.add_argument("--eval-freq",     type=int, default=25_000)
    p.add_argument("--eval-envs",     type=int, default=8)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--eval-only",     action="store_true")
    p.add_argument("--model",         default="models/rl/best/best_model.zip")
    p.add_argument("--vecnorm",       default="models/rl/vecnorm.pkl",
                   help="Path to VecNormalize stats for eval-only mode")

    args = p.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
