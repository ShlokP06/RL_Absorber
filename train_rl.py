"""
RL Training — PPO-LSTM
======================
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
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback,
)
from sb3_contrib import RecurrentPPO

sys.path.insert(0, str(Path(__file__).parent))
from src.env import CCUEnv

RL_DIR = Path("models/rl")


# ── Curriculum callback ───────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):

    def __init__(self, phase1, phase2):
        super().__init__(verbose=0)
        self.phase1, self.phase2 = phase1, phase2
        self._phase = 0

    def _set_phase(self, p):
        for env in self.training_env.envs:
            inner = getattr(env, "env", env)
            if hasattr(inner, "set_phase"):
                inner.set_phase(p)

    def _on_step(self):
        t = self.num_timesteps
        if self._phase == 0 and t >= self.phase1:
            self._phase = 1; self._set_phase(1)
            print(f"\n  Curriculum → Phase 1 (OU drift)  [{t:,} steps]\n")
        elif self._phase == 1 and t >= self.phase2:
            self._phase = 2; self._set_phase(2)
            print(f"\n  Curriculum → Phase 2 (step changes)  [{t:,} steps]\n")
        return True


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, env, n=200):
    records = []
    for ep in range(n):
        obs, _ = env.reset()
        done = False
        rewards = []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(a)
            rewards.append(r)
            done = terminated or truncated
        records.append({**info, "ep_reward": float(np.sum(rewards))})

    df = pd.DataFrame(records)
    print("\n" + "=" * 55)
    print("EVALUATION")
    print("=" * 55)
    for col in ["capture_rate", "E_specific_GJ", "ep_reward"]:
        v = df[col]
        print(f"  {col:<20} mean={v.mean():.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}")
    print(f"\n  ≥85% capture : {(df.capture_rate>=85).mean()*100:.1f}%")
    print(f"  ≥90% capture : {(df.capture_rate>=90).mean()*100:.1f}%")
    print(f"  Flood events  : {(df.flood_fraction>0.80).mean()*100:.1f}%")
    print("=" * 55)
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    RL_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    lam_range = (args.lam_min, args.lam_max)

    env_kwargs = dict(
        model_path      = args.model_path,
        scaler_path     = args.scaler_path,
        max_steps       = args.max_steps,
        lambda_range    = lam_range,
        lam_smooth      = args.lam_smooth,
        lam_integral    = args.lam_I,
        lam_energy_int  = args.lam_Ie,
        lam_recover     = args.lam_rec,
        lam_flood       = args.lam_flood,
        step_prob       = args.step_prob,
        actuator_lag    = True,
        obs_noise       = True,
        domain_rand     = True,
        continue_prob   = 0.30,
        curriculum_phase= 0,
    )

    train_env = VecNormalize(
        make_vec_env(lambda: Monitor(CCUEnv(**env_kwargs)), n_envs=args.n_envs),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0,
    )

    # Eval env must also be VecNormalize-wrapped so EvalCallback can sync stats
    eval_env = VecNormalize(
        make_vec_env(lambda: Monitor(CCUEnv(
            **{**env_kwargs,
               "step_prob": 0.0, "obs_noise": False,
               "domain_rand": False, "continue_prob": 0.0,
               "curriculum_phase": 2}
        )), n_envs=1),
        norm_obs=True, norm_reward=False,   # don't normalise reward for eval
        clip_obs=10.0,
    )

    callbacks = [
        CurriculumCallback(args.phase1, args.phase2),
        EvalCallback(eval_env,
                     best_model_save_path=str(RL_DIR / "best"),
                     log_path="logs",
                     eval_freq=max(args.eval_freq // args.n_envs, 1),
                     n_eval_episodes=50, deterministic=True, verbose=1),
        CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1),
                           save_path=str(RL_DIR / "checkpoints"),
                           name_prefix="ppo_ccu", verbose=1),
    ]

    model = RecurrentPPO(
        policy        = "MlpLstmPolicy",
        env           = train_env,
        learning_rate = args.lr,
        n_steps       = 2048,
        batch_size    = 512,
        device        = 'cuda' if torch.cuda.is_available() else 'cpu',
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.01,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = dict(
            lstm_hidden_size = args.lstm_hidden,
            n_lstm_layers    = 2,
            net_arch         = dict(pi=[256, 256], vf=[256, 256]),
        ),
        tensorboard_log = "logs/",
        verbose         = 1,
        seed            = 42,
    )

    n_p = sum(p.numel() for p in model.policy.parameters())
    print(f"\n  PPO-LSTM  params={n_p:,}  envs={args.n_envs}"
          f"  timesteps={args.timesteps:,}"
          f"  action_dim=4  obs_dim=17\n")

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=True)

    model.save(str(RL_DIR / "ppo_ccu"))
    train_env.save(str(RL_DIR / "vecnorm.pkl"))
    print(f"\n  Saved: {RL_DIR}/ppo_ccu.zip  +  vecnorm.pkl")

    df = evaluate(model, eval_env, n=300)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/eval_results.csv", index=False)
    print("  Saved: results/eval_results.csv")


# ── Eval only ─────────────────────────────────────────────────────────────────

def eval_only(args):
    env = Monitor(CCUEnv(
        model_path=args.model_path, scaler_path=args.scaler_path,
        max_steps=args.max_steps,
        lambda_range=(args.lam_min, args.lam_max),
        step_prob=0.0, obs_noise=False, domain_rand=False,
        continue_prob=0.0, curriculum_phase=2,
    ))
    model = RecurrentPPO.load(args.model, env=env)
    print(f"Loaded: {args.model}")
    df = evaluate(model, env, n=500)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/eval_results.csv", index=False)
    print("Saved: results/eval_results.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",  default="models/surrogate/model.pt")
    p.add_argument("--scaler-path", default="models/surrogate/scalers.pkl")
    p.add_argument("--max-steps",   type=int,   default=120)
    p.add_argument("--lam-min",     type=float, default=0.0)
    p.add_argument("--lam-max",     type=float, default=0.05)
    p.add_argument("--lam-smooth",  type=float, default=0.005)
    p.add_argument("--lam-I",       type=float, default=0.10)
    p.add_argument("--lam-Ie",      type=float, default=0.05)
    p.add_argument("--lam-rec",     type=float, default=0.20)
    p.add_argument("--lam-flood",   type=float, default=0.25)
    p.add_argument("--step-prob",   type=float, default=0.04)
    p.add_argument("--phase1",      type=int,   default=200_000)
    p.add_argument("--phase2",      type=int,   default=600_000)
    p.add_argument("--timesteps",   type=int,   default=2_000_000)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--n-envs",      type=int,   default=16)
    p.add_argument("--lstm-hidden", type=int,   default=512)
    p.add_argument("--eval-freq",   type=int,   default=25_000)
    p.add_argument("--eval-only",   action="store_true")
    p.add_argument("--model",       default="models/rl/ppo_ccu_dense.zip")
    args = p.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()