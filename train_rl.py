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
    python train_rl.py --resume models/rl/ppo_ccu.zip --timesteps 1000000
    python train_rl.py --eval-only --model models/rl/best/best_model.zip
"""

import argparse
import logging
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
try:
    from stable_baselines3.common.utils import sync_envs_normalization
except ImportError:
    from copy import deepcopy
    def sync_envs_normalization(train_env, eval_env) -> None:
        """Fallback for SB3 >= 2.0 where the helper was removed."""
        eval_env.obs_rms = deepcopy(train_env.obs_rms)
        eval_env.ret_rms = deepcopy(train_env.ret_rms)
from sb3_contrib import RecurrentPPO

from src.env import CCUEnv

log = logging.getLogger(__name__)

RL_DIR = Path("models/rl")


def linear_schedule(initial_value: float):
    """Linear decay from initial_value → 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# Named function (not lambda) required for SubprocVecEnv pickling on Windows.
def _make_env(**kwargs):
    return Monitor(CCUEnv(**kwargs))


class CurriculumCallback(BaseCallback):

    def __init__(self, phase1: int, phase2: int) -> None:
        super().__init__(verbose=0)
        self.phase1, self.phase2 = phase1, phase2
        self._phase = 0

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if self._phase == 0 and t >= self.phase1:
            self._phase = 1
            self.training_env.env_method("set_phase", 1)
            log.info("Curriculum → Phase 1 (OU drift)  [%d steps]", t)
        elif self._phase == 1 and t >= self.phase2:
            self._phase = 2
            self.training_env.env_method("set_phase", 2)
            log.info("Curriculum → Phase 2 (step changes)  [%d steps]", t)
        return True

class DomainMetricsCallback(BaseCallback):
    """Logs capture_rate, E_specific_GJ, flood_fraction to TensorBoard per rollout."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._ep_infos: list[dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            self._ep_infos.append(info)   # collect every step, not just terminal
        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_infos:
            return
        for key in ("capture_rate", "E_specific_GJ", "flood_fraction"):
            vals = [i[key] for i in self._ep_infos if key in i]
            if vals:
                self.logger.record(f"domain/{key}", float(np.mean(vals)))
        self._ep_infos.clear()


class SyncEvalCallback(EvalCallback):
    """EvalCallback that syncs VecNormalize obs stats from training env before each eval pass."""

    def _on_step(self) -> bool:
        if (isinstance(self.training_env, VecNormalize) and
                isinstance(self.eval_env, VecNormalize)):
            sync_envs_normalization(self.training_env, self.eval_env)
        return super()._on_step()


def evaluate(model: RecurrentPPO, env: VecNormalize, n_episodes: int = 200) -> pd.DataFrame:
    """Run n_episodes with proper LSTM state tracking. Returns per-episode metrics DataFrame."""
    records: list[dict] = []

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    n_envs = env.num_envs
    lstm_states = None
    episode_starts = np.ones(n_envs, dtype=bool)

    # Per-env accumulators for within-episode averages
    ep_caps: list[list[float]] = [[] for _ in range(n_envs)]
    ep_engs: list[list[float]] = [[] for _ in range(n_envs)]

    while len(records) < n_episodes:
        actions, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts,
            deterministic=True,
        )
        obs, rewards, dones, infos = env.step(actions)
        episode_starts = dones.copy()

        for i in range(n_envs):
            info = infos[i]
            if "capture_rate" in info:
                ep_caps[i].append(float(info["capture_rate"]))
            if "E_specific_GJ" in info:
                ep_engs[i].append(float(info["E_specific_GJ"]))

            if dones[i] and len(records) < n_episodes:
                rec: dict = {}
                # Episode-averaged metrics (more meaningful than terminal-step snapshot)
                if ep_caps[i]:
                    rec["capture_rate"] = float(np.mean(ep_caps[i]))
                if ep_engs[i]:
                    rec["E_specific_GJ"] = float(np.mean(ep_engs[i]))
                # Copy remaining scalar fields from terminal info
                for k, v in info.items():
                    if k not in ("capture_rate", "E_specific_GJ") and \
                            isinstance(v, (int, float, np.integer, np.floating)):
                        rec[k] = float(v)
                if "episode" in info:
                    rec["ep_reward"] = float(info["episode"]["r"])
                records.append(rec)
                ep_caps[i].clear()
                ep_engs[i].clear()

    df = pd.DataFrame(records[:n_episodes])
    log.info("=" * 55)
    log.info("EVALUATION  (%d episodes)", len(df))
    log.info("=" * 55)
    for col in ["capture_rate", "E_specific_GJ", "ep_reward"]:
        if col in df.columns:
            v = df[col]
            log.info("  %-20s mean=%.3f  min=%.3f  max=%.3f",
                     col, v.mean(), v.min(), v.max())
    if "capture_rate" in df.columns:
        log.info("  >=85%% capture : %.1f%%",
                 (df.capture_rate >= 85).mean() * 100)
        log.info("  >=90%% capture : %.1f%%",
                 (df.capture_rate >= 90).mean() * 100)
    if "flood_fraction" in df.columns:
        log.info("  Flood events  : %.1f%%",
                 (df.flood_fraction > 0.80).mean() * 100)
    log.info("=" * 55)
    return df


def train(args) -> None:
    RL_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available — training on CPU (slow). "
                    "Install PyTorch+CUDA: pip install torch --index-url "
                    "https://download.pytorch.org/whl/cu121")

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
        lam_over        = args.lam_over,
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

    buffer_size = args.n_envs * args.n_steps
    batch_size = min(args.batch_size, buffer_size)
    while buffer_size % batch_size != 0:
        batch_size -= 1

    callbacks = [
        CurriculumCallback(args.phase1, args.phase2),
        DomainMetricsCallback(),
        SyncEvalCallback(
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

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        log.info("Resuming from checkpoint: %s", resume_path)
        model = RecurrentPPO.load(str(resume_path), env=train_env,
                                  device=device, verbose=1,
                                  tensorboard_log="logs/")
        vecnorm_resume = RL_DIR / "vecnorm.pkl"
        if vecnorm_resume.exists():
            train_env = VecNormalize.load(str(vecnorm_resume), train_env.venv)
            model.set_env(train_env)
            log.info("Restored VecNormalize stats from %s", vecnorm_resume)
    else:
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
    log.info("RecurrentPPO  params=%d  device=%s  envs=%d  n_steps=%d  batch=%d",
             n_p, device, args.n_envs, args.n_steps, batch_size)
    log.info("LSTM: hidden=%d  layers=%d  timesteps=%d",
             args.lstm_hidden, args.lstm_layers, args.timesteps)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=True)
    elapsed = time.time() - t0
    log.info("Training complete in %.1f min", elapsed / 60)

    model.save(str(RL_DIR / "ppo_ccu"))
    train_env.save(str(RL_DIR / "vecnorm.pkl"))
    log.info("Saved: %s/ppo_ccu.zip  +  vecnorm.pkl", RL_DIR)

    df = evaluate(model, eval_env, n_episodes=300)
    df.to_csv("results/eval_results.csv", index=False)
    log.info("Saved: results/eval_results.csv")


def eval_only(args) -> None:
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
        log.info("Loaded VecNormalize stats: %s", vecnorm_path)
    else:
        log.warning("%s not found — using unnormalized env. "
                    "Evaluation metrics may be unreliable.", vecnorm_path)
        env = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = RecurrentPPO.load(args.model, env=env)
    log.info("Loaded: %s", args.model)
    df = evaluate(model, env, n_episodes=500)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/eval_results.csv", index=False)
    log.info("Saved: results/eval_results.csv")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Train RecurrentPPO on CCU env")
    p.add_argument("--model-path",  default="models/surrogate/model.pt")
    p.add_argument("--scaler-path", default="models/surrogate/scalers.pkl")
    p.add_argument("--max-steps",   type=int,   default=120)
    p.add_argument("--lam-min",     type=float, default=0.0)
    p.add_argument("--lam-max",     type=float, default=0.05,
                   help="Max energy weight — low keeps capture dominant over energy")
    p.add_argument("--lam-smooth",  type=float, default=0.015,
                   help="Smoothness penalty (lowered to allow faster recovery actions)")
    p.add_argument("--lam-I",       type=float, default=0.40,
                   help="Capture deficit integral weight — higher drives faster recovery below 90%%")
    p.add_argument("--lam-Ie",      type=float, default=0.02,
                   help="Energy integral weight (reduced 4x: don't fear high energy when capture needs it)")
    p.add_argument("--lam-above",   type=float, default=0.40,
                   help="Above-target bonus weight — steeper gradient pulling agent into 85-95%% zone")
    p.add_argument("--lam-over",    type=float, default=0.25,
                   help="Over-capture penalty weight (quadratic): discourages capture above 95%%")
    p.add_argument("--lam-flood",   type=float, default=0.10,
                   help="Flood soft penalty (reduced: hard constraint does the safety work)")
    p.add_argument("--step-prob",   type=float, default=0.04)
    p.add_argument("--phase1",      type=int,   default=300_000,
                   help="Phase 1 start (longer Phase 0 for capture fundamentals)")
    p.add_argument("--phase2",      type=int,   default=700_000)
    p.add_argument("--timesteps",   type=int,   default=2_000_000)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--n-envs",      type=int,   default=16)
    p.add_argument("--n-steps",     type=int,   default=512,
                   help="Steps per env per rollout (~4 full episodes; better LSTM gradient estimates)")
    p.add_argument("--batch-size",  type=int,   default=512,
                   help="Mini-batch size (auto-adjusted to divide buffer)")
    p.add_argument("--n-epochs",    type=int,   default=10)
    p.add_argument("--lstm-hidden", type=int,   default=256)
    p.add_argument("--lstm-layers", type=int,   default=1)
    p.add_argument("--subproc",     action="store_true",
                   help="Use SubprocVecEnv instead of DummyVecEnv")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint .zip to resume training from")
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
