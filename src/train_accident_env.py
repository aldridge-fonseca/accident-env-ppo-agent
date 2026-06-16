import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from custom_env import AccidentEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = REPO_ROOT / "models"
LOG_DIR = REPO_ROOT / "logs"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
MODEL_PATH = MODELS_DIR / "ppo_accident_env_final.zip"
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "500000"))
N_ENVS = int(os.getenv("N_ENVS", "16"))
CHECKPOINT_FREQ = int(os.getenv("CHECKPOINT_FREQ", "50000"))
EVAL_FREQ = int(os.getenv("EVAL_FREQ", "10000"))
N_EVAL_EPISODES = int(os.getenv("N_EVAL_EPISODES", "10"))
LOG_EPISODE_FREQ = int(os.getenv("LOG_EPISODE_FREQ", "1000"))

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


def make_env(rank: int, seed: int = 0):
    def _init():
        env = AccidentEnv(config={
            "observation": {"type": "LidarObservation"},
            "duration": 20,
        })
        env = Monitor(env, str(LOG_DIR / f"env_{rank}"))
        env.reset(seed=seed + rank)
        return env

    return _init


def make_eval_env():
    env = AccidentEnv(config={
        "observation": {"type": "LidarObservation"},
        "duration": 20,
    })
    return Monitor(env, str(LOG_DIR / "eval"))


class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.timesteps_log = []
        self.current_rewards = None
        self.log_freq = log_freq
        self.last_log_episode = 0

    def _on_training_start(self):
        self.current_rewards = np.zeros(self.training_env.num_envs)
        print(f"Training started with {self.training_env.num_envs} parallel environments")

    def _on_step(self):
        self.current_rewards += self.locals["rewards"]
        for index, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_rewards.append(self.current_rewards[index])
                self.timesteps_log.append(self.num_timesteps)
                self.current_rewards[index] = 0

        total_episodes = len(self.episode_rewards)
        if total_episodes > 0 and total_episodes >= self.last_log_episode + self.log_freq:
            recent_rewards = self.episode_rewards[-self.log_freq:]
            print(
                f"Episode {total_episodes}: "
                f"mean={np.mean(recent_rewards):.3f}, "
                f"min={np.min(recent_rewards):.3f}, "
                f"max={np.max(recent_rewards):.3f}"
            )
            self.last_log_episode = total_episodes
        return True


def plot_learning_curve(rewards, window: int = 100):
    plt.figure(figsize=(10, 6))
    episodes = np.arange(len(rewards))
    plt.plot(episodes, rewards, alpha=0.3, color="blue", label="Episode Reward")
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window - 1, len(rewards)), moving_avg, color="red", linewidth=2, label=f"Moving Avg ({window} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward (Return)")
    plt.title("ID-15: Learning Curve - AccidentEnv\nLidarObservation")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plot_path = PLOTS_DIR / "ID-15_training_learning_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def train():
    print(f"Training PPO on AccidentEnv for {TOTAL_TIMESTEPS:,} timesteps")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([make_eval_env])

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=torch.nn.ReLU,
    )

    def lr_schedule(progress_remaining):
        return 3e-4 * progress_remaining + 1e-5 * (1 - progress_remaining)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=lr_schedule,
        n_steps=4096,
        batch_size=256,
        n_epochs=15,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.15,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(LOG_DIR),
        verbose=1,
        device="auto",
    )

    reward_callback = RewardLoggingCallback(log_freq=LOG_EPISODE_FREQ)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_accident_env",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR),
        log_path=str(LOG_DIR),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[reward_callback, checkpoint_callback, eval_callback], progress_bar=True)
    model.save(str(MODEL_PATH))

    rewards = np.array(reward_callback.episode_rewards)
    timesteps = np.array(reward_callback.timesteps_log)
    np.save(RESULTS_DIR / "ID-15_training_rewards.npy", rewards)
    np.save(RESULTS_DIR / "ID-15_training_timesteps.npy", timesteps)
    plot_path = plot_learning_curve(rewards)

    train_env.close()
    eval_env.close()

    print(f"Model saved to {MODEL_PATH}")
    print(f"Learning curve saved to {plot_path}")
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    train()
