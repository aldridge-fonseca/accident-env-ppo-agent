import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from custom_env import AccidentEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "ppo_accident_env_final.zip"
NUM_EPISODES = int(os.getenv("EVAL_EPISODES", "500"))

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def make_eval_env():
    env = AccidentEnv(config={
        "observation": {"type": "LidarObservation"},
        "duration": 20,
    })
    return Monitor(env)


model = PPO.load(str(MODEL_PATH))
eval_env = make_eval_env()
episode_rewards = []

print(f"Running AccidentEnv performance test for {NUM_EPISODES} episodes...")
for episode in range(NUM_EPISODES):
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    episode_rewards.append(total_reward)

    if (episode + 1) % 50 == 0 or episode + 1 == NUM_EPISODES:
        print(f"Completed {episode + 1}/{NUM_EPISODES} episodes | Mean: {np.mean(episode_rewards):.3f}")

eval_env.close()

mean_reward = float(np.mean(episode_rewards))
std_reward = float(np.std(episode_rewards))
median_reward = float(np.median(episode_rewards))
min_reward = float(np.min(episode_rewards))
max_reward = float(np.max(episode_rewards))

plt.figure(figsize=(8, 6))
parts = plt.violinplot([episode_rewards], positions=[1], showmeans=True, showmedians=True)
for body in parts["bodies"]:
    body.set_facecolor("steelblue")
    body.set_alpha(0.7)
plt.text(
    1.3,
    mean_reward,
    f"Mean: {mean_reward:.3f}\nStd: {std_reward:.3f}\nMedian: {median_reward:.3f}\nMin: {min_reward:.3f}\nMax: {max_reward:.3f}",
    fontsize=10,
    verticalalignment="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
plt.ylabel("Episodic Reward (Return)")
plt.title(f"Performance Test - AccidentEnv\nLidarObservation | {NUM_EPISODES} Episodes")
plt.xticks([1], ["PPO Agent"])
plt.grid(True, alpha=0.3, axis="y")
plot_path = PLOTS_DIR / "evaluation_performance_test.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

summary_path = PLOTS_DIR / "performance_summary.md"
summary_path.write_text(
    "# AccidentEnv PPO performance summary\n\n"
    f"- Episodes: {NUM_EPISODES}\n"
    f"- Mean reward: {mean_reward:.4f}\n"
    f"- Standard deviation: {std_reward:.4f}\n"
    f"- Median reward: {median_reward:.4f}\n"
    f"- Minimum reward: {min_reward:.4f}\n"
    f"- Maximum reward: {max_reward:.4f}\n",
    encoding="utf-8",
)
np.save(RESULTS_DIR / "ID-16_episode_rewards.npy", np.array(episode_rewards))

print(f"Performance plot saved to {plot_path}")
print(f"Summary saved to {summary_path}")
print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
