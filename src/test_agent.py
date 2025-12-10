# src/test_agent.py

import time
import numpy as np
import torch
from stable_baselines3 import PPO
from polytrack_env import PolytrackEnv
from reward_shaping import compute_shaped_reward

def main():
    print("=== Polytrack AI Test Script ===")
    print("Loading trained PPO model 'ppo_mixed.zip'...")

    try:
        model = PPO.load("models/ppo_mixed.zip")
    except Exception as e:
        print("ERROR: Could not load ppo_mixed.zip")
        print(e)
        return

    print("[OK] Model loaded.")
    print("Starting environment...")

    env = PolytrackEnv()
    obs, info = env.reset()

    print("[OK] Env ready.")
    print("Agent will now drive. Press CTRL+C to stop.")

    total_steps = 0
    episode = 1
    ep_start_time = time.time()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        total_steps += 1

        prog = info.get("progress", 0)
        t = info.get("time", 0)

        if total_steps % 15 == 0:
            print(f"[EP {episode}] Step={total_steps} | Action={action} | Progress={prog*100:.2f}% | Time={t:.2f}s")

        if done or truncated:
            lap_time = info.get("time", -1)
            print("")
            print("=== LAP COMPLETE ===")
            print(f"Episode: {episode}")
            print(f"Lap Time: {lap_time:.2f} seconds")
            print(f"Final progress: {prog*100:.1f}%")
            print("====================")
            print("Resetting...\n")

            episode += 1
            obs, info = env.reset()
            total_steps = 0
            ep_start_time = time.time()

if __name__ == "__main__":
    main()
