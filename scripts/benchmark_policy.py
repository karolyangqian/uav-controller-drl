"""
Quantitative evaluation and performance metrics of trained PPO Quadrotor policy.
Measures RMSE position error, steady-state error, attitude stability, and action smoothness.
"""
import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv


def benchmark(model_path: str, n_episodes: int = 20, randomize_domain: bool = False):
    print("=" * 65)
    print(f"Benchmarking Policy: {model_path}")
    print(f"Episodes:           {n_episodes}")
    print(f"Domain Randomize:   {randomize_domain}")
    print("=" * 65)

    env = QuadrotorEnv(randomize_domain=randomize_domain)
    model = PPO.load(model_path)

    all_pos_errors = []
    steady_state_errors = []
    tilt_angles_deg = []
    action_diffs = []
    episode_rewards = []
    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_pos_errors = []
        ep_tilt = []
        ep_diffs = []
        ep_reward = 0.0

        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)

            ep_pos_errors.append(step_info["pos_error"])
            # Tilt angle from vertical: arccos(z_up) in degrees
            z_up = np.clip(step_info["z_up"], -1.0, 1.0)
            tilt_deg = np.degrees(np.arccos(z_up))
            ep_tilt.append(tilt_deg)
            ep_diffs.append(step_info["action_diff"])
            ep_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        all_pos_errors.extend(ep_pos_errors)
        # Steady-state error: last 100 steps (2 seconds)
        steady_state = np.mean(ep_pos_errors[-100:])
        steady_state_errors.append(steady_state)
        tilt_angles_deg.extend(ep_tilt)
        action_diffs.extend(ep_diffs)

        # Success criterion: steady-state error < 0.15m (15 cm)
        if steady_state < 0.15:
            successes += 1

    env.close()

    mean_rmse = np.sqrt(np.mean(np.array(all_pos_errors) ** 2))
    mean_ss_cm = np.mean(steady_state_errors) * 100.0
    std_ss_cm = np.std(steady_state_errors) * 100.0
    mean_tilt = np.mean(tilt_angles_deg)
    mean_act_diff = np.mean(action_diffs)
    success_rate = (successes / n_episodes) * 100.0

    print("\n--- Quantitative Benchmark Results ---")
    print(f"Overall Position RMSE:     {mean_rmse:.3f} m")
    print(f"Steady-State Hover Error:  {mean_ss_cm:.2f} +/- {std_ss_cm:.2f} cm")
    print(f"Mean Tilt Angle:           {mean_tilt:.2f} degrees")
    print(f"Action Smoothness (d_a):   {mean_act_diff:.4f} (lower is smoother)")
    print(f"Success Rate (<15cm err):  {success_rate:.1f}%")
    print(f"Mean Episode Reward:       {np.mean(episode_rewards):.1f} / {env.max_episode_steps * 2:.0f}")
    print("=" * 65)

    return {
        "rmse_m": mean_rmse,
        "ss_error_cm": mean_ss_cm,
        "ss_std_cm": std_ss_cm,
        "mean_tilt_deg": mean_tilt,
        "action_diff": mean_act_diff,
        "success_rate": success_rate,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark trained Quadrotor policy")
    parser.add_argument("--model-path", type=str, default="./checkpoints/best_model/best_model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--randomize-domain", action="store_true", help="Enable wind and mass disturbance")
    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        model_path = "./checkpoints/ppo_quadrotor_final.zip"

    benchmark(model_path, n_episodes=args.episodes, randomize_domain=args.randomize_domain)

