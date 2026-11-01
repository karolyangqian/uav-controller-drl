"""
Interactive evaluation and visualization of trained policy in MuJoCo viewer.
Visualizes 3D hover, step response, and disturbance rejection.
"""
import os
import sys
import time
import argparse
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Quadrotor Policy")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained PPO .zip model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--randomize-domain", action="store_true", help="Add wind and mass randomization")
    args = parser.parse_args()

    model = None
    if args.model_path and os.path.exists(args.model_path):
        from stable_baselines3 import PPO
        print(f"Loading model from: {args.model_path}")
        model = PPO.load(args.model_path)
    else:
        print("No model path provided or file not found. Running with baseline hover action ([0, 0, 0, 0]).")

    env = QuadrotorEnv(
        render_mode="human",
        randomize_domain=args.randomize_domain,
    )

    print("\nStarting evaluation... Press Ctrl+C in terminal or close the window to exit.")

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0
        step = 0
        print(f"\n--- Episode {ep} --- Target: {np.round(info['target_pos'], 2)}")

        terminated, truncated = False, False
        while not (terminated or truncated):
            start_loop = time.time()

            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Default baseline: zero action (hover thrust + zero angular rate)
                action = np.zeros(4, dtype=np.float32)

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            step += 1

            # Sleep to match real-time 50 Hz control rate
            render_elapsed = time.time() - start_loop
            sleep_time = max(0.0, env.dt_ctrl - render_elapsed)
            time.sleep(sleep_time)

            if step % 50 == 0:
                print(
                    f"Step {step:3d} | Pos Err: {step_info['pos_error']:.3f} m | "
                    f"Tilt cos(z): {step_info['z_up']:.3f} | Action Diff: {step_info['action_diff']:.3f}"
                )

        print(
            f"Episode {ep} ended at step {step}: "
            f"Total Reward: {ep_reward:.1f}, Final Pos Err: {step_info['pos_error']:.3f} m"
        )

    env.close()


if __name__ == "__main__":
    main()
