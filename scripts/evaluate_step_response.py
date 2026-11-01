"""
Step Response Analysis and Telemetry Plotting for Trained Quadrotor Policy.
Measures rise time, settling time, overshoot, and steady-state error under a 3D step input.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO


# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv


def evaluate_step_response(model_path: str, save_plot: str = "./figures/step_response.png"):
    os.makedirs(os.path.dirname(save_plot), exist_ok=True)
    env = QuadrotorEnv(randomize_domain=False)
    model = PPO.load(model_path)

    # Simulation parameters: 8 seconds at 50 Hz = 400 steps
    total_steps = 400
    times = np.linspace(0, total_steps * env.dt_ctrl, total_steps)

    # Step profile:
    # 0 to 2s: Setpoint [0.0, 0.0, 1.0]
    # 2 to 8s: Step to [1.0, 1.0, 2.0]
    setpoint_initial = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    setpoint_step = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    step_time = 2.0
    step_idx = int(step_time / env.dt_ctrl)

    obs, _ = env.reset()
    # Force initial position to [0.0, 0.0, 1.0]
    env.data.qpos[0:3] = setpoint_initial.copy()
    env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env.data.qvel[:] = 0.0
    env.target_pos = setpoint_initial.copy()
    env.data.mocap_pos[env.target_mocap_id] = env.target_pos

    # Storage arrays
    pos_history = []
    target_history = []
    tilt_history = []
    actions_history = []

    for step in range(total_steps):
        t = times[step]
        if step >= step_idx:
            env.target_pos = setpoint_step.copy()
            env.data.mocap_pos[env.target_mocap_id] = env.target_pos

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)

        state = env._get_drone_state()
        pos_history.append(state["pos_world"].copy())
        target_history.append(env.target_pos.copy())
        actions_history.append(action.copy())

        # Tilt angle in degrees
        z_up = np.clip(state["z_body_world"][2], -1.0, 1.0)
        tilt_deg = np.degrees(np.arccos(z_up))
        tilt_history.append(tilt_deg)

    pos_history = np.array(pos_history)
    target_history = np.array(target_history)
    tilt_history = np.array(tilt_history)
    actions_history = np.array(actions_history)

    # Save step response telemetry to CSV
    os.makedirs("./telemetry", exist_ok=True)
    step_df = pd.DataFrame({
        "time_sec": times,
        "x_actual_m": pos_history[:, 0],
        "y_actual_m": pos_history[:, 1],
        "z_actual_m": pos_history[:, 2],
        "x_target_m": target_history[:, 0],
        "y_target_m": target_history[:, 1],
        "z_target_m": target_history[:, 2],
        "tilt_deg": tilt_history,
        "action_thrust": actions_history[:, 0],
        "action_wx": actions_history[:, 1],
        "action_wy": actions_history[:, 2],
        "action_wz": actions_history[:, 3],
    })
    step_df.to_csv("./telemetry/step_response_telemetry.csv", index=False)
    print("[OK] Step response telemetry saved to ./telemetry/step_response_telemetry.csv")


    # Compute step response metrics after step_time (2.0s to 8.0s)
    post_step_times = times[step_idx:] - step_time
    post_step_pos = pos_history[step_idx:]
    dist_to_target = np.linalg.norm(post_step_pos - setpoint_step, axis=1)

    initial_step_dist = np.linalg.norm(setpoint_step - setpoint_initial)  # ~1.732 m
    # Rise time (time to reach 90% of distance)
    # distance to target < 0.1 * initial_step_dist
    reached_90_idx = np.where(dist_to_target <= 0.10 * initial_step_dist)[0]
    rise_time = post_step_times[reached_90_idx[0]] if len(reached_90_idx) > 0 else np.nan

    # Settling time (time after which error remains <= 0.05m / 5cm)
    within_band = dist_to_target <= 0.05
    # Find last index where it was outside band
    outside_band = np.where(~within_band)[0]
    settling_time = post_step_times[outside_band[-1] + 1] if len(outside_band) > 0 and (outside_band[-1] + 1 < len(post_step_times)) else (post_step_times[0] if len(outside_band) == 0 else np.nan)

    # Steady-state error (last 100 steps / 2 seconds)
    ss_error = np.mean(dist_to_target[-100:])

    print("=" * 65)
    print("3D Step Response Evaluation Results")
    print(f"Step: [0, 0, 1] -> [1, 1, 2] (3D Distance: {initial_step_dist:.3f} m)")
    print(f"Rise Time (Tr to 90%):     {rise_time:.2f} s")
    print(f"Settling Time (Ts +/-5cm): {settling_time:.2f} s")
    print(f"Steady-State Hover Error:  {ss_error * 100:.2f} cm")
    print(f"Max Tilt during Transition: {np.max(tilt_history[step_idx:]):.1f} degrees")
    print("=" * 65)

    # Plot 3-Panel Step Response
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Panel 1: Position tracking
    ax1.plot(times, target_history[:, 0], "r--", alpha=0.6, label="X Target")
    ax1.plot(times, pos_history[:, 0], "r-", linewidth=2.0, label="X Actual")
    ax1.plot(times, target_history[:, 1], "g--", alpha=0.6, label="Y Target")
    ax1.plot(times, pos_history[:, 1], "g-", linewidth=2.0, label="Y Actual")
    ax1.plot(times, target_history[:, 2], "b--", alpha=0.6, label="Z Target")
    ax1.plot(times, pos_history[:, 2], "b-", linewidth=2.0, label="Z Actual")
    ax1.axvline(x=step_time, color="gray", linestyle=":", label="Step Command (t=2s)")
    ax1.set_ylabel("Position (m)", fontsize=11)
    ax1.set_title("Quadrotor 3D Step Response & Tracking Trajectory", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", ncol=4, fontsize=9)

    # Panel 2: Attitude tilt angle
    ax2.plot(times, tilt_history, color="purple", linewidth=2.0, label="Tilt Angle")
    ax2.axvline(x=step_time, color="gray", linestyle=":")
    ax2.set_ylabel("Tilt (deg)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=10)

    # Panel 3: Control Actions
    ax3.plot(times, actions_history[:, 0], color="black", label="Thrust Action a_T")
    ax3.plot(times, actions_history[:, 1], color="crimson", label="Pitch Rate a_wx")
    ax3.plot(times, actions_history[:, 2], color="forestgreen", label="Roll Rate a_wy")
    ax3.plot(times, actions_history[:, 3], color="dodgerblue", label="Yaw Rate a_wz")
    ax3.axvline(x=step_time, color="gray", linestyle=":")
    ax3.set_xlabel("Time (seconds)", fontsize=11)
    ax3.set_ylabel("Action [-1, 1]", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right", ncol=4, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_plot, dpi=200)
    plt.close()
    print(f"[OK] Step response figure successfully saved to {save_plot}")

    return {
        "rise_time": rise_time,
        "settling_time": settling_time,
        "ss_error_cm": ss_error * 100.0,
        "max_tilt": float(np.max(tilt_history[step_idx:])),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/robust/best_model/best_model.zip")
    parser.add_argument("--save-plot", type=str, default="./figures/step_response.png")
    args = parser.parse_args()

    evaluate_step_response(args.model_path, args.save_plot)
