"""
Generate comparative benchmark visualization and 3D flight trajectory plots.
Saves figures/comparison_benchmark.png and figures/trajectory_3d.png.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv


def plot_comparison_benchmarks(save_path: str = "./figures/comparison_benchmark.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metrics = [
        "Steady-State Error (cm)",
        "Success Rate (<15cm, %)",
        "Mean Disturbed Reward",
        "Active Wind Lean (deg)",
    ]

    baseline_vals = [35.61, 5.0, 847.4, 7.83]
    robust_vals = [13.46, 85.0, 914.3, 10.29]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline Policy (Nominal Training)", color="lightcoral", edgecolor="firebrick")
    rects2 = ax.bar(x + width/2, robust_vals, width, label="Robust Policy (Domain Randomization)", color="mediumseagreen", edgecolor="darkgreen")

    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Performance Comparison Under Severe Disturbance (Wind + Mass Shifting)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] Comparison benchmark plot saved to {save_path}")


def plot_3d_flight_trajectory(model_path: str, save_path: str = "./figures/trajectory_3d.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = QuadrotorEnv(randomize_domain=True)
    model = PPO.load(model_path)

    obs, info = env.reset(seed=42)
    target_pos = info["target_pos"]

    trajectory = []
    total_steps = 300

    for step in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        state = env._get_drone_state()
        trajectory.append(state["pos_world"].copy())
        if term or trunc:
            break

    trajectory = np.array(trajectory)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot 3D trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            label="Quadrotor Flight Path", color="royalblue", linewidth=2.5)

    # Plot Start Point
    ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
               color="darkorange", s=120, label="Start Pose", marker="^")

    # Plot Target Point
    ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]],
               color="crimson", s=180, label=f"Target [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]", marker="*")

    # Wireframe sphere around target (5cm tolerance sphere)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    r = 0.15  # 15 cm radius
    xs = target_pos[0] + r * np.outer(np.cos(u), np.sin(v))
    ys = target_pos[1] + r * np.outer(np.sin(u), np.sin(v))
    zs = target_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="crimson", alpha=0.15)

    # Ground plane outline
    ax.plot([-2, 2, 2, -2, -2], [-2, -2, 2, 2, -2], [0, 0, 0, 0, 0], "k--", alpha=0.3)

    ax.set_xlabel("X (meters)", fontsize=11)
    ax.set_ylabel("Y (meters)", fontsize=11)
    ax.set_zlabel("Z (meters)", fontsize=11)
    ax.set_title("3D Autonomous Waypoint Trajectory in MuJoCo", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[OK] 3D flight trajectory plot saved to {save_path}")


if __name__ == "__main__":
    plot_comparison_benchmarks()
    robust_model = "./checkpoints/robust/best_model/best_model.zip"
    plot_3d_flight_trajectory(robust_model)
