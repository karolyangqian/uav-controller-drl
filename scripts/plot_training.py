"""
Plot training progress (Reward and Episode Length vs Timesteps) from Monitor CSV logs.
"""
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_monitor_data(monitor_dir: str):
    csv_files = glob.glob(os.path.join(monitor_dir, "*.monitor.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No monitor CSV files found in {monitor_dir}")

    dfs = []
    for f in csv_files:
        # Skip the first comment line (#{"t_start": ...})
        df = pd.read_csv(f, skiprows=1)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    # Sort by wall-clock time 't'
    data = data.sort_values(by="t").reset_index(drop=True)
    # Cumulative timesteps: sum of episode lengths 'l'
    data["timesteps"] = data["l"].cumsum()
    return data


def smooth(y, window=50):
    if len(y) < window:
        return y
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def main():
    parser = argparse.ArgumentParser(description="Plot PPO Training Progress")
    parser.add_argument("--monitor-dir", type=str, default="./checkpoints/monitor_train")
    parser.add_argument("--save-path", type=str, default="./figures/training_curve.png")
    parser.add_argument("--window", type=int, default=50, help="Smoothing window")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    data = load_monitor_data(args.monitor_dir)

    timesteps = data["timesteps"].values
    rewards = data["r"].values
    lengths = data["l"].values

    smoothed_rewards = smooth(rewards, window=args.window)
    smoothed_lengths = smooth(lengths, window=args.window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Reward Plot
    ax1.plot(timesteps, rewards, alpha=0.25, color="dodgerblue", label="Raw Episode Reward")
    ax1.plot(timesteps, smoothed_rewards, color="navy", linewidth=2.0, label=f"Smoothed (window={args.window})")
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.set_title("PPO Quadrotor 3D Hover & Regulation Training Curves", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Episode Length Plot
    ax2.plot(timesteps, lengths, alpha=0.25, color="forestgreen", label="Raw Episode Length")
    ax2.plot(timesteps, smoothed_lengths, color="darkgreen", linewidth=2.0, label=f"Smoothed (window={args.window})")
    ax2.set_xlabel("Total Timesteps", fontsize=12)
    ax2.set_ylabel("Episode Steps (max 500)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=200)
    plt.close()
    print(f"[OK] Training curve plot successfully saved to {args.save_path}")


if __name__ == "__main__":
    main()
