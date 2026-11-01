"""
PPO Training script for Quadrotor 3D Hover & Regulation.
Optimized for multi-core CPU parallelization on Intel Core Ultra 7 (Meteor Lake).
Uses ReLU activations for straightforward FPGA quantization and hardware mapping.
"""
import os
import sys
import argparse
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv


def make_env_fn(randomize_domain: bool = False):
    def _init():
        return QuadrotorEnv(randomize_domain=randomize_domain)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO controller for Quadrotor")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments (CPU threads)")
    parser.add_argument("--randomize-domain", action="store_true", help="Enable wind and mass randomization")
    parser.add_argument("--log-dir", type=str, default="./tb_logs/ppo_quadrotor", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Model checkpoint directory")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("Launching Quadrotor PPO Training")
    print(f"Parallel Envs:       {args.n_envs} (SubprocVecEnv)")
    print(f"Total Timesteps:     {args.total_timesteps:,}")
    print(f"Domain Randomize:    {args.randomize_domain}")
    print(f"TensorBoard Logs:    {args.log_dir}")
    print(f"Checkpoints:         {args.save_dir}")
    print("=" * 60)

    # Create vectorized parallel environments with monitor logging
    monitor_train_dir = os.path.join(args.save_dir, "monitor_train")
    os.makedirs(monitor_train_dir, exist_ok=True)
    vec_env = make_vec_env(
        make_env_fn(randomize_domain=args.randomize_domain),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_train_dir,
    )

    # Evaluation environment (unrandomized baseline)
    eval_env = make_vec_env(
        make_env_fn(randomize_domain=False),
        n_envs=2,
        vec_env_cls=SubprocVecEnv,
    )

    # Evaluation callback (eval every 25,000 steps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, "best_model"),
        log_path=os.path.join(args.save_dir, "eval_logs"),
        eval_freq=max(25_000 // args.n_envs, 2500),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 10000),
        save_path=os.path.join(args.save_dir, "checkpoints"),
        name_prefix="ppo_quad",
    )

    # Policy network: 2 hidden layers of 64 units with ReLU
    # ReLU ensures simple, efficient quantization and synthesis in FPGA PL (Day 6)
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.log_dir,
        verbose=1,
        device="cpu",  # Optimal for small MLP + vectorized MuJoCo on Intel Ultra 7
    )

    # Train model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(args.save_dir, "ppo_quadrotor_final.zip")
    model.save(final_path)
    print(f"\n[OK] Training completed! Model saved to {final_path}")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
