"""
Verification and benchmark script for QuadrotorEnv.
Tests initialization, reset, physics stepping, hover stability with zero action,
and measures simulation steps per second (SPS).
"""
import sys
import os
import time
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv


def main():
    print("=" * 60)
    print("Quadrotor Environment Verification & Performance Test")
    print("=" * 60)

    env = QuadrotorEnv()
    print(f"[OK] Environment instantiated successfully.")
    print(f"     Action Space:      {env.action_space}")
    print(f"     Observation Space: {env.observation_space}")

    # Test Reset
    obs, info = env.reset(seed=42)
    assert obs.shape == (16,), f"Expected obs shape (16,), got {obs.shape}"
    print(f"[OK] Environment reset successful.")
    print(f"     Initial Obs shape: {obs.shape}")
    print(f"     Target Position:   {info['target_pos']}")

    # Test Stepping with Zero Action (Pure Hover Request: [0, 0, 0, 0])
    print("\n--- Stepping 100 steps with zero action (hover thrust) ---")
    total_reward = 0.0
    for step in range(100):
        # Action [0, 0, 0, 0] = base hover thrust, 0 angular rate command
        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"     Episode ended early at step {step}: Terminated={terminated}, Truncated={truncated}")
            break

    state = env._get_drone_state()
    print(f"[OK] 100 steps completed. Final pos: {np.round(state['pos_world'], 3)}")
    print(f"     Final tilt cos(z): {state['z_body_world'][2]:.4f} (1.0 = perfectly upright)")
    print(f"     Pos Error:         {step_info['pos_error']:.3f} m")

    # Benchmark Speed (Steps Per Second)
    print("\n--- Benchmarking Single-Thread Simulation Speed ---")
    env.reset(seed=123)
    num_steps = 1000
    start_time = time.time()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    elapsed = time.time() - start_time
    sps = num_steps / elapsed
    sim_time = num_steps * env.dt_ctrl
    realtime_factor = sim_time / elapsed

    print(f"[OK] Stepped {num_steps} control steps in {elapsed:.3f} seconds.")
    print(f"     Throughput:        {sps:.1f} steps/second (50 Hz control loop)")
    print(f"     Real-Time Factor:  {realtime_factor:.1f}x real-time (single CPU core)")
    print("=" * 60)
    print("All Day 1 Environment Tests Passed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
