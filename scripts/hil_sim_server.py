#!/usr/bin/env bash
# Hardware-in-the-Loop (HIL) Simulation Server for Quadrotor UAV
# Streams 16-D body-frame observations to AMD Kria KR260 over UDP
# Receives 4-D control actions [T_z, wx, wy, wz] and steps MuJoCo physics
""":"
python3 "$0" "$@"
exit $?
"""

import argparse
import socket
import struct
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import mujoco

from envs.quadrotor_env import QuadrotorEnv

STATE_MAGIC = 0x55415653  # "UAVS"
ACTION_MAGIC = 0x55415641  # "UAVA"

STATE_STRUCT = struct.Struct('<II16f')  # magic (4B), seq_id (4B), obs (64B) = 72 bytes
ACTION_STRUCT = struct.Struct('<II4f')  # magic (4B), seq_id (4B), action (16B) = 24 bytes


def parse_args():
    parser = argparse.ArgumentParser(description="HIL Simulation Server for AMD Kria KR260")
    parser.add_argument("--bind-ip", type=str, default="0.0.0.0", help="IP address to bind UDP server")
    parser.add_argument("--server-port", type=int, default=5006, help="Port to receive actions from Kria")
    parser.add_argument("--client-ip", type=str, default="127.0.0.1", help="IP address of Kria KR260 client")
    parser.add_argument("--client-port", type=int, default=5005, help="Port to send states to Kria")
    parser.add_argument("--rate", type=float, default=50.0, help="Outer-loop control frequency in Hz")
    parser.add_argument("--duration", type=float, default=20.0, help="Flight duration in seconds (0 for unlimited)")
    parser.add_argument("--target-pos", type=float, nargs=3, default=[1.0, 1.0, 1.5], help="Target 3D position [x, y, z]")
    parser.add_argument("--timeout-ms", type=float, default=15.0, help="Packet timeout before failsafe in ms")
    parser.add_argument("--randomize-domain", action="store_true", help="Enable wind disturbances and mass variations")
    parser.add_argument("--render", action="store_true", help="Launch interactive 3D visual viewer")
    return parser.parse_args()


def run_hil_server():
    args = parse_args()
    dt_ctrl = 1.0 / args.rate
    timeout_sec = args.timeout_ms / 1000.0

    print("=" * 65)
    print("  UAV DRL Hardware-in-the-Loop (HIL) Simulation Server")
    print("=" * 65)
    print(f"  Bind Address:        {args.bind_ip}:{args.server_port}")
    print(f"  Kria Client Target:  {args.client_ip}:{args.client_port}")
    print(f"  Control Frequency:   {args.rate:.1f} Hz (Period: {dt_ctrl*1000:.1f} ms)")
    print(f"  Packet Timeout:      {args.timeout_ms:.1f} ms (Auto-Hover Failsafe)")
    print(f"  Target Setpoint:     {args.target_pos}")
    print(f"  Disturbance Mode:    {'Active (Wind + Mass perturbations)' if args.randomize_domain else 'Nominal'}")
    print("=" * 65)

    # Initialize environment
    env = QuadrotorEnv(randomize_domain=args.randomize_domain)
    obs, _ = env.reset()
    if args.target_pos is not None:
        env.target_pos = np.array(args.target_pos, dtype=np.float32)
        env.data.mocap_pos[env.target_mocap_id] = env.target_pos
        obs = env._get_obs()

    # Setup UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.server_port))
    sock.settimeout(timeout_sec)

    # Passive viewer setup if requested
    viewer = None
    if args.render:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            print("[INFO] MuJoCo 3D Viewer launched.")
        except Exception as e:
            print(f"[WARN] Failed to open viewer: {e}. Running in console mode.")

    client_addr = (args.client_ip, args.client_port)

    # Performance metrics tracking
    seq_id = 0
    total_packets_sent = 0
    total_packets_recv = 0
    timeouts = 0
    rtt_history = []
    pos_errors = []
    tilts = []

    failsafe_hover_action = np.zeros(4, dtype=np.float32)
    last_valid_action = np.zeros(4, dtype=np.float32)

    total_steps = int(args.duration * args.rate) if args.duration > 0 else sys.maxsize
    step = 0
    start_time = time.perf_counter()

    print("\n[READY] Streaming telemetry. Press Ctrl+C to terminate.\n")

    try:
        while step < total_steps:
            loop_start = time.perf_counter()

            # Pack 16-D observation into binary StatePacket (72 bytes)
            state_data = STATE_STRUCT.pack(STATE_MAGIC, seq_id, *obs)
            sock.sendto(state_data, client_addr)
            total_packets_sent += 1
            t_send = time.perf_counter()

            # Wait for ActionPacket (24 bytes) from client
            applied_action = failsafe_hover_action
            received = False
            while True:
                try:
                    data, addr = sock.recvfrom(64)
                    if len(data) == ACTION_STRUCT.size:
                        magic, recv_seq, a0, a1, a2, a3 = ACTION_STRUCT.unpack(data)
                        if magic == ACTION_MAGIC and recv_seq == seq_id:
                            rtt_ms = (time.perf_counter() - t_send) * 1000.0
                            rtt_history.append(rtt_ms)
                            applied_action = np.array([a0, a1, a2, a3], dtype=np.float32)
                            last_valid_action = applied_action
                            total_packets_recv += 1
                            received = True
                            break
                except socket.timeout:
                    timeouts += 1
                    applied_action = last_valid_action * 0.5  # Soft decay towards hover
                    break
                except BlockingIOError:
                    break

            # Step simulation with received action
            obs, reward, terminated, truncated, info = env.step(applied_action)
            step += 1
            seq_id = (seq_id + 1) & 0xFFFFFFFF

            pos_err = info["pos_error"]
            tilt_deg = float(np.degrees(np.arccos(np.clip(info["z_up"], -1.0, 1.0))))
            pos_errors.append(pos_err)
            tilts.append(tilt_deg)

            if viewer is not None:
                viewer.sync()

            # Console telemetry status display every 10 steps (0.2s)
            if step % 10 == 0:
                recent_rtt = np.mean(rtt_history[-10:]) if rtt_history else 0.0
                drone_pos = env.data.xpos[env.body_id]
                loss_pct = (timeouts / total_packets_sent) * 100.0 if total_packets_sent else 0.0
                status_str = (
                    f"Step {step:4d} | "
                    f"Pos: [{drone_pos[0]:5.2f}, {drone_pos[1]:5.2f}, {drone_pos[2]:5.2f}] | "
                    f"Err: {pos_err*100:5.1f} cm | "
                    f"Tilt: {tilt_deg:4.1f}° | "
                    f"RTT: {recent_rtt:5.2f} ms | "
                    f"Loss: {loss_pct:4.1f}%"
                )
                print(status_str, end="\r", flush=True)

            # Precise rate limiting
            elapsed = time.perf_counter() - loop_start
            sleep_time = dt_ctrl - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[INFO] HIL Simulation interrupted by user.")
    finally:
        sock.close()
        if viewer is not None:
            viewer.close()

    total_wall_time = time.perf_counter() - start_time
    print("\n" + "=" * 65)
    print("  HIL Benchmark Execution Summary")
    print("=" * 65)
    print(f"  Total Duration:        {total_wall_time:.2f} seconds ({step} control steps)")
    print(f"  Packets Sent:          {total_packets_sent}")
    print(f"  Packets Received:      {total_packets_recv}")
    print(f"  Packet Loss Rate:      {(timeouts / total_packets_sent * 100) if total_packets_sent else 0:.2f}%")
    if rtt_history:
        print(f"  Round-Trip Time (RTT): Min {np.min(rtt_history):.3f} ms | Avg {np.mean(rtt_history):.3f} ms | Max {np.max(rtt_history):.3f} ms")
        print(f"  RTT Jitter (Std Dev):  {np.std(rtt_history):.3f} ms")
    if pos_errors:
        steady_errors = pos_errors[int(len(pos_errors) * 0.5):]
        print(f"  Steady-State Error:    {np.mean(steady_errors)*100:.2f} ± {np.std(steady_errors)*100:.2f} cm")
        print(f"  Average Tilt Angle:    {np.mean(tilts):.2f}°")
    print("=" * 65)


if __name__ == "__main__":
    run_hil_server()
