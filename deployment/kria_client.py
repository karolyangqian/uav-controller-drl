#!/usr/bin/env python3
"""
AMD Kria KR260 HIL Client (Python / ONNX Runtime).
Designed to run on the Kria KR260 Ubuntu 24.04 ARM Cortex-A53 PS.
Receives 16-D state over UDP, computes forward inference via ONNX Runtime, and transmits 4-D actions.
"""

import argparse
import socket
import struct
import time
import os
import sys
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime is required. Install via: pip install onnxruntime")
    sys.exit(1)

STATE_MAGIC = 0x55415653  # "UAVS"
ACTION_MAGIC = 0x55415641  # "UAVA"

STATE_STRUCT = struct.Struct('<II16f')  # magic (4B), seq_id (4B), obs (64B) = 72 bytes
ACTION_STRUCT = struct.Struct('<II4f')  # magic (4B), seq_id (4B), action (16B) = 24 bytes


def parse_args():
    parser = argparse.ArgumentParser(description="Kria KR260 HIL Inference Client (ONNX Runtime)")
    parser.add_argument("--bind-ip", type=str, default="0.0.0.0", help="IP address to bind client socket")
    parser.add_argument("--client-port", type=int, default=5005, help="Port to receive state from server")
    parser.add_argument("--server-ip", type=str, default="127.0.0.1", help="Simulation server IP (host laptop)")
    parser.add_argument("--server-port", type=int, default=5006, help="Port to send action to server")
    parser.add_argument("--model-path", type=str, default="deployment/models/quadrotor_policy.onnx", help="Path to ONNX policy")
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps to run (0 for infinite)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        # Check relative to script directory
        alt_path = os.path.join(os.path.dirname(__file__), "models", "quadrotor_policy.onnx")
        if os.path.exists(alt_path):
            args.model_path = alt_path
        else:
            print(f"[ERROR] Model file not found at: {args.model_path}")
            sys.exit(1)

    print("=" * 65)
    print("  AMD Kria KR260 HIL Inference Client (ONNX Runtime)")
    print("=" * 65)
    print(f"  Listening on:        {args.bind_ip}:{args.client_port}")
    print(f"  Server Target:       {args.server_ip}:{args.server_port}")
    print(f"  Policy Model:        {args.model_path} ({os.path.getsize(args.model_path)/1024:.2f} KB)")
    print("=" * 65)

    # Load ONNX model
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 2
    session = ort.InferenceSession(args.model_path, session_options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Setup UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.client_port))
    server_addr = (args.server_ip, args.server_port)

    step_count = 0
    inference_times = []
    print("\n[READY] Listening for simulation states... (Press Ctrl+C to stop)\n")

    try:
        while True:
            data, addr = sock.recvfrom(128)
            if len(data) != STATE_STRUCT.size:
                continue

            magic, seq_id, *obs = STATE_STRUCT.unpack(data)
            if magic != STATE_MAGIC:
                continue

            # Forward inference
            t_start = time.perf_counter()
            obs_tensor = np.array([obs], dtype=np.float32)
            action = session.run([output_name], {input_name: obs_tensor})[0][0]
            t_infer = (time.perf_counter() - t_start) * 1e6  # microseconds
            inference_times.append(t_infer)

            # Transmit ActionPacket back to simulation server
            action_data = ACTION_STRUCT.pack(ACTION_MAGIC, seq_id, float(action[0]), float(action[1]), float(action[2]), float(action[3]))
            sock.sendto(action_data, server_addr)

            step_count += 1
            if step_count % 50 == 0:
                recent_avg = np.mean(inference_times[-50:])
                print(f"[KRIA PS] Step {step_count:5d} | Seq {seq_id:5d} | ONNX Infer: {recent_avg:6.1f} µs | Action: [{action[0]:5.2f}, {action[1]:5.2f}, {action[2]:5.2f}, {action[3]:5.2f}]")

            if args.max_steps > 0 and step_count >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Client stopping...")
    finally:
        sock.close()

    if inference_times:
        print("\n" + "=" * 65)
        print("  Kria KR260 ONNX Inference Summary")
        print("=" * 65)
        print(f"  Total Inferences:      {step_count}")
        print(f"  Mean Latency:          {np.mean(inference_times):.2f} µs")
        print(f"  Min / Max Latency:     {np.min(inference_times):.2f} µs / {np.max(inference_times):.2f} µs")
        print(f"  Throughput Capacity:   {1e6 / np.mean(inference_times):.1f} inferences/second")
        print("=" * 65)


if __name__ == "__main__":
    main()
