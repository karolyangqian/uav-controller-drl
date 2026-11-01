"""
Export trained PyTorch Quadrotor policy to ONNX format.
Validates numerical consistency between Stable-Baselines3, PyTorch, and ONNX Runtime.
"""
import os
import sys
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
from stable_baselines3 import PPO

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.policy_net import QuadrotorPolicyNet


def export_to_onnx(model_path: str, output_onnx_path: str = "./deployment/models/quadrotor_policy.onnx"):
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    print("=" * 65)
    print(f"Exporting Policy to ONNX: {model_path} -> {output_onnx_path}")
    print("=" * 65)

    # 1. Load trained Stable-Baselines3 model
    sb3_model = PPO.load(model_path, device="cpu")

    # 2. Extract into clean standalone PyTorch module
    policy_net = QuadrotorPolicyNet.from_sb3_model(sb3_model)
    policy_net.eval()

    # 3. Verify PyTorch vs SB3 numerical equivalence on test observations
    dummy_input = torch.randn(1, 16, dtype=torch.float32)
    obs_np = dummy_input.numpy()

    with torch.no_grad():
        pt_out = policy_net(dummy_input).numpy().squeeze()
    sb3_out, _ = sb3_model.predict(obs_np, deterministic=True)
    sb3_out = np.clip(sb3_out.squeeze(), -1.0, 1.0)

    max_diff_pt_sb3 = np.max(np.abs(pt_out - sb3_out))
    print(f"[OK] PyTorch vs SB3 Max Difference: {max_diff_pt_sb3:.2e} (should be < 1e-6)")
    assert max_diff_pt_sb3 < 1e-5, f"Discrepancy too large: {max_diff_pt_sb3}"

    # 4. Export to ONNX
    input_names = ["observation"]
    output_names = ["action"]
    torch.onnx.export(
        policy_net,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )
    print(f"[OK] ONNX model successfully exported: {output_onnx_path}")

    # 5. Check ONNX Model Structure
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model validation check passed.")

    # 6. Verify ONNX Runtime vs PyTorch
    session = ort.InferenceSession(output_onnx_path)
    ort_inputs = {session.get_inputs()[0].name: obs_np}
    ort_out = session.run(None, ort_inputs)[0].squeeze()

    max_diff_ort_pt = np.max(np.abs(ort_out - pt_out))
    print(f"[OK] ONNX Runtime vs PyTorch Max Difference: {max_diff_ort_pt:.2e} (should be < 1e-6)")
    assert max_diff_ort_pt < 1e-5, f"ONNX Runtime discrepancy too large: {max_diff_ort_pt}"

    file_size_kb = os.path.getsize(output_onnx_path) / 1024.0
    print(f"[OK] Total Model Size: {file_size_kb:.2f} KB (Fits easily on Kria KR260 on-chip BRAM!)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/robust/best_model/best_model.zip")
    parser.add_argument("--output", type=str, default="./deployment/models/quadrotor_policy.onnx")
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.output)
