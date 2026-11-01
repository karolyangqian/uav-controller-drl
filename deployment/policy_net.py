"""
Standalone PyTorch Policy Network for Quadrotor DRL Controller.
Pure PyTorch implementation without Stable-Baselines3 dependencies, optimized for ONNX export
and FPGA / Edge SoC deployment.
"""
import torch
import torch.nn as nn


class QuadrotorPolicyNet(nn.Module):
    """
    Compact Multi-Layer Perceptron (MLP) flight policy:
    Input:  16-dimensional observation vector (body frame)
    Layer 1: Linear(16, 64) -> ReLU
    Layer 2: Linear(64, 64) -> ReLU
    Layer 3: Linear(64, 4)  -> Output [T_z, wx, wy, wz] clamped to [-1.0, 1.0]
    """
    def __init__(self, in_features: int = 16, hidden_dim: int = 64, out_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.clamp(out, -1.0, 1.0)

    @classmethod
    def from_sb3_model(cls, sb3_model):
        """Construct and load weights directly from a trained Stable-Baselines3 PPO model."""
        policy_net = cls()
        sb3_state = sb3_model.policy.state_dict()

        # Map SB3 state dict to standalone policy net
        state_dict = {
            "net.0.weight": sb3_state["mlp_extractor.policy_net.0.weight"],
            "net.0.bias": sb3_state["mlp_extractor.policy_net.0.bias"],
            "net.2.weight": sb3_state["mlp_extractor.policy_net.2.weight"],
            "net.2.bias": sb3_state["mlp_extractor.policy_net.2.bias"],
            "net.4.weight": sb3_state["action_net.weight"],
            "net.4.bias": sb3_state["action_net.bias"],
        }
        policy_net.load_state_dict(state_dict)
        policy_net.eval()
        return policy_net
