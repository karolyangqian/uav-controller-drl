"""
Inner-loop rate PID controller and motor mixer for X-configuration quadrotor.
Translates high-level collective thrust and body angular rate commands [T_z, wx, wy, wz]
into individual motor forces [F0, F1, F2, F3] at physics simulation rate (e.g. 500 Hz).
"""
import numpy as np


class RatePIDController:
    def __init__(
        self,
        d: float = 0.10607,       # Arm moment arm in meters (L / sqrt(2))
        km: float = 0.016,        # Yaw torque coefficient (N*m / N)
        f_min: float = 0.0,       # Min thrust per motor (N)
        f_max: float = 4.5,       # Max thrust per motor (N)
        kp: np.ndarray = None,    # Proportional gains for [wx, wy, wz]
        ki: np.ndarray = None,    # Integral gains for [wx, wy, wz]
        kd: np.ndarray = None,    # Derivative gains for [wx, wy, wz]
    ):
        self.d = d
        self.km = km
        self.f_min = f_min
        self.f_max = f_max

        # Default tuned gains for 0.8kg quadrotor:
        # Inertia ~0.005 kg*m^2. kp = 0.15 gives ~30 rad/s bandwidth.
        self.kp = np.array([0.15, 0.15, 0.10], dtype=np.float32) if kp is None else np.array(kp, dtype=np.float32)
        self.ki = np.array([0.05, 0.05, 0.02], dtype=np.float32) if ki is None else np.array(ki, dtype=np.float32)
        self.kd = np.array([0.003, 0.003, 0.001], dtype=np.float32) if kd is None else np.array(kd, dtype=np.float32)

        self.integral = np.zeros(3, dtype=np.float32)
        self.prev_error = np.zeros(3, dtype=np.float32)

    def reset(self):
        """Reset PID states."""
        self.integral.fill(0.0)
        self.prev_error.fill(0.0)

    def update(
        self,
        thrust_cmd: float,
        rate_cmd: np.ndarray,
        current_rate: np.ndarray,
        dt: float = 0.002,
    ) -> np.ndarray:
        """
        Compute motor thrusts from collective thrust and body angular rates.

        Args:
            thrust_cmd: Desired total vertical thrust (N)
            rate_cmd: Desired body angular velocity [wx, wy, wz] (rad/s)
            current_rate: Measured body angular velocity [wx, wy, wz] (rad/s)
            dt: Time step for PID integration/differentiation (s)

        Returns:
            motor_forces: np.ndarray shape (4,) in range [f_min, f_max]
        """
        error = rate_cmd - current_rate

        # Anti-windup integral accumulation
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)

        # Derivative on error
        derivative = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error.copy()

        # Compute commanded body torques [tau_x, tau_y, tau_z]
        tau = self.kp * error + self.ki * self.integral + self.kd * derivative

        tau_x = tau[0]
        tau_y = tau[1]
        tau_z = tau[2]

        # Inverse mixer matrix for X-4 configuration
        # F0 (Front-Right, CCW)
        # F1 (Front-Left,  CW)
        # F2 (Rear-Left,   CCW)
        # F3 (Rear-Right,  CW)
        inv_4d = 1.0 / (4.0 * self.d)
        inv_4km = 1.0 / (4.0 * self.km)
        f_base = 0.25 * thrust_cmd

        f0 = f_base - inv_4d * tau_x - inv_4d * tau_y - inv_4km * tau_z
        f1 = f_base + inv_4d * tau_x - inv_4d * tau_y + inv_4km * tau_z
        f2 = f_base + inv_4d * tau_x + inv_4d * tau_y - inv_4km * tau_z
        f3 = f_base - inv_4d * tau_x + inv_4d * tau_y + inv_4km * tau_z

        forces = np.array([f0, f1, f2, f3], dtype=np.float32)
        return np.clip(forces, self.f_min, self.f_max)
