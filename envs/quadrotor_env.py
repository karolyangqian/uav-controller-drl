"""
Gymnasium environment for Quadrotor 3D Hover & Setpoint Regulation in MuJoCo.
Features:
- Rate PID inner controller running at physics rate (500 Hz).
- DRL outer-loop policy running at 50 Hz.
- Jitter reduction via action low-pass filtering and action rate-of-change penalties.
- Domain randomization hooks (mass, inertia, wind, latency).
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from envs.pid_controller import RatePIDController


class QuadrotorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = None,
        ctrl_freq: int = 50,          # DRL policy frequency (Hz)
        physics_freq: int = 500,       # MuJoCo physics frequency (Hz)
        action_filter_alpha: float = 0.75,  # EMA filter factor for actions
        render_mode: str = None,
        max_episode_steps: int = 500,  # 10 seconds per episode
        randomize_domain: bool = False,
    ):
        super().__init__()

        if xml_path is None:
            xml_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models",
                "quadrotor.xml",
            )
        self.xml_path = xml_path

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.physics_freq = physics_freq
        self.ctrl_freq = ctrl_freq
        self.substeps = int(physics_freq / ctrl_freq)
        self.dt_sim = 1.0 / physics_freq
        self.dt_ctrl = 1.0 / ctrl_freq
        self.model.opt.timestep = self.dt_sim

        self.render_mode = render_mode
        self.viewer = None
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.randomize_domain = randomize_domain

        # Drone constants
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self.target_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        ]
        self.base_mass = float(self.model.body_mass[self.body_id])
        self.base_inertia = self.model.body_inertia[self.body_id].copy()
        self.hover_thrust = self.base_mass * 9.81  # ~7.848 N
        self.thrust_range = 6.0                     # Max command swing: hover +/- 6N
        self.max_rate_xy = 3.0                      # rad/s (~170 deg/s)
        self.max_rate_z = 1.5                       # rad/s (~85 deg/s)


        # PID Inner controller
        self.pid = RatePIDController()
        self.action_filter_alpha = action_filter_alpha
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.filtered_action = np.zeros(4, dtype=np.float32)

        # Action Space: [normalized_thrust, wx_cmd, wy_cmd, wz_cmd] in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation Space (16 dims):
        # - pos_err_body (3): target pos - drone pos in body frame
        # - vel_body (3): linear velocity in body frame
        # - z_body_world (3): body z-axis unit vector in world coordinates (tilt representation)
        # - ang_vel_body (3): body angular velocity [p, q, r]
        # - prev_action (4): previous normalized action
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

        # Wind disturbance state
        self.wind_force = np.zeros(3, dtype=np.float32)

    def _get_drone_state(self):
        """Extract pos, linear vel, rotation matrix, and angular vel."""
        pos = self.data.xpos[self.body_id].copy()
        # Rotation matrix of the body (3x3)
        rot_mat = self.data.xmat[self.body_id].reshape(3, 3).copy()
        # Linear velocity in world frame (cvel: [ang_vel, lin_vel] or qvel)
        # qvel for freejoint: [vx, vy, vz, wx, wy, wz] (world frame lin vel, body frame ang vel)
        lin_vel_world = self.data.qvel[0:3].copy()
        ang_vel_body = self.data.qvel[3:6].copy()

        # Express position error and linear velocity in drone body frame
        pos_err_world = self.target_pos - pos
        pos_err_body = rot_mat.T @ pos_err_world
        lin_vel_body = rot_mat.T @ lin_vel_world

        # Body z-axis in world frame (3rd column of rot_mat)
        z_body_world = rot_mat[:, 2].copy()

        return {
            "pos_world": pos,
            "pos_err_body": pos_err_body.astype(np.float32),
            "lin_vel_world": lin_vel_world.astype(np.float32),
            "lin_vel_body": lin_vel_body.astype(np.float32),
            "rot_mat": rot_mat,
            "z_body_world": z_body_world.astype(np.float32),
            "ang_vel_body": ang_vel_body.astype(np.float32),
        }

    def _get_obs(self):
        state = self._get_drone_state()
        obs = np.concatenate([
            state["pos_err_body"],     # 3
            state["lin_vel_body"],      # 3
            state["z_body_world"],      # 3
            state["ang_vel_body"],      # 3
            self.prev_action,           # 4
        ]).astype(np.float32)

        # Add realistic IMU/sensor noise if domain randomization is enabled
        if self.randomize_domain:
            noise = self.np_random.normal(0.0, 0.01, size=obs.shape).astype(np.float32)
            obs = obs + noise

        return obs


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.pid.reset()
        self.prev_action.fill(0.0)
        self.filtered_action.fill(0.0)

        # Randomize target position [x, y, z]
        self.target_pos = self.np_random.uniform(
            low=np.array([-1.5, -1.5, 1.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 2.0], dtype=np.float32),
        )
        self.data.mocap_pos[self.target_mocap_id] = self.target_pos

        # Initial drone position with slight perturbation around origin
        drone_init_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32) + self.np_random.uniform(-0.2, 0.2, size=3)
        self.data.qpos[0:3] = drone_init_pos

        # Initial orientation: small random tilt (quaternion: [w, x, y, z])
        init_tilt = self.np_random.uniform(-0.1, 0.1, size=2)  # small roll and pitch
        self.data.qpos[3:7] = np.array([1.0, init_tilt[0], init_tilt[1], 0.0], dtype=np.float32)
        # Normalize quaternion
        self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])

        # Velocities: 0
        self.data.qvel[:] = 0.0

        # Domain randomization (if enabled)
        if self.randomize_domain:
            # Vary mass by +/- 15%
            mass_factor = self.np_random.uniform(0.85, 1.15)
            self.model.body_mass[self.body_id] = self.base_mass * mass_factor
            # Vary inertia tensor proportionally
            inertia_jitter = self.np_random.uniform(0.9, 1.1, size=3)
            self.model.body_inertia[self.body_id] = self.base_inertia * mass_factor * inertia_jitter
            # Base steady wind vector (up to 1.5 N horizontal, 0.4 N vertical)
            self.wind_base = self.np_random.uniform([-1.5, -1.5, -0.4], [1.5, 1.5, 0.4]).astype(np.float32)
            self.wind_force = self.wind_base.copy()
            # Random initial tilt up to +/- 15 deg (recovery practice)
            init_tilt = self.np_random.uniform(-0.25, 0.25, size=2)
            self.data.qpos[3:7] = np.array([1.0, init_tilt[0], init_tilt[1], 0.0], dtype=np.float32)
            self.data.qpos[3:7] /= np.linalg.norm(self.data.qpos[3:7])
        else:
            self.model.body_mass[self.body_id] = self.base_mass
            self.model.body_inertia[self.body_id] = self.base_inertia.copy()
            self.wind_base = np.zeros(3, dtype=np.float32)
            self.wind_force.fill(0.0)

        # Forward kinematics to update body positions
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {"target_pos": self.target_pos.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        # Low-pass filter action (EMA) to prevent high-frequency jitter
        self.filtered_action = (
            self.action_filter_alpha * action + (1.0 - self.action_filter_alpha) * self.filtered_action
        )

        # Unscale action to physical commands
        # 0: vertical thrust (N)
        thrust_cmd = self.hover_thrust + self.filtered_action[0] * self.thrust_range
        # 1-3: angular rate commands [wx, wy, wz] (rad/s)
        rate_cmd = np.array([
            self.filtered_action[1] * self.max_rate_xy,
            self.filtered_action[2] * self.max_rate_xy,
            self.filtered_action[3] * self.max_rate_z,
        ], dtype=np.float32)

        # Update dynamic wind gusts during episode
        if self.randomize_domain:
            turb = self.np_random.normal(0.0, 0.15, size=3).astype(np.float32)
            pulse = self.np_random.uniform(-1.0, 1.0, size=3).astype(np.float32) if self.np_random.random() < 0.05 else 0.0
            self.wind_force = self.wind_base + turb + pulse

        # Substep physics loop (Inner loop: 500 Hz)
        for _ in range(self.substeps):
            # Apply wind disturbance force to drone center
            if self.randomize_domain:
                self.data.xfrc_applied[self.body_id, 0:3] = self.wind_force

            current_rates = self.data.qvel[3:6]
            motor_forces = self.pid.update(
                thrust_cmd=thrust_cmd,
                rate_cmd=rate_cmd,
                current_rate=current_rates,
                dt=self.dt_sim,
            )
            self.data.ctrl[:] = motor_forces
            mujoco.mj_step(self.model, self.data)


        self.step_count += 1
        state = self._get_drone_state()
        obs = self._get_obs()

        # Metrics for reward and termination
        pos_error_norm = np.linalg.norm(self.target_pos - state["pos_world"])
        lin_vel_norm = np.linalg.norm(state["lin_vel_world"])
        ang_vel_norm = np.linalg.norm(state["ang_vel_body"])
        z_up = state["z_body_world"][2]  # Cosine of tilt angle (1.0 = upright, 0.0 = 90 deg tilt)
        action_diff = np.linalg.norm(action - self.prev_action)

        # Reward shaping:
        # 1. Exponential position tracking reward (peak = 1.0 at target)
        r_pos = np.exp(-2.0 * (pos_error_norm ** 2))
        # 2. Linear velocity damping near target
        r_vel = -0.1 * lin_vel_norm
        # 3. Upright alignment reward
        r_upright = 0.5 * max(0.0, z_up)
        # 4. Angular velocity damping (anti-wobble)
        r_ang_vel = -0.05 * ang_vel_norm
        # 5. Action smoothness penalty (ANTI-JITTER)
        r_smooth = -0.05 * (action_diff ** 2) - 0.01 * np.sum(action ** 2)
        # 6. Survival bonus
        r_alive = 0.5

        reward = float(r_pos + r_vel + r_upright + r_ang_vel + r_smooth + r_alive)

        # Terminations & Truncations
        terminated = False
        # Crash condition 1: tilted past ~75 degrees (cos(75 deg) ~= 0.26)
        if z_up < 0.25:
            terminated = True
            reward -= 10.0
        # Crash condition 2: hit the ground or too low
        elif state["pos_world"][2] < 0.1:
            terminated = True
            reward -= 10.0
        # Out of bounds: drifted too far (> 5m from origin)
        elif np.linalg.norm(state["pos_world"][0:2]) > 5.0 or state["pos_world"][2] > 4.0:
            terminated = True
            reward -= 10.0

        truncated = self.step_count >= self.max_episode_steps

        info = {
            "pos_error": float(pos_error_norm),
            "lin_vel": float(lin_vel_norm),
            "z_up": float(z_up),
            "r_pos": float(r_pos),
            "action_diff": float(action_diff),
        }

        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
