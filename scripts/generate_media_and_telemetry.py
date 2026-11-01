"""
Render flight clips/images from MuJoCo and log comprehensive time-series telemetry to CSV.
Outputs:
- telemetry/flight_telemetry.csv
- media/render_flight_snapshot.png
- media/render_hover_snapshot.png
- media/flight_simulation.mp4
- media/flight_simulation.gif
"""
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import mujoco
from stable_baselines3 import PPO

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadrotor_env import QuadrotorEnv


def generate_media_and_telemetry(
    model_path: str = "./checkpoints/robust/best_model/best_model.zip",
    media_dir: str = "./media",
    telemetry_dir: str = "./telemetry",
    total_steps: int = 250, # 5.0 seconds at 50 Hz
):
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(telemetry_dir, exist_ok=True)

    print("=" * 65)
    print("Generating Flight Media & Comprehensive Telemetry Logs")
    print(f"Policy:    {model_path}")
    print(f"Media:     {media_dir}")
    print(f"Telemetry: {telemetry_dir}")
    print("=" * 65)

    env = QuadrotorEnv(randomize_domain=True)
    model = PPO.load(model_path)

    # Setup offscreen renderer: 640x480 standard resolution
    width, height = 640, 480
    renderer = mujoco.Renderer(env.model, height=height, width=width)


    # Initialize environment
    obs, info = env.reset(seed=101)
    target_pos = info["target_pos"]

    # Configure camera for an elevated perspective view of the quadrotor arena
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.5, 0.5, 1.4]
    camera.distance = 3.5
    camera.elevation = -22.0
    camera.azimuth = 135.0

    frames = []
    telemetry_rows = []

    print(f"Recording {total_steps} flight steps (5.0s mission)...")
    for step in range(total_steps):
        t = step * env.dt_ctrl

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, step_info = env.step(action)

        state = env._get_drone_state()
        pos = state["pos_world"]
        vel = state["lin_vel_world"]
        ang_vel = state["ang_vel_body"]

        # Calculate Euler angles (roll, pitch, yaw) in degrees
        R = state["rot_mat"]
        pitch_rad = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        roll_rad = np.arctan2(R[2, 1], R[2, 2])
        yaw_rad = np.arctan2(R[1, 0], R[0, 0])

        # Control commands and motor forces
        ctrl_thrust = float(env.hover_thrust + env.filtered_action[0] * env.thrust_range)
        motor_forces = env.data.ctrl.copy()
        wind_vec = env.wind_force.copy()

        # Telemetry record
        telemetry_rows.append({
            "step": step,
            "time_sec": round(t, 3),
            "pos_x_m": round(float(pos[0]), 4),
            "pos_y_m": round(float(pos[1]), 4),
            "pos_z_m": round(float(pos[2]), 4),
            "target_x_m": round(float(target_pos[0]), 4),
            "target_y_m": round(float(target_pos[1]), 4),
            "target_z_m": round(float(target_pos[2]), 4),
            "pos_error_m": round(float(step_info["pos_error"]), 4),
            "vel_x_mps": round(float(vel[0]), 4),
            "vel_y_mps": round(float(vel[1]), 4),
            "vel_z_mps": round(float(vel[2]), 4),
            "roll_deg": round(float(np.degrees(roll_rad)), 2),
            "pitch_deg": round(float(np.degrees(pitch_rad)), 2),
            "yaw_deg": round(float(np.degrees(yaw_rad)), 2),
            "omega_p_radps": round(float(ang_vel[0]), 3),
            "omega_q_radps": round(float(ang_vel[1]), 3),
            "omega_r_radps": round(float(ang_vel[2]), 3),
            "action_thrust": round(float(action[0]), 4),
            "action_wx": round(float(action[1]), 4),
            "action_wy": round(float(action[2]), 4),
            "action_wz": round(float(action[3]), 4),
            "motor_0_N": round(float(motor_forces[0]), 3),
            "motor_1_N": round(float(motor_forces[1]), 3),
            "motor_2_N": round(float(motor_forces[2]), 3),
            "motor_3_N": round(float(motor_forces[3]), 3),
            "wind_fx_N": round(float(wind_vec[0]), 3),
            "wind_fy_N": round(float(wind_vec[1]), 3),
            "wind_fz_N": round(float(wind_vec[2]), 3),
        })

        # Render frame
        renderer.update_scene(env.data, camera=camera)
        frame_rgb = renderer.render()
        frames.append(frame_rgb)

        # Save snapshot of mid-flight maneuver (step 40)
        if step == 40:
            flight_img = Image.fromarray(frame_rgb)
            flight_img.save(os.path.join(media_dir, "render_flight_snapshot.png"))

        # Save snapshot of precision hover (step 200)
        if step == 200:
            hover_img = Image.fromarray(frame_rgb)
            hover_img.save(os.path.join(media_dir, "render_hover_snapshot.png"))

        if term or trunc:
            break

    env.close()

    # 1. Save Telemetry CSV
    telemetry_csv = os.path.join(telemetry_dir, "flight_telemetry.csv")
    df = pd.DataFrame(telemetry_rows)
    df.to_csv(telemetry_csv, index=False)
    print(f"[OK] Telemetry saved to {telemetry_csv} ({len(df)} rows)")

    # 2. Save Animated GIF (downsampled to 25 fps for compact size)
    gif_path = os.path.join(media_dir, "flight_simulation.gif")
    gif_frames = [Image.fromarray(f) for f in frames[::2]] # every 2nd frame = 25 fps
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=40, # 40ms per frame = 25 fps
        loop=0,
    )
    print(f"[OK] Animated GIF saved to {gif_path} ({os.path.getsize(gif_path) / 1024:.1f} KB)")

    # 3. Save MP4 Video using ffmpeg
    mp4_path = os.path.join(media_dir, "flight_simulation.mp4")
    # Save raw frames to temporary pipe or files
    temp_raw = "/tmp/mujoco_flight_raw.yuv"
    raw_bytes = b"".join(f.tobytes() for f in frames)
    with open(temp_raw, "wb") as f:
        f.write(raw_bytes)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", "50",
        "-i", temp_raw,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "22",
        mp4_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if os.path.exists(temp_raw):
        os.remove(temp_raw)

    print(f"[OK] MP4 Video saved to {mp4_path} ({os.path.getsize(mp4_path) / 1024:.1f} KB)")
    print(f"[OK] High-Res Snapshots saved to {media_dir}/render_flight_snapshot.png and render_hover_snapshot.png")
    print("=" * 65)


if __name__ == "__main__":
    generate_media_and_telemetry()
