# Project Artifacts, Media, and Telemetry Documentation

This document catalogs all figures, video clips, 3D trajectory renders, and numerical telemetry logs generated during the training, evaluation, and benchmarking of the DRL Quadrotor flight controller in MuJoCo.

---

## 📁 Artifacts Directory Structure

```
.
├── figures/
│   ├── training_curve.png          # Day 2: Baseline PPO training curve (Reward & Length)
│   ├── robust_training_curve.png   # Day 3: Robust PPO training curve under domain randomization
│   ├── comparison_benchmark.png    # Day 3: Baseline vs Robust head-to-head performance
│   ├── step_response.png           # Day 4: 3-panel 3D step response (Pos, Tilt, Actions)
│   └── trajectory_3d.png           # Day 4: 3D spatial flight trajectory in MuJoCo arena
├── media/
│   ├── flight_simulation.mp4       # Full 50 FPS H.264 video of autonomous flight mission
│   ├── flight_simulation.gif       # 25 FPS animated GIF for web and documentation embedding
│   ├── render_flight_snapshot.png  # High-resolution render of high-speed translation maneuver
│   └── render_hover_snapshot.png   # High-resolution render of stable waypoint hover
├── telemetry/
│   ├── flight_telemetry.csv        # 27-column time-series flight telemetry log (50 Hz)
│   └── step_response_telemetry.csv # Time-series log of 3D step transition dynamics
├── deployment/
│   ├── models/
│   │   └── quadrotor_policy.onnx   # Exported ONNX model (9.36 KB footprint)
│   ├── weights.h                   # Signed 16-bit Q8.8 fixed-point C arrays for FPGA
│   ├── test_forward.c              # Standalone embedded C testbench (1.38 us latency)
│   ├── kria_client.c               # Day 5: Pure C Q8.8 HIL client (< 2 us latency)
│   └── kria_client.py              # Day 5: Python ONNX Runtime HIL client
├── scripts/
│   ├── hil_sim_server.py           # Day 5: 50 Hz control / 500 Hz physics HIL server
│   └── benchmark_hil.py            # Day 5: Automated HIL benchmark & latency stress test
└── docs/
    └── HIL_DEPLOYMENT_GUIDE.md     # Day 5: Physical AMD Kria KR260 deployment manual
```

---

## 🎥 1. Flight Media & Video Clips

| File | Type | Description |
| :--- | :--- | :--- |
| **[`media/flight_simulation.mp4`](file:///home/karol/ITB/kuliah/uav-controller/media/flight_simulation.mp4)** | Video (H.264, 50 FPS) | 5.0-second continuous flight mission showing quadrotor takeoff, translation, brake, and hover under aerodynamic wind gusts. Playable via VLC, MPV, or web browser. |
| **[`media/flight_simulation.gif`](file:///home/karol/ITB/kuliah/uav-controller/media/flight_simulation.gif)** | Animated GIF (25 FPS) | Lightweight looping clip of the autonomous flight mission for presentation slides and markdown previews. |
| **[`media/render_flight_snapshot.png`](file:///home/karol/ITB/kuliah/uav-controller/media/render_flight_snapshot.png)** | High-Res Image | Action shot at $t = 0.8\text{ s}$ showing the quadrotor pitching forward into a high-speed acceleration maneuver. |
| **[`media/render_hover_snapshot.png`](file:///home/karol/ITB/kuliah/uav-controller/media/render_hover_snapshot.png)** | High-Res Image | Steady-state shot at $t = 4.0\text{ s}$ showing the quadrotor stabilized inside the translucent green target sphere. |

To play the video from terminal:
```bash
mpv media/flight_simulation.mp4
# or
xdg-open media/flight_simulation.mp4
```

---

## 📈 2. Training Curves & Performance Graphs

### A. Learning Progression
* **[`figures/training_curve.png`](file:///home/karol/ITB/kuliah/uav-controller/figures/training_curve.png)**: Dual-panel plot of the Day 2 baseline training run ($1.5\text{M}$ steps). Shows episode reward rising from $\approx 40$ to $951.5$ and survival steps reaching $500/500$.
* **[`figures/robust_training_curve.png`](file:///home/karol/ITB/kuliah/uav-controller/figures/robust_training_curve.png)**: Dual-panel plot of the Day 3 robust training run under domain randomization. Shows convergence to $\approx 916$ reward despite active wind gusts and mass shifts.

### B. Comparative Disturbance Benchmark
* **[`figures/comparison_benchmark.png`](file:///home/karol/ITB/kuliah/uav-controller/figures/comparison_benchmark.png)**: Head-to-head bar chart comparing Baseline vs. Robust policies under severe disturbances:
  * Steady-state error: $35.6\text{ cm}$ (Baseline) $\to$ **$13.5\text{ cm}$ (Robust)** ($62\%$ reduction).
  * Target acquisition success rate: $5.0\%$ $\to$ **$85.0\%$**.
  * Active wind lean angle: $7.8^\circ \to \mathbf{10.3^\circ}$.

### C. 3D Trajectory & Step Dynamics
* **[`figures/trajectory_3d.png`](file:///home/karol/ITB/kuliah/uav-controller/figures/trajectory_3d.png)**: 3D spatial path plot depicting the drone departing from start pose $[0, 0, 1]\text{ m}$ and converging directly into the $15\text{ cm}$ tolerance sphere around target coordinates $[1.0, 1.0, 1.8]\text{ m}$.
* **[`figures/step_response.png`](file:///home/karol/ITB/kuliah/uav-controller/figures/step_response.png)**: 3-panel step response telemetry ($[0, 0, 1] \to [1, 1, 2]\text{ m}$):
  * **Panel 1 (Positions)**: $X, Y, Z$ trajectory tracking vs. reference targets ($T_r = 1.91\text{ s}$, $e_{ss} = 5.96\text{ cm}$).
  * **Panel 2 (Attitude)**: Smooth, controlled tilt transition peaking at $14.1^\circ$.
  * **Panel 3 (Control Actions)**: Smooth continuous actions $[a_T, a_{\omega_x}, a_{\omega_y}, a_{\omega_z}]$ without high-frequency chattering.

---

## 📊 3. Telemetry Logs & CSV Schemas

### A. Flight Mission Telemetry: [`telemetry/flight_telemetry.csv`](file:///home/karol/ITB/kuliah/uav-controller/telemetry/flight_telemetry.csv)
Recorded at $50\text{ Hz}$ across 250 control steps ($5.0\text{ seconds}$).

| Column Name | Unit | Description |
| :--- | :--- | :--- |
| `step` | int | Control step index ($0$ to $249$) |
| `time_sec` | s | Wall simulation time ($0.00$ to $4.98\text{ s}$) |
| `pos_x_m`, `pos_y_m`, `pos_z_m` | m | Current 3D position of the quadrotor center in world frame |
| `target_x_m`, `target_y_m`, `target_z_m` | m | Commanded 3D target setpoint coordinates |
| `pos_error_m` | m | Euclidean 3D distance error to target ($\|\mathbf{p}_{\text{target}} - \mathbf{p}\|$) |
| `vel_x_mps`, `vel_y_mps`, `vel_z_mps` | m/s | Linear velocities in world frame |
| `roll_deg`, `pitch_deg`, `yaw_deg` | deg | Drone attitude Euler angles |
| `omega_p_radps`, `omega_q_radps`, `omega_r_radps` | rad/s | Body angular velocities ($[p, q, r]$) |
| `action_thrust`, `action_wx`, `action_wy`, `action_wz` | [-1, 1] | Commanded actions from the DRL policy |
| `motor_0_N`, `motor_1_N`, `motor_2_N`, `motor_3_N` | N | Individual motor thrusts computed by inner PID mixer ($[0, 4.5]\text{ N}$) |
| `wind_fx_N`, `wind_fy_N`, `wind_fz_N` | N | Active aerodynamic disturbance force vectors applied to the drone |

### B. Step Response Telemetry: [`telemetry/step_response_telemetry.csv`](file:///home/karol/ITB/kuliah/uav-controller/telemetry/step_response_telemetry.csv)
Contains 400 rows ($8.0\text{ seconds}$) of position tracking, attitude tilt, and normalized action commands during the 3D step transition ($[0, 0, 1] \to [1, 1, 2]$).

---

## 💻 4. Python Snippet: Loading & Plotting Telemetry

To inspect or plot custom telemetry curves from Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load telemetry CSV
df = pd.read_csv("telemetry/flight_telemetry.csv")

# Plot Position Error and Motor Thrusts over Time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax1.plot(df["time_sec"], df["pos_error_m"] * 100, color="crimson", linewidth=2.0)
ax1.set_ylabel("Position Error (cm)")
ax1.set_title("Live Flight Telemetry Analysis")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(df["time_sec"], df["motor_0_N"], label="Motor 0 (Front-Right)")
ax2.plot(df["time_sec"], df["motor_1_N"], label="Motor 1 (Front-Left)")
ax2.plot(df["time_sec"], df["motor_2_N"], label="Motor 2 (Rear-Left)")
ax2.plot(df["time_sec"], df["motor_3_N"], label="Motor 3 (Rear-Right)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Thrust (N)")
ax2.legend(ncol=2)
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
```

---

## ⚡ 5. Hardware-in-the-Loop (HIL) Suite (Day 5)

| File | Type | Description |
| :--- | :--- | :--- |
| **[`scripts/hil_sim_server.py`](file:///home/karol/ITB/kuliah/uav-controller/scripts/hil_sim_server.py)** | Python Script | 50 Hz control / 500 Hz physics simulation server. Packages 16-D body-frame observations into 72-byte binary UDP packets and steps physics upon receiving 24-byte action packets. Features automatic hover failsafe. |
| **[`deployment/kria_client.c`](file:///home/karol/ITB/kuliah/uav-controller/deployment/kria_client.c)** | C Source | Standalone POSIX C socket client using signed 16-bit Q8.8 fixed-point arithmetic (`weights.h`). Inference latency: **$1.38\,\mu\text{s}$** ($> 720\text{ kHz}$ throughput). Binary size: $37\text{ KB}$. |
| **[`deployment/kria_client.py`](file:///home/karol/ITB/kuliah/uav-controller/deployment/kria_client.py)** | Python Script | Client running the $9.36\text{ KB}$ ONNX model on the Kria KR260 ARM Cortex-A53 PS using `onnxruntime`. |
| **[`scripts/benchmark_hil.py`](file:///home/karol/ITB/kuliah/uav-controller/scripts/benchmark_hil.py)** | Python Script | Automated closed-loop HIL benchmark testbench. Measures round-trip time (RTT), jitter, hover error, and tests network latency injection ($2\text{ ms}$) and packet drops ($5\%$). |
| **[`docs/HIL_DEPLOYMENT_GUIDE.md`](file:///home/karol/ITB/kuliah/uav-controller/docs/HIL_DEPLOYMENT_GUIDE.md)** | Markdown Guide | Step-by-step setup guide for physical Ethernet connection between host laptop (`192.168.1.10`) and AMD Kria KR260 (`192.168.1.11`). |

---

## 🔄 6. How to Re-Generate All Artifacts & Run Benchmarks

To re-run and regenerate all figures, media clips, and telemetry files:

```bash
cd /home/karol/ITB/kuliah/uav-controller
source .venv/bin/activate

# 1. Re-generate comparative bar charts and 3D trajectory plot:
python scripts/plot_comparison.py

# 2. Re-generate 3D step response plot & telemetry CSV:
python scripts/evaluate_step_response.py

# 3. Re-render video clips (MP4/GIF), high-res snapshots, and flight telemetry CSV:
python scripts/generate_media_and_telemetry.py

# 4. Run closed-loop HIL benchmark with Embedded C Q8.8 client:
python scripts/benchmark_hil.py --client c --duration 10.0

# 5. Run closed-loop HIL benchmark with Python ONNX client:
python scripts/benchmark_hil.py --client python --duration 10.0

# 6. Run HIL network stress test (2 ms network delay + 5% packet drops):
python scripts/benchmark_hil.py --client c --duration 10.0 --latency-ms 2.0 --drop-rate 0.05
```
