# DRL Quadrotor Controller in MuJoCo (HIL with AMD Kria KR260)

Deep Reinforcement Learning (DRL) 3D flight controller trained in **MuJoCo** with hardware-in-the-loop (HIL) deployment on the **AMD Kria KR260** FPGA/SoC platform.

---

## 🚁 Architecture Overview

```
                      Observation (16-D Body-Frame Features)
    ┌────────────────────────────────────────────────────────────────────────┐
    │  - Relative position error: e_p,body (3)                               │
    │  - Linear velocity: v_body (3)                                         │
    │  - Projected gravity / Body z-axis in world: z_body (3)                │
    │  - Angular velocity: omega_body (3)                                    │
    │  - Previous action: a_t-1 (4)                                          │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                    DRL Outer-Loop Policy (50 Hz)                       │
    │               MLP: 16 -> 64 -> 64 -> 4 (ReLU Activation)               │
    │                                                                        │
    │  Action Space: [T_z, wx_cmd, wy_cmd, wz_cmd] in [-1, 1]^4              │
    │  + Exponential Moving Average (EMA) Action Low-Pass Filter (alpha=0.75)│
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                   Inner-Loop Rate PID (500 Hz)                         │
    │       Translates [T_z, wx, wy, wz] -> Motor Forces [F0, F1, F2, F3]    │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                      MuJoCo Physics Simulation                         │
    │           0.8 kg Quadrotor (X-Configuration), timestep = 0.002s        │
    └────────────────────────────────────────────────────────────────────────┘
```

---

## 💡 Anti-Jitter & Robustness Engineering

Addressing real-world deployment issues (actuator chattering and environment brittleness):
1. **Actuator Smoothing:**
   - **Action Filter:** Raw policy actions are filtered through an exponential moving average (EMA) $\mathbf{a}_t^{\text{applied}} = \alpha \mathbf{a}_t + (1 - \alpha) \mathbf{a}_{t-1}^{\text{applied}}$.
   - **Action Smoothness Penalties:** $r_{\text{smooth}} = -\lambda_1 \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 - \lambda_2 \|\mathbf{a}_t\|^2$.
2. **Domain Randomization:**
   - Mass randomized by $\pm 15\%$.
   - Wind disturbance force vectors applied to the body.
   - Initial orientation perturbations up to $\pm 15^\circ$.
3. **Hardware-Friendly Design:**
   - Pure **ReLU** activations enable efficient, low-latency synthesis onto FPGA PL (AMD Kria KR260).
   - Compact network footprint ($< 6\text{ KB}$ in INT8) fits directly into on-chip BRAM.

---

## 📁 Repository Structure

```
.
├── models/
│   └── quadrotor.xml          # MuJoCo MJCF model (0.8 kg X-quad, 4 thruster actuators)
├── envs/
│   ├── __init__.py
│   ├── pid_controller.py      # 500 Hz rate PID controller + inverse mixer
│   └── quadrotor_env.py       # Gymnasium environment wrapper with anti-jitter filters
├── figures/                   # Training curves, 3D trajectories & step response plots
│   ├── training_curve.png
│   ├── robust_training_curve.png
│   ├── comparison_benchmark.png
│   ├── step_response.png
│   └── trajectory_3d.png
├── media/                     # Rendered video clips & high-res snapshots
│   ├── flight_simulation.mp4  # 50 FPS H.264 video of flight mission
│   ├── flight_simulation.gif  # Animated GIF preview
│   ├── render_flight_snapshot.png
│   └── render_hover_snapshot.png
├── telemetry/                 # Time-series flight logs (CSV)
│   ├── flight_telemetry.csv
│   └── step_response_telemetry.csv
├── deployment/                # Edge SoC / FPGA export artifacts
│   ├── models/
│   │   └── quadrotor_policy.onnx # 9.36 KB ONNX model
│   ├── weights.h              # Q8.8 fixed-point C arrays for FPGA
│   └── test_forward.c         # C testbench (1.38 us latency)
├── docs/
│   ├── index.html             # Interactive academic project landing page & report
│   ├── FORMAT_GUIDE.md        # Report formatting rules & Pandoc compilation guide
│   ├── ARTIFACTS_AND_TELEMETRY.md # Detailed catalog of all media, plots & telemetry
│   ├── preamble.tex           # LaTeX styling template (Times New Roman 12pt, justified, black)
│   ├── report_template.md     # Pre-structured academic report draft
│   └── report.pdf             # Compiled report PDF

├── scripts/
│   ├── test_env.py            # Day 1 verification & speed benchmarking
│   ├── train_ppo.py           # Day 2 multi-core PPO training script (ReLU MLP)
│   ├── eval_policy.py         # Real-time interactive 3D viewer & evaluation
│   ├── benchmark_policy.py    # Quantitative tracking RMSE, steady-state, and tilt metrics
│   ├── plot_training.py       # Publication-ready reward & episode length curves generator
│   ├── evaluate_step_response.py # 3D step response analysis and plotting
│   ├── export_onnx.py         # ONNX export and validation
│   ├── quantize_policy.py     # Q8.8 fixed-point quantization and C-header exporter
│   ├── generate_media_and_telemetry.py # Offscreen renderer for MP4, GIF, and CSV telemetry
│   ├── plot_comparison.py     # Bar chart and 3D trajectory plot generator
│   └── build_report.sh        # One-command Pandoc report compiler (PDF + DOCX)
├── requirements.txt           # Python package dependencies
└── README.md


```

---

## 🚀 Quickstart

### 1. Setup Virtual Environment & Install Dependencies

```bash
# Navigate to project root
cd /home/karol/ITB/kuliah/uav-controller

# Create a virtual environment using Python 3.10+
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Environment & Benchmark Physics Speed
```bash
python scripts/test_env.py
```

### 3. Train Baseline PPO Policy (Multi-Core CPU)
```bash
# Train for 1,500,000 steps across 8 parallel environments (~9 minutes on Intel Core Ultra 7)
python scripts/train_ppo.py --n-envs 8 --total-timesteps 1500000
```
Monitor real-time training progress via TensorBoard:
```bash
tensorboard --logdir tb_logs/ppo_quadrotor
```

### 4. Plot Training Learning Curves
```bash
python scripts/plot_training.py
```
This parses the monitor logs and saves a dual-panel learning curve figure to `figures/training_curve.png`.

### 5. Quantitative Benchmark
```bash
# Benchmark nominal conditions
python scripts/benchmark_policy.py --model-path checkpoints/best_model/best_model.zip --episodes 20

# Benchmark under wind disturbances & mass perturbations
python scripts/benchmark_policy.py --model-path checkpoints/best_model/best_model.zip --episodes 20 --randomize-domain
```

### 6. Interactive 3D Viewer Visualization
```bash
python scripts/eval_policy.py --model-path checkpoints/best_model/best_model.zip
```
*(Left-click + drag to rotate camera, Right-click to zoom).*

