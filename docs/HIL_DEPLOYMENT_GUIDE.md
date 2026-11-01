# AMD Kria KR260 Hardware-in-the-Loop (HIL) Deployment Guide

This guide details how to execute Hardware-in-the-Loop (HIL) flight control between the **host laptop** (running MuJoCo physics) and the **AMD Kria KR260** Robotics Starter Kit (running Ubuntu 24.04 on the ARM Cortex-A53 PS and Zynq UltraScale+ FPGA PL).

---

## 🏗️ Architecture & Communication Overview

```
┌────────────────────────────────────────────────────────┐
│               HOST LAPTOP (Intel Core Ultra)           │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ MuJoCo 6-DoF Physics Simulation (500 Hz)         │  │
│  │  - Quadrotor Dynamics & Wind Gust Disturbances   │  │
│  │  - Inner Rate-PID (500 Hz) & Analytical Mixer    │  │
│  │  - Real-Time / Passive 3D Visual Viewer          │  │
│  └──────────────────────▲───────────────────────────┘  │
│                         │ ActionPacket (24 bytes)      │
│                         │ [magic, seq_id, 4 floats]    │
│  ┌──────────────────────┴───────────────────────────┐  │
│  │ HIL Simulation Server (`scripts/hil_sim_server.py`)│ │
│  └──────────────────────┬───────────────────────────┘  │
└─────────────────────────┼──────────────────────────────┘
                          │ Ethernet UDP Socket (< 0.5 ms RTT)
                          │ Default Ports: 5005 (State) / 5006 (Action)
┌─────────────────────────┼──────────────────────────────┐
│  ┌──────────────────────▼───────────────────────────┐  │
│  │ HIL Inference Client                             │  │
│  │  - Option A: Python ONNX (`kria_client.py`)      │  │
│  │  - Option B: Embedded C Q8.8 (`kria_client`)     │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │ Forward Pass: 1.38 us        │
│                                                        │
│   AMD KRIA KR260 (Ubuntu 24.04 on ARM Cortex-A53 PS)   │
└────────────────────────────────────────────────────────┘
```

### Packet Protocol Specification
- **StatePacket (`<II16f`, 72 bytes)**:
  - `magic = 0x55415653` ("UAVS")
  - `seq_id`: 32-bit unsigned sequence identifier
  - `obs[16]`: 16 floating-point body-frame observations
- **ActionPacket (`<II4f`, 24 bytes)**:
  - `magic = 0x55415641` ("UAVA")
  - `seq_id`: Echoed 32-bit sequence identifier (for RTT calculation)
  - `action[4]`: Commanded vertical thrust and desired angular rates $[T_z, \omega_x, \omega_y, \omega_z]$

---

## 💻 Stage 1: Local Loopback Testing (Laptop Only)

Before plugging into the Kria, verify the full pipeline locally:

```bash
# Terminal 1: Run the automated verification testbench (tests C Q8.8 client)
.venv/bin/python scripts/benchmark_hil.py --client c --duration 10.0

# Terminal 2: Test network latency stress test (2 ms delay + 5% packet loss)
.venv/bin/python scripts/benchmark_hil.py --client c --duration 10.0 --latency-ms 2.0 --drop-rate 0.05

# Terminal 3: Test Python ONNX Runtime client
.venv/bin/python scripts/benchmark_hil.py --client python --duration 10.0
```

---

## ⚡ Stage 2: Connecting the Physical AMD Kria KR260

### Step 1: Ethernet Network Configuration
Connect an RJ45 Ethernet patch cable directly between your laptop and the Kria KR260 Ethernet port.

#### On Host Laptop:
Assign a static IP to the Ethernet interface:
```bash
# Check your Ethernet interface name (e.g., eth0, enp0s31f6)
ip link

# Assign static IP 192.168.1.10
sudo ip addr add 192.168.1.10/24 dev eth0
sudo ip link set eth0 up
```

#### On AMD Kria KR260:
Assign static IP `192.168.1.11`:
```bash
sudo ip addr add 192.168.1.11/24 dev eth0
sudo ip link set eth0 up

# Verify connectivity from Kria
ping 192.168.1.10
```

---

### Step 2: Deploy Code to Kria KR260

From your laptop, copy the standalone `deployment/` directory to the Kria:
```bash
scp -r deployment/ ubuntu@192.168.1.11:~/uav_deployment/
```

SSH into the Kria:
```bash
ssh ubuntu@192.168.1.11
cd ~/uav_deployment
```

---

### Step 3: Run Inference on Kria KR260

You have two execution modes on the Kria:

#### Option A: Ultra-Fast Embedded C Client (Q8.8 Fixed-Point, Recommended)
Zero external dependencies, single-command compilation:
```bash
# Compile with native GCC -O3
gcc -O3 kria_client.c -o kria_client

# Launch the client (listens on 5005, sends to laptop 192.168.1.10:5006)
./kria_client -p 5005 -s 192.168.1.10 -r 5006
```

#### Option B: Python ONNX Runtime Client
```bash
# Install ONNX Runtime on Kria Ubuntu (one-time setup)
pip install onnxruntime numpy

# Run ONNX client
python3 kria_client.py --bind-ip 0.0.0.0 --client-port 5005 --server-ip 192.168.1.10 --server-port 5006
```

---

### Step 4: Launch Simulation Server on Laptop

On your laptop, start the simulation server:
```bash
# Headless benchmark mode
.venv/bin/python scripts/hil_sim_server.py --client-ip 192.168.1.11 --duration 30.0

# Or with interactive 3D visual viewer
.venv/bin/python scripts/hil_sim_server.py --client-ip 192.168.1.11 --duration 0 --render
```

---

## 📊 Performance Benchmarks Summary

Measured on local loopback and simulated Ethernet network:

| Metric | C Q8.8 Client (`kria_client`) | Python ONNX Client (`kria_client.py`) |
| :--- | :---: | :---: |
| **Inference Time** | **$1.38\,\mu\text{s}$** | $45\text{--}65\,\mu\text{s}$ |
| **Loopback Round-Trip Time (RTT)** | **$0.122\text{ ms}$** | $0.365\text{ ms}$ |
| **Control Loop Margin (@ 50 Hz, 20 ms)** | **$99.4\%$ idle** | $98.2\%$ idle |
| **Steady-State Hover Error** | **$5.26 \pm 0.17\text{ cm}$** | $5.33 \pm 0.21\text{ cm}$ |
| **Survival under 2 ms Latency + 5% Drops** | **$100\%$ (Zero crashes)** | $100\%$ (Zero crashes) |
| **Binary Memory Footprint** | **$37\text{ KB}$ executable** | Python runtime + $9.4\text{ KB}$ model |
