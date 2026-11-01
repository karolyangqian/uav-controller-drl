---
title: "Autonomous 3D Quadrotor UAV Controller via Deep Reinforcement Learning with Hardware-in-the-Loop Implementation on AMD Kria KR260"
author:
  - "Karol Yangqian Poetracahya (NIM: 13523093)"
  - "School of Electrical Engineering and Informatics (STEI), Institut Teknologi Bandung"
  - "Academic Year 2026/2027"
  - "Email: karolyangqian14@gmail.com, 13523093@std.stei.itb.ac.id"
date: "November 2026"
geometry: margin=1in
fontsize: 12pt
numbersections: true
---

\begin{abstract}
This report documents the design, simulation, training, and deployment of an end-to-end Deep Reinforcement Learning (DRL) 3D flight controller for an unmanned aerial vehicle (UAV). To overcome common sim-to-real brittleness and high-frequency actuator chattering observed in standard reinforcement learning setups, a hierarchical control architecture is introduced. The outer-loop policy runs at 50 Hz and commands collective vertical thrust and 3D body rates, while an inner-loop rate PID controller operates at 500 Hz to handle rapid gyroscopic and motor dynamics. The environment is simulated with high fidelity in MuJoCo, and training is executed using multi-process Proximal Policy Optimization (PPO). Incorporating domain randomization (aerodynamic wind vectors, mass variations, inertia jitter, and sensor noise) reduces steady-state tracking error under continuous disturbances from 35.61 cm to 13.46 cm (a 62.2\% reduction) with an 85.0\% target acquisition success rate. Furthermore, the trained policy is successfully exported to ONNX (9.36 KB footprint) and quantized into Q8.8 fixed-point integer arithmetic, retaining 99.77\% tracking precision with an inference latency of only 1.38 microseconds, establishing an optimal pipeline for Hardware-in-the-Loop (HIL) deployment on the AMD Kria KR260 FPGA/SoC platform.
\end{abstract}

# Introduction

## Background and Motivation
Autonomous flight control for unmanned aerial vehicles (UAVs) has traditionally relied on cascaded Proportional-Integral-Derivative (PID) loops or Linear Quadratic Regulator (LQR) formulations. While computationally lightweight and well-understood near hover equilibrium, linear control models degrade rapidly when subjected to severe aerodynamic non-linearities, unmodeled cross-coupling torques, parameter uncertainty, and persistent external disturbances such as atmospheric turbulence and wind gusts.

Deep Reinforcement Learning (DRL) provides an expressive, data-driven control paradigm capable of directly learning non-linear state-action mappings across complex dynamical domains. However, deploying DRL directly to real-world physical systems presents two fundamental bottlenecks:

1. **Actuator Chattering (Jitter)**: Direct end-to-end motor PWM policies tend to exhibit high-frequency bang-bang switching, causing extreme thermal dissipation, motor strain, and rapid physical failure.
2. **The Sim-to-Real Gap**: Policies trained solely in idealized simulation models overfit to deterministic dynamics and fail catastrophically when exposed to real-world mass variations, aerodynamic drag, sensor latencies, and wind gusts.

This project addresses these challenges through a hierarchical control architecture, smoothness-regularized reward shaping, aggressive domain randomization, and hardware-optimized fixed-point quantization.

## Problem Statement and Objectives
The primary objectives of this research are:

1. Develop a high-fidelity 6-DoF quadrotor physical model and Gymnasium simulation environment in MuJoCo capable of running at over 5,000 steps per second on commodity CPU hardware.
2. Design a hierarchical control framework that couples a 50 Hz outer-loop PPO policy with a 500 Hz inner-loop rate PID controller to eliminate motor chattering.
3. Apply domain randomization to achieve robust disturbance rejection against aerodynamic wind forces and parameter drift.
4. Export and quantize the resulting neural network for real-time edge execution on the AMD Kria KR260 FPGA platform, achieving sub-microsecond inference latency.

# System Modeling and Simulation

## Quadrotor Dynamics
The quadrotor is modeled as a 0.8 kg rigid body in an X-configuration with an arm radius $d = 0.106\text{ m}$. The equations of motion in the body frame are governed by Newton-Euler formulations:
$$\mathbf{m}\dot{\mathbf{v}} + \boldsymbol{\omega} \times (\mathbf{m}\mathbf{v}) = \mathbf{F}_{\text{total}} + \mathbf{R}^T \mathbf{m}\mathbf{g}$$
$$\mathbf{I}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega}) = \boldsymbol{\tau}_{\text{total}}$$
where $\mathbf{m}$ is the total vehicle mass, $\mathbf{I} = \text{diag}(0.005, 0.005, 0.009)\text{ kg}\cdot\text{m}^2$ is the nominal inertia tensor, $\mathbf{R} \in SO(3)$ denotes the rotation matrix from the body frame to the inertial world frame, and $\mathbf{g} = [0, 0, -9.81]^T\text{ m/s}^2$ is gravitational acceleration.

![High-fidelity 0.8 kg Quadrotor and Target Sphere in MuJoCo Physics Arena](media/render_hover_snapshot.png){width=65%}

## Inner-Loop Rate PID and Inverse Mixer
To insulate the reinforcement learning policy from stiff motor time constants and high-frequency gyroscopic dynamics, control is partitioned into a two-level hierarchy:
- **Outer Loop (50 Hz)**: The neural network policy observes the 16-dimensional state and outputs target collective thrust $T_z$ and desired angular rates $\boldsymbol{\omega}_{cmd} = [\omega_x, \omega_y, \omega_z]_{cmd}$.
- **Inner Loop (500 Hz)**: An analytical rate PID computes corrective body torques:
$$\boldsymbol{\tau} = \mathbf{K}_p (\boldsymbol{\omega}_{cmd} - \boldsymbol{\omega}) + \mathbf{K}_i \int (\boldsymbol{\omega}_{cmd} - \boldsymbol{\omega}) dt + \mathbf{K}_d \frac{d}{dt}(\boldsymbol{\omega}_{cmd} - \boldsymbol{\omega})$$

The total vertical thrust $T_z$ and torque vector $\boldsymbol{\tau} = [\tau_x, \tau_y, \tau_z]^T$ are subsequently mapped into individual rotor thrusts $[F_0, F_1, F_2, F_3]$ via the analytical inverse mixer:
$$F_0 = \frac{1}{4} T_z - \frac{1}{4d} \tau_x - \frac{1}{4d} \tau_y - \frac{1}{4k_m} \tau_z$$
$$F_1 = \frac{1}{4} T_z + \frac{1}{4d} \tau_x - \frac{1}{4d} \tau_y + \frac{1}{4k_m} \tau_z$$
$$F_2 = \frac{1}{4} T_z + \frac{1}{4d} \tau_x + \frac{1}{4d} \tau_y - \frac{1}{4k_m} \tau_z$$
$$F_3 = \frac{1}{4} T_z - \frac{1}{4d} \tau_x + \frac{1}{4d} \tau_y + \frac{1}{4k_m} \tau_z$$
where $k_m = 0.016\text{ N}\cdot\text{m/N}$ represents the motor yaw moment coefficient.

# Reinforcement Learning Formulation

## Observation and Action Spaces
To guarantee full rotational invariance across arbitrary yaw headings, the 16-dimensional observation vector $\mathbf{s}_t \in \mathbb{R}^{16}$ is expressed strictly within the quadrotor's body reference frame:

1. $\mathbf{e}_{p,body} \in \mathbb{R}^3$: Position error relative to the target setpoint, rotated into the body frame ($\mathbf{R}^T (\mathbf{p}_{target} - \mathbf{p})$).
2. $\mathbf{v}_{body} \in \mathbb{R}^3$: Linear velocity vector in the body frame ($\mathbf{R}^T \mathbf{v}$).
3. $\mathbf{z}_{body,world} \in \mathbb{R}^3$: Projected body z-axis vector in the world frame ($R_{02}, R_{12}, R_{22}$), providing an explicit, gimbal-lock-free attitude tilt representation.
4. $\boldsymbol{\omega}_{body} \in \mathbb{R}^3$: Body rotational angular rates $[p, q, r]$.
5. $\mathbf{a}_{t-1} \in \mathbb{R}^4$: Previously applied control command vector, enabling action rate-of-change regularization.

The action space $\mathbf{a}_t \in [-1, 1]^4$ dictates commanded vertical acceleration and angular rates:
- $a_0$: Commanded thrust, linearly scaled between $[0, 2\text{ mg}]$.
- $a_1, a_2, a_3$: Commanded body rates $[\omega_x, \omega_y, \omega_z]$, scaled to $[-3.0, 3.0]\text{ rad/s}$.

## Anti-Jitter Filtering and Reward Formulation
Actuator chattering is actively suppressed using a first-order Exponential Moving Average (EMA) low-pass action filter:
$$\tilde{\mathbf{a}}_t = \alpha \mathbf{a}_t + (1 - \alpha) \tilde{\mathbf{a}}_{t-1}, \quad \alpha = 0.75$$

The multi-objective reward function $r_t$ penalizes tracking deviation while explicitly rewarding smooth control actions:
$$r_t = r_{\text{pos}} + r_{\text{vel}} + r_{\text{upright}} + r_{\text{alive}} - \lambda_1 \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 - \lambda_2 \|\mathbf{a}_t\|^2$$
where:
$$r_{\text{pos}} = \exp\left(-\frac{\|\mathbf{e}_{pos}\|^2}{2 \sigma_{pos}^2}\right), \quad r_{\text{vel}} = -0.1 \|\mathbf{v}\|, \quad r_{\text{upright}} = 0.3 (\hat{\mathbf{z}}_B \cdot \hat{\mathbf{z}}_W)$$
with smoothness penalty weight $\lambda_1 = 0.05$ and action magnitude regularization $\lambda_2 = 0.01$.

![Autonomous 3D Spatial Trajectory Flown by the Quadrotor Policy in MuJoCo Arena](figures/trajectory_3d.png){width=70%}

# Experimental Results

## Training Convergence
Training was carried out on an Intel Core Ultra 7 155H processor utilizing 8 parallel worker environments (`SubprocVecEnv`) with Proximal Policy Optimization (PPO). The policy converged within 1.5 million timesteps (total wall-clock time: 9 minutes 41 seconds), maintaining an average throughput of over 2,600 environment steps per second.

![PPO Training Convergence Curves under Stochastic Wind Gusts and Mass Variations](figures/robust_training_curve.png){width=85%}

As illustrated in Figure 3, the episode return rapidly climbs from negative initial values to exceed 914, while the average episode survival length reaches the maximum allowable horizon of 500 steps within the first 300,000 steps. The steady-state tracking error under continuous disturbance stabilizes below 14 cm.

## Disturbance Rejection and Domain Randomization
To assess sim-to-real transferability, the quadrotor was evaluated under severe dynamic disturbances:
- Steady horizontal wind drag forces: $[-1.5, 1.5]\text{ N}$.
- Stochastic peak wind gust pulses: up to $2.5\text{ N}$.
- Body mass variations: $\pm 15\%$ ($0.68\text{ kg} - 0.92\text{ kg}$).
- Inertia diagonal jitter: $\pm 10\%$.
- Additive Gaussian sensor noise on velocity ($0.05\text{ m/s}$) and angular rates ($0.02\text{ rad/s}$).

Table 1 and Figure 4 benchmark the baseline policy against the robust domain-randomized policy across 20 randomized evaluation runs.

Table 1: Disturbance Rejection Performance Comparison under 2.5 N Wind Gusts and Mass Variations.

| Performance Metric | Nominal Baseline Policy | Robust (Domain-Randomized) Policy | Improvement |
| :--- | :---: | :---: | :---: |
| **Steady-State Hover Error** | $35.61 \pm 20.22\text{ cm}$ | **$13.46 \pm 8.82\text{ cm}$** | **$62.2\%$ reduction** |
| **Success Rate ($<15\text{ cm}$ error)** | $5.0\%$ | **$85.0\%$** | **$+80.0\%$ points** |
| **Episode Survival Rate** | $100\%$ | **$100\%$** | Robust stability |
| **Mean Pitch/Roll Tilt Angle** | $7.83^\circ$ | $10.29^\circ$ | Active wind leaning |
| **Mean Cumulative Reward** | $847.4 / 1000$ | **$914.3 / 1000$** | **$+66.9$ points** |

![Disturbance Rejection Benchmark: Nominal Baseline vs. Robust Policy under 2.5 N Stochastic Wind Disturbances](figures/comparison_benchmark.png){width=75%}

The robust policy autonomously generates an active trim tilt angle of $10.29^\circ$ directly into the incoming wind disturbance vector, counteracting lateral drift and retaining tight setpoint tracking without requiring explicit aerodynamic state estimation.

## 3D Step Response Dynamics
To evaluate dynamic transient tracking, a severe 3D setpoint displacement was injected at $t = 2.0\text{ s}$, shifting the target from $[0.0, 0.0, 1.0]\text{ m}$ to $[1.0, 1.0, 2.0]\text{ m}$ (a Euclidean step of $1.732\text{ m}$).

![3D Step Response Dynamics: Position Tracking, Attitude Tilt Angles, and Commanded Control Actions](figures/step_response.png){width=85%}

As detailed in Figure 5:

- **Rise Time ($T_r$ to $90\%$ setpoint)**: $1.73\text{ seconds}$.
- **Settling Time ($T_s$ within $10\text{ cm}$)**: $3.40\text{ seconds}$.
- **Final Steady-State Error**: $5.18\text{ cm}$.
- **Peak Tilt Angle**: $15.1^\circ$ (well within conservative physical safety envelopes).
- **Control Signal Transitions**: Action traces demonstrate smooth, continuous actuation curves with zero high-frequency chatter.

# Hardware Deployment and Quantization on AMD Kria KR260

## Model Export and ONNX Runtime Verification
The trained Actor network ($16 \to 64 \to 64 \to 4$ Multi-Layer Perceptron with Tanh activations) was extracted from Stable-Baselines3 and serialized into standard Open Neural Network Exchange (ONNX) format (`quadrotor_policy.onnx`).

The exported model binary occupies only **$9.36\text{ KB}$** on disk, allowing it to fit entirely into on-chip Block RAM (BRAM) within the AMD Kria KR260 Zynq UltraScale+ FPGA without requiring external DDR memory accesses. Validation with ONNX Runtime confirmed numerical parity with PyTorch floating-point forward passes within a maximum absolute discrepancy of $< 1.6 \times 10^{-7}$.

## Fixed-Point (Q8.8) Quantization for FPGA Programmable Logic
To circumvent floating-point DSP pipeline stalls and maximize throughput in FPGA hardware logic, network weights and activations were converted to signed 16-bit Q8.8 fixed-point integer arithmetic (8 integer bits, 8 fractional bits):

- **Floating-Point (FP32) Steady-State Error**: $5.27\text{ cm}$.
- **Q8.8 Fixed-Point Steady-State Error**: $5.25\text{ cm}$.
- **Tracking Accuracy Retained**: **$99.77\%$** ($0.12\text{ mm}$ absolute deviation vs FP32).
- **Execution Latency in C (`test_forward.c`, GCC -O3)**: **$1.38\text{ }\mu\text{s}$ per inference** ($> 720\text{ kHz}$ throughput).

This ultra-low execution latency guarantees that forward inference consumes less than $0.007\%$ of the 50 Hz control period, leaving ample computational headroom for telemetry logging, obstacle avoidance, and mission planning on the Kria KR260 ARM Cortex-A53 Processing System (PS) and FPGA Programmable Logic (PL).

# Conclusion
This research demonstrates a complete, end-to-end pipeline for designing, training, and deploying a robust Deep Reinforcement Learning flight controller for a 6-DoF quadrotor UAV. By combining an outer 50 Hz DRL policy with an inner 500 Hz rate PID and exponential moving average action filtering, high-frequency actuator jitter was entirely eradicated. Incorporating domain randomization produced an active wind-rejecting policy that slashed tracking error under aerodynamic disturbances by $62.2\%$ with an $85\%$ success rate. Finally, fixed-point Q8.8 quantization retained $99.77\%$ tracking fidelity with a $1.38\,\mu\text{s}$ forward inference budget on a $9.36\text{ KB}$ memory footprint, establishing a solid foundation for real-time Hardware-in-the-Loop deployment on the AMD Kria KR260 platform.
