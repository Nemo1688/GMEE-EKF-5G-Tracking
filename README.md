# GMEE-EKF-5G-Tracking
MATLAB implementation of GMEE-EKF for robust 5G channel estimation and tracking.
# GMEE-EKF-5G-Tracking

**# GMEE-EKF-5G-Tracking** — A MATLAB implementation of the **Generalized Minimum Error Entropy Extended Kalman Filter (GMEE-EKF)**, applied to dynamic 5G channel estimation and tracking.  
This repository demonstrates the robustness and accuracy of GMEE-EKF compared with traditional Least Squares (LS) and MEE-based filters under non-Gaussian noise environments.

---

##  Overview

This project focuses on **robust state estimation** using information-theoretic learning (ITL) criteria.  
The GMEE-EKF combines the entropy-based error measurement of MEE with the flexibility of the **Generalized Gaussian Distribution (GGD)** to achieve higher stability and lower mean-square error under impulsive or mixed Gaussian noise.

The simulation includes:
- Time-varying Rayleigh fading channel modeling (Jakes/Clarke model)
- GMEE-EKF vs LS comparison across 100 Monte Carlo trials
- Performance visualization: time-domain tracking, MSE plots, CDFs, and histograms
- Configurable parameters for SNR, Doppler shift, multipath, and noise models

---

##  Research Context

This implementation extends the concept of **Minimum Error Entropy (MEE)** filtering to a generalized framework, enabling enhanced adaptability in highly dynamic and non-Gaussian environments such as **5G wireless communications**.  
The methodology can also be applied to GNSS signal tracking, sensor fusion, and non-linear system estimation problems.

---

##  Repository Structure
| Parameter | Description                    | Example  |
| --------- | ------------------------------ | -------- |
| `fc`      | Carrier frequency              | 28 GHz   |
| `Npath`   | Number of multipath components | 5        |
| `Nsym`    | Number of OFDM symbols         | 100      |
| `SNR`     | Signal-to-noise ratio (dB)     | 10–30 dB |
| `fd`      | Doppler frequency              | 200 Hz   |
| `N_monte` | Monte Carlo runs               | 100      |
