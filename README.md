# Radar Signal Tracking Project  
A hybrid Python + C++ radar tracking simulation featuring a constant‑velocity Kalman filter, gating, nearest‑neighbor and PDA association, and a full performance comparison between Python and C++ engines. This project 
simulates 3 objects with random initial positions and random constant velocities, simultaneously capturing measurments of each using state estimation due to the addition of clutter, uncertainty, and overall simulation of "noisy" 
data measurements (an attempt at mimicing "realistic" radar tracking). Parameters and metrics can be altered within the code such as the amount of measurements per frame, spread of measurments, etc.

---

## 🚀 Features

### 🧠 Tracking Algorithms
- Constant‑Velocity Kalman Filter (predict + update)
- Mahalanobis distance gating
- Nearest‑Neighbor (NN) association
- Probabilistic Data Association (PDA)
- Full simulation of clutter, noise, and target motion 

### ⚡ Hybrid Python/C++ Engine
- Python implementation for clarity and debugging
- C++ implementation for speed and performance (personal interest in performance comparisons and an excuse to use C++)

### 📊 Performance Benchmarking
The simulation measures:
- Python vs C++ predict time
- Python vs C++ update time
- Gating time
- Association time
- Per‑frame total time
- Total runtime

All timing is collected automatically and summarized at the end of the run.

### 🎨 Visualization & UI (Not implmented yet, but planning to soon add)
- Streamlit dashboard for interactive playback (Coming soon)

- Plots for:
  - True trajectory
  - Estimated trajectory
  - Clutter points
  - Gating ellipses
  - Innovation history
  - Covariance trace
  - Heat density map of clutter

---

