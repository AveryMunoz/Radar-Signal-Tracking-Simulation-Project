# Visualizer file to help see project growth and outputs (also helps with debugging)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# =========================================================
# 1. Single Frame Visualization
# =========================================================
def plot_single_frame(true_positions, measurement, frame_index=None):
    plt.figure(figsize=(8, 8))
    
    if true_positions is not None:
        plt.scatter(true_positions[:, 0], true_positions[:, 1], 
                    c='blue', label='True Positions', s=50)
    
    if measurement is not None:
        plt.scatter(measurement[:, 0], measurement[:, 1], 
                    c='red', label='Radar Measurements', s=20, alpha=0.7)
        
    title = "Radar Frame"
    if frame_index is not None:
        title += f" (t = {frame_index})"
    plt.title(title)
    
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")    
    plt.legend()
    plt.grid(True)  
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# =========================================================
# 2. Trajectory vs All Measurements
# =========================================================
def plot_trajectory_with_measurements(trajectory, all_frames):
    num_objects = trajectory.shape[0]

    # Collect all measurements once
    meas_x = []
    meas_y = []
    for frame in all_frames:
        if len(frame) > 0:
            meas_x.extend(frame[:, 0])
            meas_y.extend(frame[:, 1])

    plt.figure(figsize=(8, 8))

    # Colors that contrast well with red clutter
    colors = ["#00B7EB", "#32CD32", "#1E3A8A"]  # cyan, lime, deep blue

    # Plot each object's trajectory with its custom color
    for obj in range(num_objects):
        true_x = trajectory[obj, :, 0]
        true_y = trajectory[obj, :, 1]
        plt.plot(
            true_x,
            true_y,
            color=colors[obj],
            linewidth=2,
            label=f"Object {obj} Trajectory"
        )

    # Plot all measurements
    plt.scatter(meas_x, meas_y, c='red', s=10, alpha=0.5, label='Radar Measurements')

    plt.title("True Trajectories vs Radar Measurements")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# =========================================================
# 3. Innovation Norm Over Time
# =========================================================
def plot_innovation_history(innovation_history):
    plt.figure(figsize=(8,4))
    plt.plot(innovation_history, label="Innovation Norm")
    plt.title("Innovation Norm Over Time")
    plt.xlabel("Frame")
    plt.ylabel("||Innovation||")
    plt.grid(True)
    plt.legend()
    plt.show()


# =========================================================
# 4. Covariance Trace Over Time
# =========================================================
def plot_cov_trace_history(cov_trace_history):
    plt.figure(figsize=(8,4))
    plt.plot(cov_trace_history, label="Covariance Trace")
    plt.title("Covariance Trace Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Trace(P)")
    plt.grid(True)
    plt.legend()
    plt.show()


# =========================================================
# 5. Innovation Components (X and Y separately)
# =========================================================
def plot_innovation_components(innovation_components):
    innovation_components = np.array(innovation_components)
    plt.figure(figsize=(8,4))
    plt.plot(innovation_components[:,0], label="Innovation X")
    plt.plot(innovation_components[:,1], label="Innovation Y")
    plt.title("Innovation Components Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Innovation Component")
    plt.grid(True)
    plt.legend()
    plt.show()


# # =========================================================
# # 6. PDA Betas Over Time
# # =========================================================
# def plot_beta_history(beta_history):
#     plt.figure(figsize=(8,4))
#     for i, betas in enumerate(beta_history):
#         plt.plot(betas, alpha=0.5)
#     plt.title("PDA Betas Over Time")
#     plt.xlabel("Measurement Index")
#     plt.ylabel("Beta Value")
#     plt.grid(True)
#     plt.show()


# =========================================================
# 7. Measurement Density Heatmap
# =========================================================
def plot_measurement_density(all_frames, bins=50):
    all_points = np.vstack(all_frames)
    plt.figure(figsize=(6,6))
    plt.hist2d(all_points[:,0], all_points[:,1], bins=bins, cmap='hot')
    plt.colorbar(label="Density")
    plt.title("Measurement Density Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# =========================================================
# 8. Gating Visualization (per frame)
# =========================================================
def plot_gating_frame(predicted_z, raw_meas, gated_meas, R, frame_index):
    plt.figure(figsize=(6,6))

    # Raw measurements
    if len(raw_meas) > 0:
        plt.scatter(raw_meas[:,0], raw_meas[:,1], c='gray', label="Raw Measurements")

    # Gated measurements
    if len(gated_meas) > 0:
        plt.scatter(gated_meas[:,0], gated_meas[:,1], c='green', label="Gated Measurements")

    # Predicted measurement
    plt.scatter(predicted_z[0], predicted_z[1], c='red', marker='x', s=100, label="Predicted z")

    # Gate ellipse (3-sigma)
    vals, vecs = np.linalg.eigh(R)
    width, height = 3*np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    ellipse = Ellipse(predicted_z, width, height, angle, fill=False, edgecolor='blue', linewidth=2)
    plt.gca().add_patch(ellipse)

    plt.title(f"Gating Visualization — Frame {frame_index}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
