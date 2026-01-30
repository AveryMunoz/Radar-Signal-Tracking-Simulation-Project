import numpy as np
import matplotlib.pyplot as plt
import time
import random

from RealPositionSimulation import objectTrajectory
from RadarModel import RadarModel
from KalmanMath import KalmanMath
from Gating import Gate
from AssociateNN import NearestNeighborAssociate
from AssociatePDA import ProbabilisticDataAssociation

from Visualizer import (
    plot_single_frame,
    plot_trajectory_with_measurements,
    plot_innovation_history,
    plot_cov_trace_history,
    plot_innovation_components,
    plot_measurement_density,
    plot_gating_frame
)

import tracker_cpp   # C++ module

np.random.seed(42)
random.seed(42) # Necessary to allow both C++ and Python engine to use the exact same starting metrics

# =========================================================
# Toggle: Python vs C++ math engine
# =========================================================

USE_CPP = False    # Set False to run pure Python version (True allows for performance comparisons)

# Alias C++ functions for dual testing feature (Python vs C++ for performance)
if USE_CPP:
    kalman_predict = tracker_cpp.kalman_predict
    kalman_update = tracker_cpp.kalman_update
    gate_measurements = tracker_cpp.gate_measurements
    associate_nn = tracker_cpp.associate_nn
    associate_pda = tracker_cpp.associate_pda

# =========================================================
# 1. Load true trajectories
# =========================================================
trajectory = objectTrajectory
num_objects = trajectory.shape[0]
num_frames = trajectory.shape[1]

# =========================================================
# 2. Create radar model
# =========================================================
radar = RadarModel()

# =========================================================
# 3. Generate radar measurements for all frames
# =========================================================
all_frames = radar.simulate_all_frames(trajectory, mapSize=2500)

# =========================================================
# 4. Create one Kalman filter per object
# =========================================================
dt = 1.0

filters = [
    KalmanMath(dt, process_noise=1.0, measurement_noise=30)
    for _ in range(num_objects)
]

# Initialize each filter with the first true position
for i in range(num_objects):
    x0, y0 = trajectory[i, 0, :]
    x_init = np.array([[x0], [y0], [0.0], [0.0]])
    filters[i].x = x_init.copy()

# =========================================================
# 5. Create gating + association modules
# =========================================================

# Simply determines which association method we will use, predetermined by the user
ASSOCIATION_METHOD = "PDA"   # "NN" or "PDA"

if ASSOCIATION_METHOD == "NN":
    associator = NearestNeighborAssociate()
elif ASSOCIATION_METHOD == "PDA":
    R = np.array([[30**2, 0],
                  [0, 30**2]])
    associator = ProbabilisticDataAssociation(R)

# =========================================================
# 6. Tracking loop
# =========================================================
filtered_tracks = [[] for _ in range(num_objects)]
innovation_history = []
innovation_components = []
cov_trace_history = []
beta_history = []

PRINT_EVERY = 10
PLOT_GATING = False

# Time declarations to observe performance metrics (personal curiousity on how C++ performance compares, even though it's on a small scale)
frame_times = []
gating_times = []
association_times = []
update_times = []

predict_times_py = []
predict_times_cpp = []
update_times_py = []
update_times_cpp = []

for t in range(num_frames):

    frame_start = time.time()

    frame_measurements = all_frames[t].copy()

    print(f"\n================ Frame {t} ================")
    print(f"Raw measurements: {frame_measurements}")

    for i in range(num_objects):
        kf = filters[i]        # C++-backed filter (when USE_CPP)

        # -------------------------------------------------
        # PYTHON PREDICT TIMING (always measured)
        # -------------------------------------------------
        t_pred_py = time.time()
        kf.predict()
        predict_times_py.append(time.time() - t_pred_py)

        # -------------------------------------------------
        # 1. Predict
        # -------------------------------------------------
        gate = Gate(gate_threshold=80)

        if USE_CPP:
            # Convert to Fortran order for C++
            xF = np.asfortranarray(kf.x)
            PF = np.asfortranarray(kf.P)
            FF = np.asfortranarray(kf.F)
            QF = np.asfortranarray(kf.Q)

            # --- C++ PREDICT TIMING ---
            t_pred_cpp = time.time()
            kalman_predict(xF, PF, FF, QF)
            predict_times_cpp.append(time.time() - t_pred_cpp)

            # Copy results back into the Python filter
            kf.x[:] = xF
            kf.P[:] = PF

        else:
            # Python predict (main filter)
            kf.predict()

            # No C++ timing in Python-only mode
            predict_times_cpp.append(0.0)

        predicted_z = (kf.H @ kf.x).flatten()

        # --- PYTHON GATING ---
        gated_py, _ = gate.gate_measurement(predicted_z, frame_measurements)

        # -------------------------------------------------
        # 2. Gate
        # -------------------------------------------------
        t0 = time.time()

        if USE_CPP:
            indices = gate_measurements(
                frame_measurements,
                predicted_z,
                gate.gate_threshold
            )
            indices = np.array(indices, dtype=int)
            if len(indices) > 0:
                gated = frame_measurements[indices]
            else:
                gated = np.empty((0, frame_measurements.shape[1]))
        else:
            gated, _ = gate.gate_measurement(predicted_z, frame_measurements)

        gating_times.append(time.time() - t0)

        print(f"Obj {i} predicted z: {predicted_z}")
        print(f"Obj {i} gated measurements: {gated}")

        if PLOT_GATING:
            plot_gating_frame(predicted_z, frame_measurements, gated, R, t)

        # -------------------------------------------------
        # 3. Associate
        # -------------------------------------------------
        t0 = time.time()

        z_bar = None
        info = None

        if USE_CPP:
            if ASSOCIATION_METHOD == "NN":
                idx = associate_nn(gated, predicted_z, kf.S)
                z_bar = gated[idx] if idx != -1 else None
                info = None

            elif ASSOCIATION_METHOD == "PDA":
                result = associate_pda(
                    gated,
                    predicted_z,
                    kf.R,
                    gate.gate_threshold,
                    radar.lambda_clutter
                )
                z_bar = result.z_fused
                info = result.betas
        else:
            z_bar, info = associator.choose(predicted_z, gated)

        association_times.append(time.time() - t0)

        if ASSOCIATION_METHOD == "NN":
            print(f"Obj {i} NN chose: {z_bar}")
        else:
            print(f"Obj {i} PDA betas: {info}")
            print(f"Obj {i} PDA fused z̄: {z_bar}")
            if info is not None:
                beta_history.append(info)

        # -------------------------------------------------
        # 4. Update
        # -------------------------------------------------
        if z_bar is None or len(z_bar) == 0:
            # No update
            update_times.append(0.0)
            update_times_py.append(0.0)
            update_times_cpp.append(0.0)
        else:
            if USE_CPP:
                # Convert everything to Fortran order for C++
                zF = np.asfortranarray(np.asarray(z_bar).reshape(2, 1))
                HF = np.asfortranarray(kf.H)
                RF = np.asfortranarray(kf.R)
                xF = np.asfortranarray(kf.x)
                PF = np.asfortranarray(kf.P)

                # C++ UPDATE TIMING 
                t_up_cpp = time.time()
                kalman_update(xF, PF, zF, HF, RF)
                update_times_cpp.append(time.time() - t_up_cpp)

                # Copying results back into the Python filter
                kf.x[:] = xF
                kf.P[:] = PF

                # PYTHON UPDATE TIMING
                t_up_py = time.time()
                kf.update(z_bar)
                update_times_py.append(time.time() - t_up_py)

                # Compare C++ vs Python filter states
                dx = kf.x - kf.x
                dP = kf.P - kf.P
                print(
                    f"Frame {t} Obj {i} | "
                    f"||dx||={np.linalg.norm(dx):.6e} | "
                    f"||dP||_F={np.linalg.norm(dP):.6e}"
                )

            else:
                # PYTHON UPDATE TIMING
                t_up_py = time.time()
                kf.update(z_bar)
                update_times_py.append(time.time() - t_up_py)

                # No C++ update in Python-only mode
                update_times_cpp.append(0.0)

            # Total update time (pipeline metric)
            update_times.append(update_times_py[-1] + update_times_cpp[-1])

        # -------------------------------------------------
        # Innovation + diagnostics
        # -------------------------------------------------
        if z_bar is not None and len(z_bar) != 0:
            z_bar_vec = np.asarray(z_bar).reshape(2, 1)
            innovation = z_bar_vec - (kf.H @ kf.x)
            innovation_norm = np.linalg.norm(innovation)

            innovation_history.append(innovation_norm)
            innovation_components.append([innovation[0,0], innovation[1,0]])
            cov_trace_history.append(np.trace(kf.P))

            if t % PRINT_EVERY == 0:
                print(
                    f"Frame {t:3d} | Obj {i} | "
                    f"InnovNorm={innovation_norm:8.2f} | "
                    f"CovTrace={np.trace(kf.P):8.2f}"
                )

        # -------------------------------------------------
        # 5. Store filtered position
        # -------------------------------------------------
        filtered_tracks[i].append([kf.x[0,0], kf.x[1,0]])

    frame_times.append(time.time() - frame_start) 


# =========================================================
# 7. Final summary (Statistics and Performance Comparison Metrics)
# =========================================================
print("\n=== FINAL SUMMARY ===")
print(f"Mean Innovation Norm: {np.mean(innovation_history):.3f}")
print(f"Final Innovation Norm: {innovation_history[-1]:.3f}")
print(f"Mean Covariance Trace: {np.mean(cov_trace_history):.3f}")
print(f"Final Covariance Trace: {cov_trace_history[-1]:.3f}")

print("\n=== PERFORMANCE SUMMARY ===")
print(f"Total Runtime: {sum(frame_times):.3f} sec")
print(f"Avg Frame Time: {np.mean(frame_times)*1000:.3f} ms")
print(f"Max Frame Time: {np.max(frame_times)*1000:.3f} ms")
print(f"Min Frame Time: {np.min(frame_times)*1000:.3f} ms")

print("\n--- Breakdown ---")
print(f"Gating Avg: {np.mean(gating_times)*1000:.3f} ms")
print(f"Association Avg: {np.mean(association_times)*1000:.3f} ms")
print(f"Update Avg: {np.mean(update_times)*1000:.3f} ms")

print("\n=== ENGINE COMPARISON ===")
print(f"Predict PY Avg: {np.mean(predict_times_py)*1000:.3f} ms")
print(f"Predict CPP Avg: {np.mean(predict_times_cpp)*1000:.3f} ms")
print(f"Update PY Avg: {np.mean(update_times_py)*1000:.3f} ms")
print(f"Update CPP Avg: {np.mean(update_times_cpp)*1000:.3f} ms")

# =========================================================
# 8. Visualizations
# =========================================================
plot_innovation_history(innovation_history)
plot_cov_trace_history(cov_trace_history)
plot_innovation_components(innovation_components)

plot_measurement_density(all_frames)

# Truth vs filtered tracks
plt.figure(figsize=(8, 8))

for i in range(num_objects):
    plt.plot(
        trajectory[i, :, 0],
        trajectory[i, :, 1],
        label=f"Truth {i}"
    )

for i in range(num_objects):
    ft = np.array(filtered_tracks[i])
    plt.plot(
        ft[:, 0],
        ft[:, 1],
        '--',
        label=f"Filtered {i}"
    )

plt.legend()
plt.grid(True)
plt.title("Truth vs Kalman Filtered Tracks")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plot_trajectory_with_measurements(
    trajectory,
    all_frames
)