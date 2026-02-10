
import matplotlib.pyplot as plt
import numpy as np

trace_colors = ["#2e5774", "#592a00", "#1e731e", "#ffffff", "#652476"]

def plot_tracking_view(frame, max_range=1500):
    fig, ax = plt.subplots(figsize=(7, 7))

   # Combine all measurements into one big array (easier for real-time plotting later)
    meas_history = frame.get("measurement_history", [])
    
    if len(meas_history) > 0:
        all_meas = np.vstack([np.array(m) for m in meas_history if len(m) > 0])
        ax.scatter(all_meas[:, 0], all_meas[:, 1], s=12, c="red", alpha=0.8)

    # Extract tracking data
    filt = frame.get("filtered_positions", [])
    history = frame.get("track_history", [])
    truth = frame.get("truth_positions", []) 

    # Plot each object's data (filtered, truth, and all measurments recorded)
    for i in range(len(filt)):
        color = trace_colors[i % len(trace_colors)]

        # Truth measurements (true position of our object, marked by an X)
        if truth and len(truth) > i:
            t_pos = truth[i]
            ax.scatter(t_pos[0], t_pos[1], c=color, marker="x", s=80, linewidths=2)

        # Track history (allows us to keep track of where the object WAS, essentially printing its path)
        if history and len(history[i]) > 1:
            hist_arr = np.array(history[i])
            ax.plot(hist_arr[:, 0], hist_arr[:, 1], c=color, linewidth=2, alpha=0.7)

        # Filtered estimate (current position, as estimated thorugh kalman filtering)
        f_pos = filt[i]
        ax.scatter(f_pos[0], f_pos[1], c=color, s=60, edgecolors="black", linewidths=1)

    # Plotting it all
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Tracking View (Truth, Filtered Tracks, and Radar Returns)")

    return fig

def plot_truth_view(frame, max_range=1500):
    # For comparison purposes, this plot will show only the true positions of the object
    fig, ax = plt.subplots(figsize=(7, 7))

    truth = frame.get("truth_positions", [])

    # Similar to before, we cycle thorugh true positions and store them for plotting
    for i, pos in enumerate(truth):
        ax.scatter(pos[0], pos[1], c="green", marker="x", s=80, linewidths=2)

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Truth View (Ground Truth Only)")

    return fig

