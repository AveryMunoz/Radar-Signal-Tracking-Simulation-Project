# Main application (the frontend work)

import streamlit as st
import requests
import matplotlib.pyplot as plt
from AppVisualizer import plot_tracking_view, plot_truth_view
import time

Backend_URL = "http://127.0.0.1:8000"

st.title("Welcome to the Radar Tracking Simulator Application!")

st.header("What does this simulation offer?")
st.text(
    "This simulation allows the user to manipulate different parameters in radar tracking "
    "and state estimation. You can change the number of objects, measurement noise, clutter, "
    "gating thresholds, and more. It simulates realistic radar tracking using Kalman filtering "
    "and association methods."
)

# Sidebar controls for user experimenting with uncertainty parameters
st.sidebar.header("Simulation Settings")

num_objects = st.sidebar.selectbox("Number of objects you'd like to track:", [1, 2, 3, 4, 5])
association_method = st.sidebar.radio(
    "Association Method",
    ["NN", "PDA"],
    help="NN = Nearest Neighbor. PDA = Probabilistic Data Association."
)

sigma_base = st.sidebar.slider("Sigma Base (Measurement Noise)", 1, 75, 45)
range_ref = st.sidebar.slider("Range Reference (Noise Growth)", 2500, 7500, 5000)
lambda_clutter = st.sidebar.slider("Lambda Clutter (False Alarms)", 5, 100, 25)
gate_threshold = st.sidebar.slider("Gate Threshold", 15, 50, 40)
sim_speed = st.sidebar.selectbox("Select the speed of the simulation:", ["Normal", "Fast", "Super Sim"])

# Session State 
if "running" not in st.session_state:
    st.session_state.running = False

# Everything that will be passed into our backend files for calculations and frame output
config = {
    "num_objects": num_objects,
    "association_method": association_method,
    "sigma_base": sigma_base,
    "range_ref": range_ref,
    "lambda_clutter": lambda_clutter,
    "gate_threshold": gate_threshold,
    "process_noise": 1.0,
    "measurement_noise": 30.0,
    "max_range": 10000.0
}

# Includes start and reset buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Simulation"):
        # Send config
        resp = requests.post(f"{Backend_URL}/configure", json=config)
        try:
            config_response = resp.json()
        except Exception:
            st.error("Backend /configure did not return JSON.")
            st.code(resp.text)
            st.stop()

        # Reset engine
        resp = requests.get(f"{Backend_URL}/reset")
        try:
            reset_response = resp.json()
        except Exception:
            st.error("Backend /reset did not return JSON.")
            st.code(resp.text)
            st.stop()

        if "error" in reset_response:
            st.error(reset_response["error"])
            st.stop()

        st.session_state.running = True


with col2:
    if st.button("Reset Simulation"):
        st.session_state.running = False

        resp = requests.get(f"{Backend_URL}/reset")
        try:
            reset_response = resp.json()
        except Exception:
            st.error("Backend /reset did not return JSON.")
            st.code(resp.text)
            st.stop()

        st.success("Simulation reset! Ready to start again.")


# Simulation loop, meant to mimic "real-time" tracking of an object
if st.session_state.running:
    placeholder = st.empty()

    while st.session_state.running:

        # Get next frame
        resp = requests.get(f"{Backend_URL}/step")
        try:
            frame = resp.json()
        except Exception:
            st.error("Backend /step did not return JSON.")
            st.code(resp.text)
            st.stop()

        if "error" in frame:
            st.error(frame["error"])
            st.session_state.running = False
            break

        # Two side-by-side plots
        colA, colB = placeholder.columns(2)

        with colA:
            fig1 = plot_tracking_view(frame)
            colA.pyplot(fig1)
            plt.close(fig1)

        with colB:
            fig2 = plot_truth_view(frame)
            colB.pyplot(fig2)
            plt.close(fig2)

        # Speed control
        if sim_speed == "Normal":
            time.sleep(0.02)
        elif sim_speed == "Fast":
            time.sleep(0.005)
        else:
            time.sleep(0.0001)
