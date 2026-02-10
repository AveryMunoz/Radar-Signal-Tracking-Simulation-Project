import numpy as np
from RealPositionSimulation import objectTrajectory
from RadarModel import RadarModel
from KalmanMath import KalmanMath
from Gating import Gate
from AssociateNN import NearestNeighborAssociate
from AssociatePDA import ProbabilisticDataAssociation

class RealtrackerEngine():
    def __init__(self, config):
        self.config = config
        self.reset(config)
    
    def reset(self, config):
        self.config = config

        # Load trajectory 
        self.trajectory = objectTrajectory[:config["num_objects"]]
        self.num_objects = config["num_objects"]
        self.num_frames = self.trajectory.shape[1]

        # Radar model
        self.radar = RadarModel(
            max_range=config["max_range"],
            sigma_base=config["sigma_base"],
            range_ref=config["range_ref"],
            lambda_clutter=config["lambda_clutter"]
        )

        # Generate all radar frames
        self.all_frames = self.radar.simulate_all_frames(
            self.trajectory,
            mapSize=2500
        )

        # Create Kalman filters
        self.filters = [
            KalmanMath(
                dt=1.0,
                process_noise=config["process_noise"],
                measurement_noise=config["measurement_noise"]
            )
            for _ in range(self.num_objects)
        ]

        # Initialize each filter with the first true position
        for i in range(self.num_objects):
            x0, y0 = self.trajectory[i, 0, :]
            self.filters[i].x = np.array([[x0], [y0], [0.0], [0.0]])

        # Gating
        self.gate = Gate(gate_threshold=config["gate_threshold"])

        # Association
        if config["association_method"] == "NN":
            self.associator = NearestNeighborAssociate()
        else:
            R = np.array([
                [config["measurement_noise"]**2, 0],
                [0, config["measurement_noise"]**2]
            ])
            self.associator = ProbabilisticDataAssociation(R)

        # Storage for filtered tracks
        self.filtered_tracks = [[] for _ in range(self.num_objects)]

        # storage for ALL measurements across all frames
        self.measurement_history = []

        # Reset frame counter
        self.current_frame = 0

    
    def step(self):
        t = self.current_frame
        frame_measurements = self.all_frames[t]

        # append this frame's measurements to history
        self.measurement_history.append(frame_measurements.tolist())

        # Prepare output container
        frame_output = {
            "frame_index": t,
            "measurements": frame_measurements.tolist(),
            "predicted_positions": [],
            "filtered_positions": []
        }

        # Loop over each object
        for i in range(self.num_objects):
            kf = self.filters[i]

            # Prediction step
            kf.predict()
            predicted_z = (kf.H @ kf.x).flatten()
            frame_output["predicted_positions"].append(predicted_z.tolist())

            # Gating step
            gated, _ = self.gate.gate_measurement(predicted_z, frame_measurements)

            # Association step
            z_bar, info = self.associator.choose(predicted_z, gated)

            # Update step
            if z_bar is not None and len(z_bar) != 0:
                kf.update(z_bar)

            # Storing the filtered positions
            filtered_pos = [kf.x[0,0], kf.x[1,0]]
            self.filtered_tracks[i].append(filtered_pos)
            frame_output["filtered_positions"].append(filtered_pos)

        # Next time step
        self.current_frame += 1

        # --- JSON‑SAFE CONVERSION ---
        safe_output = {
            "frame_index": int(t),
            "measurements": frame_measurements.tolist(),

            "predicted_positions": [
                [float(v) for v in pos] for pos in frame_output["predicted_positions"]
            ],

            "filtered_positions": [
                [float(v) for v in pos] for pos in frame_output["filtered_positions"]
            ],

            "track_history": [
                [[float(v) for v in pos] for pos in track]
                for track in self.filtered_tracks
            ],

            "measurement_history": self.measurement_history,

            # truth positions for this frame
            "truth_positions": [
                [float(self.trajectory[i, t, 0]), float(self.trajectory[i, t, 1])]
                for i in range(self.num_objects)
            ]
        }
        
        return safe_output