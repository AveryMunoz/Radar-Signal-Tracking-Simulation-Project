# Filters list of measurements so that only reasonable measurments remain and can be analyzed

import numpy as np

class Gate:
    def __init__(self, gate_threshold):
        self.gate_threshold = gate_threshold
        
    def gate_measurement(self, predicted_z, measurements):
        if measurements is None or len(measurements) == 0:
            return [], [] # If no measurements, then gate is empty
    
        diff = measurements - predicted_z 
        dists = np.linalg.norm(diff, axis=1) # Magnitude of the distances
        
        gated = []
        for meas, dist in zip(measurements, dists): 
            if dist <= self.gate_threshold:
                gated.append(meas) # Loops through each measurment, keeping only measurements in range

        return gated, dists.tolist() # Passed position measurements (measurements within reasonable range) and distances for all measurements