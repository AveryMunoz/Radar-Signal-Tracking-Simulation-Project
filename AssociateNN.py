# Finds the closest measurment to the predicted measurment (the "Nearest Neighbor")

import numpy as np

class NearestNeighborAssociate:
    def __init__(self):
        pass
    
    def choose(self, predicted_z, gated_measurements):
        if gated_measurements is None or len(gated_measurements) == 0:
            return None, None # Once again, no gated measurements available so we return nothing
        
        meas_array = np.array(gated_measurements) # Convert stored measurments from Gating.py into array for vector math (imported as a list initially)
        
        diffs = meas_array - predicted_z 
        dists = np.linalg.norm(diffs, axis=1) # Computes distances of the passed measurments to search for NN
        
        idx = np.argmin(dists) # Finds the closest measurment
        z = meas_array[idx]
            
        return z, idx # Returns Nearest Neighbor as well as index (For debugging purposes)