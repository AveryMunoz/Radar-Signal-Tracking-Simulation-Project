# Probabilistic Data Association file (almost takes like the "average" of where the predicted track could/should be)

import numpy as np

class ProbabilisticDataAssociation:
    def __init__(self, R):
        self.R = R
        self.R_inv = np.linalg.inv(R)
        
    def choose(self, predicted_z, gated_measurements):
        if gated_measurements is None or len(gated_measurements) == 0:
            return None, None # No gated measurements available, return nothing
        
        meas_array = np.array(gated_measurements) # Convert stored measurments from Gating.py into array for vector math (imported as a list initially)
        diffs = meas_array - predicted_z
        
        d2 = np.einsum('ij,jk,ik->i', diffs, self.R_inv, diffs) # Mahalanobis distance calculation for each measurement
        
        likelihoods = np.exp(-0.5 * d2) # Compute likelihoods based on Mahalanobis distances (Gaussian approximation)    
        
        sum_L = np.sum(likelihoods)
        if sum_L == 0:
            betas = np.ones_like(likelihoods) / len(likelihoods) # Avoid division by zero, assign equal weights
        else:
            betas = likelihoods / sum_L # Normalize likelihoods to get association probabilities (Probability scaling)
        
        z_bar = np.sum(meas_array * betas[:, np.newaxis], axis=0)
        
        return z_bar, betas 
        
        