# This file essentially runs through the Kalman filtering equations, allowing for state estimation of the objects were trying to track

import numpy as np

# Kalman Equations for each time step 
class KalmanMath:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt
        self.x = np.zeros((4, 1))  # State vector: [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # Initial covariance matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # State transition matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])  # Measurement matrix
        self.Q = process_noise * np.array([[dt**4/4, 0, dt**3/2, 0],
                                           [0, dt**4/4, 0, dt**3/2],
                                           [dt**3/2, 0, dt**2, 0],
                                           [0, dt**3/2, 0, dt**2]]) # Process noise covariance
        self.R = measurement_noise * np.eye(2)  # Measurement noise covariance
    
    def predict(self):
        # Predict the next state
        self.x = np.dot(self.F, self.x)
        self.P = (self.F @ self.P) @ self.F.T + self.Q
        
    def update(self, z):
        z = np.reshape(z, (2, 1))
        y = z - (self.H @ self.x) 
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        
    @property
    def S(self):
        return self.H @ self.P @ self.H.T + self.R

        