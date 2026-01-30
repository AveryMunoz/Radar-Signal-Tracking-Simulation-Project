# Real Position Simulation: Generates true position of observed objects moving in 2D space
#                           Assumes the point of observation is at origin (0,0)

import numpy as np
import matplotlib.pyplot as plt

# Base simulation parameters
boxSize = 1000 
mapSize = 1500
numObjects = 3
totalTime = 100
velocityRange = [-20, 20]
dt = 1

positions = np.random.uniform(0, boxSize, size = (numObjects, 2))
velocities = np.random.uniform(velocityRange[0], velocityRange[1], size = (numObjects, 2))
objectTrajectory = np.zeros((numObjects, totalTime, 2))
objectTrajectory[:, 0, :] = positions

for t in range (1, totalTime):
    positions = positions + velocities * dt
    objectTrajectory[:, t, :] = positions
    
print("Initial Positions: ")
for i in range(numObjects):
    print(f"Object {i+1}: {objectTrajectory[i, 0, :]}")

print("\nFinal Positions: ")
for i in range(numObjects):
    print(f"Object {i+1}: {objectTrajectory[i, -1, :]}")

# --- 1. TRUE TRAJECTORIES ---
