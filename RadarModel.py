import numpy as np

class RadarModel:
    def __init__(self, radar_pos = np.array([0.0, 0.0]), max_range = 3000, sigma_base = 50, range_ref = 5000, lambda_clutter = 50):
        self.radar_pos = radar_pos
        self.max_range = max_range
        self.sigma_base = sigma_base
        self.range_ref = range_ref
        self.lambda_clutter = lambda_clutter
     
    # Geometry Functions   
    def compute_range(self, position):
        return np.linalg.norm(position - self.radar_pos)

    def compute_radial_velocity(self, velocity, position):
        r_vec = position - self.radar_pos
        r_hat = r_vec / np.linalg.norm(r_vec)
        return np.dot(velocity, r_hat)

    # Noise and Uncertainty Models
    def sigma_range(self, r):
        return self.sigma_base * (r / self.range_ref)**2

    # Detection Model
    def detection_probability(self, r, P_Max = 0.95, k=2):
        if r > self.max_range:
            return 0.0
        else:
            return P_Max * (1 - (r / self.max_range)**k)

    # Probability of detection ACTUALLY being detected for realism
    def is_detected(self, r):
        return np.random.rand() < self.detection_probability(r)

    # Clutter
    def generate_clutter(self, mapSize):
        n = np.random.poisson(self.lambda_clutter)
        xs = np.random.uniform(-mapSize, mapSize, n)
        ys = np.random.uniform(-mapSize, mapSize, n)
        return np.column_stack((xs, ys))

    # Measurement Generation
    def generate_measurement(self, true_pos):
        r = self.compute_range(true_pos)
        sigma = self.sigma_range(r)
        noise = np.random.normal(0, sigma, size = 2)
        return true_pos + noise

    def simulate_frame(self, objectPositions, mapSize): 
        measurement = []
        
        # A true detecction
        for position in objectPositions:
            r = self.compute_range(position)
            if self.is_detected(r):
                meas = self.generate_measurement(position)
                measurement.append(meas)
                
        # cLutter
        clutter = self.generate_clutter(mapSize)
        for c in clutter:
            measurement.append(c)
            
        return np.array(measurement)

    def simulate_all_frames(self, trajectory, mapSize):
        numObjects, totalTime, _ = trajectory.shape
        all_frames = []
        
        for t in range(totalTime):
            positions = trajectory[:, t, :]
            frame = self.simulate_frame(positions, mapSize)
            all_frames.append(frame)
            
        return all_frames
