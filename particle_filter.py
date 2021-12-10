import numpy as np
from scipy.stats import multivariate_normal

class ParticleFilter:
    def __init__(self, num_particles, init_measurement, motion_noise_type="gaussian", sensor_noise_type="gaussian", R=None, Q=None, init_distribution="uniform",):
        self.N = num_particles
        self.N_TH = self.N//2
        self.particles = self.initialize_particles(init_measurement, init_distribution)
        self.weight = np.ones((self.N, 1))/self.N
        self.motion_noise_type = motion_noise_type
        self.sensor_noise_type = sensor_noise_type
        if self.motion_noise_type == "gaussian":
            self.R = R
        if self.sensor_noise_type == "gaussian":
            self.Q = Q

    def initialize_particles(self, init_measurement, distribution, init_range=(-0.1,0.1)):
        init_state = np.concatenate([init_measurement, np.array([0])])
        particles = np.zeros((self.N, 3)) + init_state
        if distribution == "uniform":
            low, high = init_range
            noise_x = np.random.uniform(low, high, self.N)
            noise_y = np.random.uniform(low, high, self.N)
            particles[:, 0] = noise_x + particles[:, 0]
            particles[:, 1] = noise_y + particles[:, 1]
        elif distribution == "gaussian":
            mu = 0
            sigma = (abs(init_range[0])+abs(init_range[1]))/2
            noise_x = np.random.uniform(mu, sigma, self.N)
            noise_y = np.random.uniform(mu, sigma, self.N)
            particles[:, 0] = noise_x + particles[:, 0]
            particles[:, 1] = noise_y + particles[:, 1]
        return particles
    
    def calc_motion_noise(self):
        if self.motion_noise_type == "gaussian":
            noise =  np.random.multivariate_normal([0,0,0], self.R).reshape(3,1)
        return noise

    def sensing_pdf(self, x):
        if self.sensor_noise_type == "gaussian":
            p = multivariate_normal.pdf(np.squeeze(x), cov=self.Q)
        return p

    def filter(self, z, u, A, B, C):
        for i in range(self.N):
            # Calculate next step of this particle
            p = self.particles[i].reshape(3,1)
            w = self.weight[i]
            p = A @ p + B @ u + self.calc_motion_noise()
            z_p = C @ p
            # Importance weight
            diff = z - z_p
            w *= self.sensing_pdf(diff)

            self.particles[i:i+1] = p.T
            self.weight[i] = w
        
        # Normalize
        self.weight = self.weight/self.weight.sum()
        N_eff = 1 / (self.weight.T @ self.weight).item()
        if N_eff < self.N_TH:
            self.resample()
        x = self.particles.T @ self.weight
        return x

    def resample(self):
        # Low-variance sampling
        wegiht_cum = np.cumsum(self.weight)
        base = np.arange(0., 1., 1/self.N)
        resample_id = base + np.random.uniform(0, 1/self.N)
        indices = []
        id = 0
        for i in range(self.N):
            while resample_id[i] > wegiht_cum[id]:
                id += 1
            indices.append(id)
        
        self.particles = self.particles[indices]
        self.weight = np.ones(self.weight.shape)/self.N

if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    from sensor import measure
    
    # Load data
    with open("data.pickle", "rb") as f:
        data = pickle.load(f)
    path = data["path"]
    control = data["control"]
    N = path.shape[0]
    # measurements = data["measurement"]
    
    # z0 = measurements[0]
    x0 = path[0].reshape(3,1)
    z0 = measure(x0, np.array([[1,0,0],[0,1,0]]))
    measurements = z0.T
    R = np.matrix([[3e-3, 1e-3, 0],
                   [1e-3, 3e-3, 0],
                   [0, 0, 1e-4]])
    Q = np.matrix([[3e-2, 4e-3],
                   [4e-3, 3e-2]])
    pf = ParticleFilter(100, z0.reshape(2), R=R, Q=Q)

    path_est = []
    for i in range(1, N):
        x_true = path[i].reshape(3,1)
        A = np.eye(3)
        B = np.array([[np.cos(path[i-1, 2]), 0],
                      [np.sin(path[i-1, 2]), 0],
                      [0, 1]])
        C = np.array([[1,0,0],
                      [0,1,0]])
        u = control[i].reshape(2,1)
        # z = measurements[i].reshape(2,1)
        z = measure(x_true, C, distribution="gaussian")
        measurements = np.vstack((measurements, z.T))
        x = pf.filter(z, u, A, B, C)
        path_est.append(x)

    # Exclude the first state
    path = path[1:]
    # Draw estimated path, real path, and measurements
    path_est_x = np.array([x[0,0] for x in path_est])
    path_est_y = np.array([x[1,0] for x in path_est])
    path_x = path[:, 0]
    path_y = path[:, 1]
    meas_x = measurements[:, 0]
    meas_y = measurements[:, 1]
    plt.plot(path_x, path_y, color="b", label="Noisy path")
    plt.plot(path_est_x, path_est_y, color="r", label="PF path")
    plt.scatter(meas_x, meas_y, label="Measurements")
    plt.legend()
    plt.show()

    diff_x = path_est_x-path_x
    diff_y = path_est_y-path_y
    error = np.sum(np.sqrt(np.square(diff_x)+np.square(diff_y)))
    print("Error: ", error)