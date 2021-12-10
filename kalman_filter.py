import numpy as np

class KalmanFilter:
    def __init__(self, R, Q) -> None:
        self.R = R
        self.Q = Q
    
    def predict(self, mu, Sigma, u, A, B):
        mu_next = A @ mu + B @ u
        Sigma_next = A @ Sigma @ A.T + self.R
        return mu_next, Sigma_next

    def update(self, mu, Sigma, z, C):
        I = np.identity(Sigma.shape[0])
        K = Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + self.Q)
        mu_next = mu + K @ (z - C @ mu)
        Sigma_next = (I - K @ C) @ Sigma
        return mu_next, Sigma_next

    def filter(self, mu, Sigma, z, u, A, B, C):
        mu, Sigma = self.predict(mu, Sigma, u, A, B)
        mu, Sigma = self.update(mu, Sigma, z, C)
        return mu, Sigma

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
    x0 = path[0].reshape(3,1)
    z0 = measure(x0, np.array([[1,0,0],[0,1,0]]))
    measurements = z0.T

    mu = path[0]
    Sigma = np.matrix([[1,0,0],
                       [0,1,0],
                       [0,0,1]])
    R = np.matrix([[3e-3, 1e-3, 0],
                   [1e-3, 3e-3, 0],
                   [0, 0, 1e-4]])
    Q = np.matrix([[3e-2, 4e-3],
                   [4e-3, 3e-2]])
    kf = KalmanFilter(R, Q)

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
        mu, Sigma = kf.filter(mu, Sigma, z, u, A, B, C)
        path_est.append(mu)
    
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
    plt.plot(path_est_x, path_est_y, color="r", label="KF path")
    plt.scatter(meas_x, meas_y, label="Measurements")
    plt.legend()
    plt.show()

    diff_x = path_est_x-path_x
    diff_y = path_est_y-path_y
    error = np.sum(np.sqrt(np.square(diff_x)+np.square(diff_y)))
    print("Error: ", error)