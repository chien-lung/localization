import numpy as np

class KalmanFilter:
    def __init__(self, R, Q, update_times=1) -> None:
        self.index = 0
        self.update_times = update_times
        self.R = R
        self.Q = Q
    
    def predict(self, mu, Sigma, u, A, B):
        """Prediction step of KF

        Args:
            mu (np.array): previous state (also a mean). Shape is (3,1)
            Sigma (np.array): covariance of the state. Shape is (3,3)
            u (np.array): control. Shape is (2,1)
            A (np.array): transition matrix of state. Shape is (3,3)
            B (np.array): transition matrix of control. Shape is (3,2)

        Returns:
            np.array: predicted state. Shape is (3,1)
            np.array: predicted covariance. Shape is (3,3)
        """
        mu_next = A @ mu + B @ u
        Sigma_next = A @ Sigma @ A.T + self.R
        return mu_next, Sigma_next

    def update(self, mu, Sigma, z, C):
        """Correction step of KF

        Args:
            mu (np.array): state (also a mean). Shape is (3,1)
            Sigma (np.array): covariance of the state. Shape is (3,3)
            z (np.array): measurement. Shape is (2,1)
            C (np.array): sensor matrix. Shape is (2,3)

        Returns:
            np.array: updated state. Shape is (3,1)
            np.array: updated covariance. Shape is (3,3)
        """
        I = np.identity(Sigma.shape[0])
        K = Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + self.Q)
        mu_next = mu + K @ (z - C @ mu)
        Sigma_next = (I - K @ C) @ Sigma
        return mu_next, Sigma_next

    def filter(self, mu, Sigma, z, u, A, B, C):
        """Filter the previous to the current state

        Args:
            mu (np.array): state (also a mean). Shape is (3,1)
            Sigma (np.array): covariance of the state. Shape is (3,3)
            z (np.array): measurement. Shape is (2,1)
            u (np.array): control. Shape is (2,1)
            A (np.array): transition matrix of state. Shape is (3,3)
            B (np.array): transition matrix of control. Shape is (3,2)
            C (np.array): sensor matrix. Shape is (2,3)

        Returns:
            np.array: updated state. Shape is (3,1)
            np.array: updated covariance. Shape is (3,3)
        """
        self.index += 1
        mu, Sigma = self.predict(mu, Sigma, u, A, B)
        if self.index % self.update_times == 0:
            mu, Sigma = self.update(mu, Sigma, z, C)
            self.index = 0
        return mu, Sigma

if __name__ == "__main__":
    import time
    import pickle
    import matplotlib.pyplot as plt
    from sensor import measure
    
    # Load data
    with open("data.pickle", "rb") as f:
        data = pickle.load(f)
    path = data["path"]
    controls = data["control"]
    N = path.shape[0]
    # Uncomment to use saved measurements
    measurements = data["measurement"]
    # Uncomment to measure at each time
    # x0 = path[0].reshape(3,1)
    # z0 = measure(x0, np.array([[1,0,0],[0,1,0]]))
    # measurements = z0.T

    # Iniliazite Kalman Filter
    mu = path[0].reshape(3,1)
    Sigma = np.eye(3)
    R = np.matrix([[1e-2, 1e-4, 0],
                   [1e-4, 1e-2, 0],
                   [0, 0, 0]])
    Q = np.matrix([[8e-2, 1e-3],
                   [1e-3, 8e-2]])
    kf = KalmanFilter(R, Q)

    # Execute KF along the path
    start_time = time.time()
    path_est = []
    for i in range(1, N):
        x_true = path[i].reshape(3,1)
        A = np.eye(3)
        B = np.array([[np.cos(path[i-1, 2]), 0],
                      [np.sin(path[i-1, 2]), 0],
                      [0, 1]])
        C = np.array([[1,0,0],
                      [0,1,0]])
        u = controls[i].reshape(2,1)
        # Uncomment to use saved measurements
        z = measurements[i].reshape(2,1)
        # Uncomment to measure at each time
        # z = measure(x_true, C, distribution="triangular")
        # measurements = np.vstack((measurements, z.T))
        mu, Sigma = kf.filter(mu, Sigma, z, u, A, B, C)
        path_est.append(mu)
    exec_time = time.time() - start_time
    
    # Exclude the first state
    path = path[1:]
    # Draw estimated path, real path, and measurements
    path_est_x = np.array([x[0,0] for x in path_est])
    path_est_y = np.array([x[1,0] for x in path_est])
    path_x = path[:, 0]
    path_y = path[:, 1]
    meas_x = measurements[:, 0]
    meas_y = measurements[:, 1]
    plt.plot(path_x, path_y, color="black", label="Path")
    plt.plot(path_est_x, path_est_y, color="blue", label="KF path")
    plt.scatter(meas_x, meas_y, label="Measurements")
    plt.legend()
    plt.show()

    # Execution time
    print("Execution time: ", exec_time)
    
    # Compute error
    diff_x = path_est_x-path_x
    diff_y = path_est_y-path_y
    error = np.sum(np.sqrt(np.square(diff_x)+np.square(diff_y)))
    print("Error: ", error)