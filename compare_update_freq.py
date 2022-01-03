import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sensor import measure
from kalman_filter import KalmanFilter

def main(t):
    # Load data
    with open("data.pickle", "rb") as f:
        data = pickle.load(f)
    path = data["path"]
    controls = data["control"]
    N = path.shape[0]
    measurements = data["measurement"]
    # x0 = path[0].reshape(3,1)
    # z0 = measure(x0, np.array([[1,0,0],[0,1,0]]))
    # measurements = z0.T
    
    mu = path[0].reshape(3,1)
    Sigma = np.eye(3)
    R = np.matrix([[1e-2, 1e-4, 0],
                   [1e-4, 1e-2, 0],
                   [0, 0, 0]])
    Q = np.matrix([[8e-2, 1e-3],
                   [1e-3, 8e-2]])
    kf = KalmanFilter(R, Q, update_times=t)
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
        z = measurements[i].reshape(2,1)
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
    # plt.plot(path_x, path_y, color="black", label="Path")
    # plt.plot(path_est_x, path_est_y, color="blue", label="KF path")
    # plt.scatter(meas_x, meas_y, label="Measurements")
    # plt.title(f"Update each {t} times")
    # plt.legend()
    # plt.show()

    diff_x = path_est_x-path_x
    diff_y = path_est_y-path_y
    error = np.sum(np.sqrt(np.square(diff_x)+np.square(diff_y)))
    print("Error: ", error)

    print("Execution time: ", exec_time)
    return error, exec_time

if __name__ == "__main__":
    x = [1, 2, 5, 10, 20]
    es = []
    ts = []
    for u in x:
        e, t = main(u)
        es.append(e)
        ts.append(t*1000)
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(x, es, "b-", label="Error")
    ax1.set_xlabel("Update frequency (prediction/update)")
    ax1.set_ylabel("Error")
    ax2 = ax1.twinx()
    line2 = ax2.plot(x, ts, "r-", label="Exec time")
    lines = line1+line2
    labels = [l.get_label() for l in lines]
    ax2.set_ylabel("Execution time (ms)")
    fig.suptitle("Comparison of update frequency.")
    ax1.legend(lines, labels)
    plt.show()