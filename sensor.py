import numpy as np
from scipy.stats import triang
from scipy.stats import binom

def calc_noise(distribution):
    if distribution == "gaussian":
        mu = np.zeros(2)
        Q = np.matrix([[3e-2, 4e-3],
                         [4e-3, 3e-2]])
        noise = np.random.multivariate_normal(mu, Q).reshape(2,1)
    elif distribution == "triangular":
        ratio = 0.9
        loc = -0.2
        scale = 0.25
        noise = binom.rvs(ratio, loc, scale, size=(2,1))
    elif distribution == "binomial":
        ratio = 0.9
        loc = -0.2
        scale = 0.25
        noise = binom.rvs(ratio, loc, scale, size=(2,1))
    elif distribution == "poisson":
        ratio = 0.9
        loc = -0.2
        scale = 0.25
        noise = np.random.poisson(ratio, loc, scale, size=(2,1))
    elif distribution == "bernoulli":
        ratio = 0.9
        loc = -0.2
        scale = 0.25
        noise = np.random.vinomial(ratio, loc, scale, size=(2,1))
    else:
        noise = np.zeros((2,1))
    return noise

def measure(x, C, distribution="gaussian"):
    z = C @ x + calc_noise(distribution)
    return z

def get_measurements(path, distribution="gaussian"):
    C = np.matrix([[1,0,0],
                   [0,1,0]])
    measurements = np.zeros((0, 2))
    for i in range(path.shape[0]):
        x = path[i].reshape(3,1)
        z = measure(x, C, distribution)
        measurements = np.vstack((measurements, z.T))
    return np.array(measurements)

def main():
    import pickle
    import matplotlib.pyplot as plt

    # Load path
    with open("data.pickle", "rb") as f:
        data_dict = pickle.load(f)
    path = data_dict["path"]
    path_ideal = data_dict["path_ideal"]

    # Generate measurements
    measurements = get_measurements(path)
    data_dict["measurement"] = measurements

    # Draw ideal path, noisy path, measurements
    path_x = path[:, 0]
    path_y = path[:, 1]
    path_x_ideal = path_ideal[:, 0]
    path_y_ideal = path_ideal[:, 1]
    meas_x = measurements[:, 0]
    meas_y = measurements[:, 1]
    plt.plot(path_x, path_y, color="b", label="Noisy path")
    plt.plot(path_x_ideal, path_y_ideal, color="r", label="Ideal path")
    plt.scatter(meas_x, meas_y, label="Measurements")
    plt.legend()
    plt.show()

    # Save measurements
    save = input("Enter y to save: ")
    if save == "y":
        with open("data.pickle", "wb") as f:
            pickle.dump(data_dict, f)
        print("Data is saved")

if __name__ == "__main__":
    main()