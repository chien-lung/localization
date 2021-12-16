import numpy as np
from scipy.stats import triang, rayleigh

def calc_noise(distribution):
    """Sample noise from the given distribution

    Args:
        distribution (str): name of distribution

    Returns:
        np.array: sampled noise. Shape is (2,1)
    """
    if distribution == "gaussian":
        mu = np.zeros(2)
        cov = np.matrix([[3e-2, 4e-3],
                         [4e-3, 3e-2]])
        noise = np.random.multivariate_normal(mu, cov).reshape(2,1)
    elif distribution == "triangular":
        ratio = 0.1
        loc = -0.2
        scale = 0.25
        noise = triang.rvs(ratio, loc, scale, size=(2,1))
    elif distribution == "rayleigh":
        loc = -0.5
        scale = 0.55
        noise = rayleigh.rvs(loc, scale, size=(2,1))
    else:
        print("Undefined distribution.")
        noise = np.zeros((2,1))
    return noise

def measure(x, C, distribution="gaussian"):
    """Sensor measurement with noise

    Args:
        x (np.array): state. Shape is (3,1)
        C (np.array): sensor matrix, Shape is (2,3)
        distribution (str, optional): [description]. Defaults to "gaussian".

    Returns:
        np.array: measurement. Shape is (2,1)
    """
    z = C @ x + calc_noise(distribution)
    return z

def get_measurements(path, distribution="gaussian"):
    """Get measurements along the path

    Args:
        path (np.array): the robot's path. Shape is (N,3)
        distribution (str, optional): [description]. Defaults to "gaussian".

    Returns:
        np.array: measurements along the path. Shape is (N,2)
    """
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

    # Generate measurements
    sensor_noise_distribution = "triangular"
    print("Sensor noise type: ", sensor_noise_distribution)
    measurements = get_measurements(path, distribution=sensor_noise_distribution)
    data_dict["sensor_noise_distribution"] = sensor_noise_distribution
    data_dict["measurement"] = measurements

    # Draw ideal path, noisy path, measurements
    path_x = path[:, 0]
    path_y = path[:, 1]
    meas_x = measurements[:, 0]
    meas_y = measurements[:, 1]
    plt.plot(path_x, path_y, color="black", label="Path")
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