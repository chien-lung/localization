import numpy as np

def calc_noise(path, mu, cov):
    noises = np.zeros((1, 3))
    for i in range(1, path.shape[0]):
        noise = np.random.multivariate_normal(mu, cov)
        noises = np.vstack((noises, noise))
    return noises

def get_control(path, noise):
    A = np.matrix([[1,0,0],
                   [0,1,0],
                   [0,0,1]])
    # x1 = Ax0 + Bu1 + n1
    controls = np.zeros((1, 2))
    for i in range(1, path.shape[0]):
        x0 = path[i-1, :].reshape(3,1)
        B = np.matrix([[np.cos(x0[2,0]), 0],
                       [np.sin(x0[2,0]), 0],
                       [0, 1]])
        x1 = path[i, :].reshape(3,1)
        n1 = noise[i, :].reshape(3,1)
        u1 = np.linalg.pinv(B) @ (x1 - A @ x0 - n1)
        controls = np.vstack((controls, u1.T))
    return controls

def main():
    import pickle
    import matplotlib.pyplot as plt
    # Load path
    with open("data.pickle", "rb") as f:
        data_dict = pickle.load(f)
    path = data_dict["path"]

    # Generate noise
    mu = np.zeros(path.shape[1])
    R = np.matrix([[3e-3, 1e-3, 0],
                     [1e-3, 3e-3, 0],
                     [0, 0, 1e-4]])
    noises = calc_noise(path, mu, R)
    data_dict["motion_noise"] = noises

    # Generate control according to noises
    controls = get_control(path, noises)
    data_dict["control"] = controls

    # Assume the noise was included in the path, then
    # we want to generate path without noise, i.e., path_ideal
    path_x = path[:, 0]
    path_y = path[:, 1]
    path_ideal = path[0].reshape(1,3)
    path_x_no_noise = [path[0,0]]
    path_y_no_noise = [path[0,1]]
    x0 = path[0].reshape(3,1)
    for i in range(1, path.shape[0]):
        A = np.matrix([[1,0,0],
                       [0,1,0],
                       [0,0,1]])
        B = np.matrix([[np.cos(x0[2,0]), 0],
                       [np.sin(x0[2,0]), 0],
                       [0, 1]])
        u = controls[i].T
        x1 = A @ x0 + B @ u
        path_x_no_noise.append(x1[0,0])
        path_y_no_noise.append(x1[1,0])
        x0 = x1
        path_ideal = np.vstack((path_ideal, x1.T))
    
    data_dict["path_ideal"] = path_ideal

    # Draw ideal path and noisy path
    plt.plot(path_x, path_y, color="b", label="Noisy path")
    plt.plot(path_x_no_noise, path_y_no_noise, color="r", label="Ideal path")
    plt.legend()
    plt.show()

    # Save noise, control, ideal path
    save = input("Enter y to save: ")
    if save == "y":
        with open("data.pickle", "wb") as f:
            pickle.dump(data_dict, f)
        print("Data is saved")

if __name__ == "__main__":
    main()
