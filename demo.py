import time
import pickle
import numpy as np
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS

from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line, draw_path
from kalman_filter import KalmanFilter
from particle_filter import ParticleFilter
from sensor import measure

def load_data(filename="data.pickle"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    path = data["path"]
    controls = data["control"]
    measurements = data["measurement"]
    N = path.shape[0]
    return path, controls, measurements, N

def calc_error(x_est, x_true):
    diff = x_est - x_true
    return np.linalg.norm(diff[:2])

def config_to_state(config):
    return np.array([list(config)]).T

def state_to_config(state):
    return tuple(np.array(state).reshape(3))


def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2_proj_env.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))
    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    path_gt = []
    
    # System dynamics
    A = np.eye(3)
    C = np.array([[1,0,0],
                  [0,1,0]])

    # Load data
    path, controls, measurements, N = load_data()
    path_gt = [state_to_config(x) for x in path]

    ########## Kalman Filter ###########
    start_time = time.time()
    mu = path[0].reshape(3,1)
    Sigma = np.eye(3)
    R = np.matrix([[1e-2, 1e-4, 0],
                   [1e-4, 1e-2, 0],
                   [0, 0, 0]])
    Q = np.matrix([[8e-2, 1e-3],
                   [1e-3, 8e-2]])
    kf = KalmanFilter(R, Q)

    error_kf = 0
    path_kf = []
    for i in range(1, N):
        x_true = path[i].reshape(3,1)
        B = np.array([[np.cos(x_true[2,0]), 0],
                      [np.sin(x_true[2,0]), 0],
                      [0, 1]])
        u = controls[i].reshape(2,1)
        z = measurements[i].reshape(2,1)
        # z = measure(x_true, C, distribution="triangular")
        # measurements = np.vstack((measurements, z.T))
        mu, Sigma = kf.filter(mu, Sigma, z, u, A, B, C)
        path_kf.append(state_to_config(mu))
        error_kf += calc_error(mu, x_true)
    time_kf = time.time() - start_time
    ####################################

    ######### Particle Filter ##########
    start_time = time.time()
    num_particles = 100
    pf_sensor_noise_type = "gaussian"
    z0 = measurements[0]
    R = np.matrix([[1e-2, 1e-4, 0],
                   [1e-4, 1e-2, 0],
                   [0, 0, 0]])
    Q = np.matrix([[8e-2, 1e-3],
                   [1e-3, 8e-2]])
    pf = ParticleFilter(num_particles, z0.reshape(2), R=R, Q=Q, sensor_noise_type=pf_sensor_noise_type)

    error_pf = 0
    path_pf = []
    for i in range(1, N):
        x_true = path[i].reshape(3,1)
        B = np.array([[np.cos(path[i-1, 2]), 0],
                      [np.sin(path[i-1, 2]), 0],
                      [0, 1]])
        u = controls[i].reshape(2,1)
        z = measurements[i].reshape(2,1)
        # z = measure(x_true, C, distribution="gaussian")
        # measurements = np.vstack((measurements, z.T))
        x_est = pf.filter(z, u, A, B, C)
        path_pf.append(state_to_config(x_est))
        error_pf += calc_error(x_est, x_true)
    time_pf = time.time() - start_time
    ####################################

   
    # Draw each path 
    draw_path(path_kf, color=(0,0,255))
    draw_path(path_pf, color=(255,0,0))
    draw_path(path_gt)

    # Show execution time
    print("Execution time:")
    print("KF: ", time_kf)
    print("PF: ", time_pf)

    # Show error
    print("Error:")
    print("KF: ", error_kf)
    print("PF: ", error_pf)
    
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path_gt, sleep=0.05)

    # Check collision
    for node in path_kf:
        if collision_fn(node):
            draw_sphere_marker((node[0], node[1], 1.5), 0.05, (0, 0, 1, 1))
    for node in path_pf:
        if collision_fn(node):
            draw_sphere_marker((node[0], node[1], 1.2), 0.05, (1, 0, 0, 1))
    
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()