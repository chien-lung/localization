# Localization
EECS 498 Algorithmic Robotics - Full 2021

Prof. Dmitry Berenson

Implement Kalman Filter (KF) and Particle Filter (PF) to localize the PR2 robot with the given path.
To realize more details, please read our [final project report](Final_Project_Report.pdf).


# Getting started

## Install necessary packages
We use the following packages in this project: `numpy`, `scipy`, `pybullet`, and `matplotlib`. 

```
$ bash install.sh
```

## Generate the data
We assume we already has a path which includes motion noise (i.e., `data_path.pickle`). This path is a N by 3 matrix, which each row represents a state.

![path](images/path.png)

### Control and motion noise generation
We obtain the control at each time step by generaing the motion noise and solving the motion model. By running this command, you will have `data.pickle`. This contains controls which is a N by 2 matrix, each row represents a control. Note that this ideal path is not involve in the preojct, just for visualization.
```
$ python motion.py
```

![ideal_path](images/ideal_path.png)

### Measurement generation
To fairly compare algorithms with the same measurements, we obtain the measurements of all states. By running this command, you will update `data.pickle`. This contains measurements which is a N by 2 matrix, each row represents a measurement. Note that we implement several distribution (e.g., Gaussian, triangular, and Rayleigh), which you can specify in the file.
```
$ python sensor.py
```

![measurements](images/measurements.png)

## Run KF and PF
After having controls, measurements, and the path, we can start to run the filtering algorithms.

### Kalman Filter
By running this command, you will see a figure containing the estimated path (blue) and the real path (black).
```
$ python kalman_filter.py
Execution time:  0.010622262954711914
Error:  20.401129962175126
```

![kf](images/kf_tri.png)

### Particle Filter
By running this command, you will see a figure containing the estimated path (red) and the real path (black).
```
$ python particle_filter.py 
Execution time:  4.653516054153442
Error:  7.694496798550877
```

![pf](images/pf_tri.png)

## Compare KF and PF
Besides plotting the path on the xy-plane, we can also simulate the PR2 robot in the given enviroment.
```
$ python demo.py
Execution time:
KF:  0.18025827407836914
PF:  12.547127723693848
Error:
KF:  20.062203182620284
PF:  7.642154967053671
```

![pr2](images/pr2_tri.png)

# Visualization
## Noise distribution
To visualize different noises we choose, we plot their distribution to realize whether this distribution is suitable for our implementation. 

```
$ python plot_distribution.py
```

![dist](images/distribution.png)

## Comparison
In addition to the comparison between KF and PF, we also want to compare the filtering algorithm with different parameters.

### Number of particles used in PF
We demonstrate the tradeoff between execution time and error for PF using different number of particles.

```
$ python compare_num_of_particles.py
```

![comp1](images/comp_num_of_particles.png)

### Update frequency of KF
We demonstrate the tradeoff between execution time and error for KF with different update frequency.

```
$ python compare_update_freq.py
```

![comp2](images/comp_update_freq.png)

# Contact
Chien-Lung Chou ([lungchou@umich.edu](lungchou@umich.edu))

Wan-Yi Yu ([wendyu@umich.edu](wendyu@umich.edu))