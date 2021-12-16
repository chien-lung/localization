import numpy as np
from numpy.lib.twodim_base import tri
from scipy.stats import triang, rayleigh, multivariate_normal
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Gaussian
    mu = np.zeros(2)
    Q = np.matrix([[3e-2, 4e-3],
                   [4e-3, 3e-2]])
    x, y = np.mgrid[-0.3:0.3:0.01, -0.3:0.3:0.01]
    pos = np.dstack((x,y))
    r = multivariate_normal(mu, Q)
    plt.contourf(x, y, r.pdf(pos))
    plt.colorbar()
    plt.title("Gaussian: mu=0, cov=[[3e-2,4e-3],[4e-3,3e-2]]")
    plt.show()

    # Triangular
    ratio = 0.1
    loc = -0.2
    scale = 0.25
    x = np.linspace(triang.ppf(0.001, ratio, loc, scale), triang.ppf(0.999, ratio, loc, scale), 1000)
    r = triang.rvs(ratio, loc, scale, size=1000)
    plt.plot(x, triang.pdf(x, ratio, loc, scale), "r-", lw=5, alpha=0.8,label="Triangular pdf")
    plt.hist(r, color="blue", density=True, alpha=0.3, label="Sampled noises")
    plt.axvline(loc+ratio*scale, color="black",label="Mode")
    plt.title("Triangular: ratio=0.1, loc=-0.2, scale=0.25")
    plt.legend()
    plt.show()
    
    # Rayleigh
    loc = -0.5
    scale = 0.55
    x = np.linspace(rayleigh.ppf(0.001, loc, scale), rayleigh.ppf(0.999, loc, scale), 1000)
    r = rayleigh.rvs(loc, scale, size=1000)
    plt.plot(x, rayleigh.pdf(x, loc, scale), "r-", lw=5, alpha=0.8,label="Rayleigh pdf")
    plt.hist(r, color="blue", density=True, alpha=0.3, label="Sampled noises")
    plt.axvline(1*scale+loc, color="black",label="Mode")
    plt.title("Rayleigh: loc=-0.5, scale=0.55")
    plt.legend()
    plt.show()
