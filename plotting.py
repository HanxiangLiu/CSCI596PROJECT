# plotting.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from objective_function import schwefel_function
import numpy as np

def initialize_plot():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-512, 512, 400)
    y = np.linspace(-512, 512, 400)
    X, Y = np.meshgrid(x, y)
    Z = schwefel_function(X, Y)
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.3)
    return fig, ax

def update_plot(ax, algorithm_data, color, label):
    X, Y, Z = algorithm_data
    scat = ax.scatter(X, Y, Z, color=color, label=label)
    return scat
