# plotting.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

from functions.objective_function import schwefel_function

def initialize_plots():
    fig = plt.figure(figsize=(18, 6))

    # Create three subplots for PSO, GA, and ACO
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    axes = [ax1, ax2, ax3]

    # Plot the surface on each subplot
    for ax in axes:
        x = np.linspace(-512, 512, 400)
        y = np.linspace(-512, 512, 400)
        X, Y = np.meshgrid(x, y)
        Z = schwefel_function(X, Y)
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.3)
        ax.view_init(elev=34, azim=134)

    return fig, axes

def update_plot(ax, algorithm_data, color, label):
    X, Y, Z = algorithm_data
    return ax.scatter(X, Y, Z, color=color, label=label)
