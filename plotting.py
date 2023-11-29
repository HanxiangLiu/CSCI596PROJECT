# plotting.py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from functions.objective_function import schwefel_function

def initialize_plots():
    fig = plt.figure(figsize=(12, 8))  # Adjust the figure size for 2x2 layout

    # Create four subplots for algorithms
    ax1 = fig.add_subplot(221, projection='3d')  # PSO
    ax2 = fig.add_subplot(222, projection='3d')  # GA
    ax3 = fig.add_subplot(223, projection='3d')  # ACO
    ax4 = fig.add_subplot(224, projection='3d')  # ABC

    axes = [ax1, ax2, ax3, ax4]

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
