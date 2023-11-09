# plotting.py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from functions.objective_function import schwefel_function

def initialize_plots():
    fig = plt.figure(figsize=(18, 12))  # Adjusted the figure size for 2x3 layout

    # Create six subplots for algorithms and two placeholders
    # First row: PSO, GA, ACO
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(232, projection='3d')
    ax3 = fig.add_subplot(233, projection='3d')
    # Second row: ABC, Placeholder 1, Placeholder 2
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')  # Placeholder, no algorithm yet
    ax6 = fig.add_subplot(236, projection='3d')  # Placeholder, no algorithm yet

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # Plot the surface on each active subplot
    for ax in axes[:-2]:  # Skipping the last two placeholders
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
