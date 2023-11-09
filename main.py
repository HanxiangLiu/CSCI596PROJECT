# main.py
import numpy as np
import matplotlib.pyplot as plt
from algorithm_interface import OptimizationAlgorithm
from plotting import initialize_plots, update_plot
from matplotlib.animation import FuncAnimation

# Import the algorithm classes
from pso_algorithm import PSOAlgorithm
from ga_algorithm import GAAlgorithm
from aco_algorithm import ACOAlgorithm

# Parameters for algorithm initialization (these should be defined according to your problem and algorithm requirements)
num_particles = 50  # This is an example, adjust accordingly
search_space = [-512, 512]  # Example search space
num_iterations = 100  # Number of iterations for the animation
pso_params = {'num_particles': num_particles, 'search_space': search_space, 'w': 0.5, 'c1': 0.1, 'c2': 0.1}
ga_params = {'population_size': num_particles, 'search_space': search_space}
aco_params = {'num_ants': num_particles, 'search_space': search_space, 'alpha': 1.0, 'decay': 0.1}

# Initialize algorithms with the necessary parameters
pso = PSOAlgorithm(**pso_params)
ga = GAAlgorithm(**ga_params)
aco = ACOAlgorithm(**aco_params)

# Initialize plots for each algorithm
fig, axes = initialize_plots()

# Lists to keep track of scatter plot artists for each algorithm
scatter_plots_pso = []
scatter_plots_ga = []
scatter_plots_aco = []

# Prepare the animation update function
def update(frame):
    # Update algorithms
    pso.update()
    ga.update()
    aco.update()

    # Clear the previous scatter plots for each algorithm
    for scat in scatter_plots_pso + scatter_plots_ga + scatter_plots_aco:
        scat.remove()

    scatter_plots_pso.clear()
    scatter_plots_ga.clear()
    scatter_plots_aco.clear()

    # Update plots for each algorithm on their respective axis
    pso_data = pso.get_positions()
    ga_data = ga.get_positions()
    aco_data = aco.get_positions()

    # Plot and store the scatter plot artists
    scatter_plots_pso.append(update_plot(axes[0], pso_data, 'red', 'PSO'))
    scatter_plots_ga.append(update_plot(axes[1], ga_data, 'blue', 'GA'))
    scatter_plots_aco.append(update_plot(axes[2], aco_data, 'purple', 'ACO'))

    # Update the figure
    fig.canvas.draw()
    return fig,

# Initialize the animation
ani = FuncAnimation(fig, update, frames=range(num_iterations), blit=False, interval=100, repeat=False)

# Show the plot with the legend
for ax in axes:
    ax.legend()

plt.show()
