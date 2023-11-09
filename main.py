# main.py
import numpy as np
import matplotlib.pyplot as plt
from algorithm_interface import OptimizationAlgorithm
from plotting import initialize_plot, update_plot
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

# Initialize plot
fig, ax = initialize_plot()

# Prepare the animation update function
def update(frame):
    # Update algorithms
    pso.update()
    ga.update()
    aco.update()

    # Clear the previous scatter plot
    while len(ax.collections) > 0:
        ax.collections[0].remove()

    # Update plots for each algorithm
    pso_data = pso.get_positions()
    ga_data = ga.get_positions()
    aco_data = aco.get_positions()

    # Ensure labels are set for legend
    update_plot(ax, pso_data, 'red', 'PSO')
    update_plot(ax, ga_data, 'blue', 'GA')
    update_plot(ax, aco_data, 'green', 'ACO')

    # Update the figure
    fig.canvas.draw()
    return fig,

# Initialize the animation
ani = FuncAnimation(fig, update, frames=range(num_iterations), blit=False, interval=100, repeat=False)

# Make sure to call plt.legend() after the plots have been created
# You can force a draw to make sure all artists are created
fig.canvas.draw()
plt.legend()

# Show the plot
plt.show()
