# main.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from plotting import initialize_plots, update_plot

# Import the algorithm classes
from algorithms.pso_algorithm import PSOAlgorithm
from algorithms.ga_algorithm import GAAlgorithm
from algorithms.aco_algorithm import ACOAlgorithm
from algorithms.abc_algorithm import ABCAlgorithm 

# Parameters for algorithm initialization (these should be defined according to your problem and algorithm requirements)
num_particles = 50  # This is an example, adjust accordingly
search_space = [-512, 512]  # Example search space
num_iterations = 50  # Number of iterations for the animation
pso_params = {'num_particles': num_particles, 'search_space': search_space, 'w': 0.5, 'c1': 0.1, 'c2': 0.1}
ga_params = {'population_size': num_particles, 'search_space': search_space}
aco_params = {'num_ants': num_particles, 'search_space': search_space, 'alpha': 1.0, 'decay': 0.1}
abc_params = {'num_bees': num_particles, 'search_space': search_space}

# Initialize algorithms with the necessary parameters
pso = PSOAlgorithm(**pso_params)
ga = GAAlgorithm(**ga_params)
aco = ACOAlgorithm(**aco_params)
abc = ABCAlgorithm(**abc_params)

# Initialize plots for each algorithm
fig, axes = initialize_plots()

# Define labels for the legend of each subplot
algorithm_names = ['PSO', 'GA', 'ACO', 'ABC'] 

# Initialize empty scatter plots with labels for the legend
scatter_plots = []
for ax, name in zip(axes, algorithm_names):
    scatter = ax.scatter([], [], [], color='none', label=name)  # Dummy scatter for legend
    scatter_plots.append(scatter)

# Create legends based on the initial (empty) scatter plots
for ax in axes:
    ax.legend(loc='upper right')

# Lists to keep track of scatter plot artists for each algorithm
scatter_plots_pso = []
scatter_plots_ga = []
scatter_plots_aco = []
scatter_plots_abc = []  

# Text objects to display the iteration number
iteration_texts = [ax.text2D(0.05, 0.95, '', transform=ax.transAxes) for ax in axes]

# Prepare the animation update function
def update(frame):
    # Update algorithms
    pso.update()
    ga.update()
    aco.update()
    abc.update()

    # Clear the previous scatter plots for each algorithm
    for scat in scatter_plots_pso + scatter_plots_ga + scatter_plots_aco + scatter_plots_abc:
        scat.remove()

    scatter_plots_pso.clear()
    scatter_plots_ga.clear()
    scatter_plots_aco.clear()
    scatter_plots_abc.clear()

    # Update plots for each algorithm on their respective axis
    pso_data = pso.get_positions()
    ga_data = ga.get_positions()
    aco_data = aco.get_positions()
    abc_data = abc.get_positions()

    # Plot and store the scatter plot artists
    scatter_plots_pso.append(update_plot(axes[0], pso_data, 'red', 'PSO'))
    scatter_plots_ga.append(update_plot(axes[1], ga_data, 'blue', 'GA'))
    scatter_plots_aco.append(update_plot(axes[2], aco_data, 'purple', 'ACO'))
    scatter_plots_abc.append(update_plot(axes[3], abc_data, 'black', 'ABC'))

    # Update iteration number and algorithm name
    for iter_text in iteration_texts:
        iter_text.set_text(f'Iteration: {frame}')

    # Update the figure
    fig.canvas.draw()
    fig.savefig(f'result/img/iteration_{frame}.png', dpi=200)
    return fig,

# Initialize the animation
ani = FuncAnimation(fig, update, frames=range(num_iterations), blit=False, interval=100, repeat=False)

# Show the plot
plt.show()