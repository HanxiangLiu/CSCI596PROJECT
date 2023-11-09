# abc_algorithm.py
import numpy as np
from .algorithm_interface import OptimizationAlgorithm
from functions.objective_function import schwefel_function

class ABCAlgorithm(OptimizationAlgorithm):
    def __init__(self, num_bees, search_space, limit=100):
        self.num_bees = num_bees
        self.search_space = search_space
        self.limit = limit  # Scout bee limit before abandonment
        self.initialize()

    def initialize(self):
        # Initialize food sources randomly within the search space
        self.food_sources = np.random.uniform(self.search_space[0], self.search_space[1], (self.num_bees, 2))
        self.fitness = np.array([-schwefel_function(x, y) for x, y in self.food_sources])
        self.trial_counter = np.zeros(self.num_bees)
        self.best_source = np.copy(self.food_sources[np.argmax(self.fitness)])
        self.best_fitness = np.max(self.fitness)

    def update(self):
        # Employed bee phase
        for i in range(self.num_bees):
            new_source = self.get_neighbor(i)
            new_fitness = -schwefel_function(*new_source)
            if new_fitness > self.fitness[i]:
                self.food_sources[i] = new_source
                self.fitness[i] = new_fitness
                self.trial_counter[i] = 0
            else:
                self.trial_counter[i] += 1

        # Calculate probabilities for onlooker bees
        prob = (0.9 * self.fitness / np.max(self.fitness)) + 0.1

        # Onlooker bee phase
        for i in range(self.num_bees):
            if np.random.rand() < prob[i]:
                new_source = self.get_neighbor(i)
                new_fitness = -schwefel_function(*new_source)
                if new_fitness > self.fitness[i]:
                    self.food_sources[i] = new_source
                    self.fitness[i] = new_fitness
                    self.trial_counter[i] = 0
                else:
                    self.trial_counter[i] += 1

        # Scout bee phase
        for i in range(self.num_bees):
            if self.trial_counter[i] > self.limit:
                self.food_sources[i] = np.random.uniform(self.search_space[0], self.search_space[1], 2)
                self.fitness[i] = -schwefel_function(*self.food_sources[i])
                self.trial_counter[i] = 0

        # Memorize the best food source found so far
        index_best = np.argmax(self.fitness)
        if self.fitness[index_best] > self.best_fitness:
            self.best_source = np.copy(self.food_sources[index_best])
            self.best_fitness = self.fitness[index_best]

    def get_positions(self):
        # Return the current positions of bees and their fitness values
        return self.food_sources[:, 0], self.food_sources[:, 1], -self.fitness

    def get_neighbor(self, index):
        # Generate a neighbor food source
        phi = np.random.uniform(-1, 1, 2)
        k = np.random.randint(self.num_bees)
        while k == index:  # Ensure a different bee is selected
            k = np.random.randint(self.num_bees)
        new_source = self.food_sources[index] + phi * (self.food_sources[index] - self.food_sources[k])
        new_source = np.clip(new_source, self.search_space[0], self.search_space[1])
        return new_source

# The following code is for testing the algorithm independently
if __name__ == "__main__":
    # Test ABCAlgorithm
    num_bees = 10
    search_space = [-500, 500]
    abc = ABCAlgorithm(num_bees, search_space)

    # Test initialization
    assert abc.food_sources.shape == (num_bees, 2), "Initialization of food sources failed."

    # Test update
    previous_best_fitness = abc.best_fitness
    abc.update()
    assert abc.best_fitness >= previous_best_fitness, "Update failed to maintain/improve best fitness."

    # Test get_positions
    positions = abc.get_positions()
    assert len(positions) == 3, "get_positions should return a tuple of three elements."
    assert len(positions[0]) == num_bees, "get_positions returned incorrect number of positions."

    print("abc_algorithm.py tests passed.")
