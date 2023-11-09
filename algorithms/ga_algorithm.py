# ga_algorithm.py
import numpy as np
from .algorithm_interface import OptimizationAlgorithm
from functions.objective_function import schwefel_function

class GAAlgorithm(OptimizationAlgorithm):
    def __init__(self, population_size, search_space):
        self.population_size = population_size
        self.search_space = search_space
        self.initialize()

    def initialize(self):
        self.population = np.random.uniform(self.search_space[0], self.search_space[1], (self.population_size, 2))
        self.scores = np.array([schwefel_function(individual[0], individual[1]) for individual in self.population])

    def select(self):
        num_parents = self.population_size // 2
        parents = np.empty((num_parents, self.population.shape[1]))
        for i in range(num_parents):
            max_fitness_idx = np.argmin(self.scores)
            parents[i, :] = self.population[max_fitness_idx, :]
            self.scores[max_fitness_idx] = 99999999999
        return parents

    def crossover(self, parents):
        offspring_size = (self.population_size - parents.shape[0], self.population.shape[1])
        offspring = np.empty(offspring_size)
        crossover_point = np.uint8(offspring_size[1] / 2)
        for k in range(offspring_size[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(self, offspring_crossover):
        for idx in range(offspring_crossover.shape[0]):
            random_value = np.random.uniform(-1, 1, 1)
            offspring_crossover[idx, :] += random_value
        return offspring_crossover

    def update(self):
        parents = self.select()
        offspring_crossover = self.crossover(parents)
        offspring_mutation = self.mutation(offspring_crossover)
        self.population[:parents.shape[0], :] = parents
        self.population[parents.shape[0]:, :] = offspring_mutation
        self.scores = np.array([schwefel_function(individual[0], individual[1]) for individual in self.population])

    def get_positions(self):
        # This method should return the positions and scores in a format that can be used by the plotting function
        # Assuming the plotting function expects a 3-tuple of arrays (X, Y, Z)
        X = self.population[:, 0]
        Y = self.population[:, 1]
        Z = np.array([schwefel_function(x, y) for x, y in self.population])
        return X, Y, Z


if __name__ == "__main__":
    # Test GAAlgorithm
    population_size = 20
    search_space = [-500, 500]
    ga = GAAlgorithm(population_size, search_space)

    # Test initialization
    assert ga.population.shape == (population_size, 2), "Initialization of population failed."

    # Test select
    parents = ga.select()
    assert parents.shape[0] == population_size // 2, "Selection did not return the correct number of parents."

    # Test crossover
    offspring = ga.crossover(parents)
    assert offspring.shape == (population_size - parents.shape[0], 2), "Crossover did not return the correct number of offspring."

    # Test mutation
    mutated_offspring = ga.mutation(offspring)
    assert mutated_offspring.shape == offspring.shape, "Mutation did not return the correct shape."

    # Test update
    previous_best_score = np.min(ga.scores)
    ga.update()
    assert np.min(ga.scores) <= previous_best_score, "Update failed to maintain/improve best score."

    print("ga_algorithm.py tests passed.")