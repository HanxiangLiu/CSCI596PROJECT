# aco_algorithm.py
import numpy as np
from .algorithm_interface import OptimizationAlgorithm
from functions.objective_function import schwefel_function

class ACOAlgorithm(OptimizationAlgorithm):
    def __init__(self, num_ants, search_space, alpha, decay):
        self.num_ants = num_ants
        self.search_space = search_space
        self.alpha = alpha
        self.decay = decay
        self.initialize()

    def initialize(self):
        self.positions = np.random.uniform(self.search_space[0], self.search_space[1], (self.num_ants, 2))
        self.scores = np.array([schwefel_function(x, y) for x, y in self.positions])
        self.pheromone_map = np.ones(self.num_ants)
        self.best_score = np.min(self.scores)
        self.best_position = self.positions[np.argmin(self.scores)]

    def update(self):
        for i, pos in enumerate(self.positions):
            new_pos = pos + np.random.uniform(-10, 10, 2)
            new_pos = np.clip(new_pos, self.search_space[0], self.search_space[1])
            new_score = schwefel_function(new_pos[0], new_pos[1])
            self.scores[i] = new_score
            self.positions[i] = new_pos
            self.pheromone_map[i] = (1 - self.decay) * self.pheromone_map[i] + self.alpha * (1 / (1 + new_score))
            if new_score < self.best_score:
                self.best_score = new_score
                self.best_position = new_pos

    def get_positions(self):
        return self.positions[:, 0], self.positions[:, 1], self.scores


if __name__ == "__main__":
    # Test ACOAlgorithm
    num_ants = 10
    search_space = [-500, 500]
    alpha = 1
    decay = 0.1
    aco = ACOAlgorithm(num_ants, search_space, alpha, decay)

    # Test initialization
    assert aco.positions.shape == (num_ants, 2), "Initialization of positions failed."

    # Test update
    previous_best_score = aco.best_score
    aco.update()
    assert aco.best_score <= previous_best_score, "Update failed to maintain/improve best score."

    # Test get_positions
    positions = aco.get_positions()
    assert len(positions) == 3, "get_positions should return a tuple of three elements."
    assert len(positions[0]) == num_ants, "get_positions returned incorrect number of positions."

    print("aco_algorithm.py tests passed.")