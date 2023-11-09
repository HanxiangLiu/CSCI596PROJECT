# pso_algorithm.py
import numpy as np
from algorithm_interface import OptimizationAlgorithm
from objective_function import schwefel_function

class PSOAlgorithm(OptimizationAlgorithm):
    def __init__(self, num_particles, search_space, w, c1, c2):
        self.num_particles = num_particles
        self.search_space = search_space
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.initialize()

    def initialize(self):
        self.positions = np.random.uniform(self.search_space[0], self.search_space[1], (self.num_particles, 2))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, 2))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.array([schwefel_function(x, y) for x, y in self.positions])
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)

    def update(self):
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities[i] = self.w * self.velocities[i] + \
                                 self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + \
                                 self.c2 * r2 * (self.global_best_position - self.positions[i])
            self.positions[i] += self.velocities[i]
            score = schwefel_function(self.positions[i, 0], self.positions[i, 1])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i]
                if score < self.global_best_score:
                    self.global_best_position = self.positions[i]
                    self.global_best_score = score

    def get_positions(self):
        # Get the current positions along with their corresponding function values
        return self.positions[:, 0], self.positions[:, 1], [schwefel_function(x, y) for x, y in self.positions]
    


if __name__ == "__main__":
    # Test PSOAlgorithm
    num_particles = 30
    search_space = [-500, 500]
    w = 0.5
    c1 = 0.8
    c2 = 0.9
    pso = PSOAlgorithm(num_particles, search_space, w, c1, c2)

    # Test initialization
    assert pso.positions.shape == (num_particles, 2), "Initialization of positions failed."

    # Test update
    previous_global_best_score = pso.global_best_score
    pso.update()
    assert pso.global_best_score <= previous_global_best_score, "Update failed to maintain/improve global best score."

    # Test get_positions
    positions = pso.get_positions()
    assert len(positions) == 3, "get_positions should return a tuple of three elements."
    assert len(positions[0]) == num_particles, "get_positions returned incorrect number of positions."

    print("pso_algorithm.py tests passed.")