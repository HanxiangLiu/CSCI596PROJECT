# algorithm_interface.py
from abc import ABC, abstractmethod

class OptimizationAlgorithm(ABC):

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """ Initialize the algorithm with necessary parameters. """
        pass

    @abstractmethod
    def update(self):
        """ Update the positions of particles or solutions. """
        pass

    @abstractmethod
    def get_positions(self):
        """ Get the current positions of particles or solutions. """
        pass
