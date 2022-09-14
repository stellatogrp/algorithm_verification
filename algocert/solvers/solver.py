from abc import ABC, abstractmethod


class Solver(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def canonicalize(self, **kwargs):
        pass
