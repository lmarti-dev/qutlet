import numpy as np


class OptimisationStep:
    def __init__(self, index: int, params: np.ndarray):
        self.index = index
        self.params = params

    def __repr__(self) -> str:
        return "<OptimisationStep index={} params={}>".format(self.index, self.params)
