"""
module docstring
"""

import numpy as np


class OptimisationStep:
    """
    class docstring
    """

    def __init__(self, index: int, params: np.ndarray):
        self.index = index
        self.params = params

    def __repr__(self) -> str:
        return "<OptimisationStep index={} params={}>".format(self.index, self.params)
