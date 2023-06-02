# External Imports
import pytest
import numpy as np

from typing import Tuple, Dict

# Internal Imports
from fauvqe import AbstractModel, CVaR, ExpectationValue, Ising

class MockModel(AbstractModel):
    def __init__(self):
        super().__init__("GridQubit", [1, 1])

    def copy(self):
        return MockModel()
    
    def to_json_dict(self) -> Dict:
        return {}

    @classmethod
    def from_json_dict(cls, params: Dict):
        return cls()

    def energy(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([0]), np.array([0])
    
    def _set_hamiltonian(self, reset: bool = True):
        self.hamiltonian = cirq.PauliSum()


@pytest.mark.parametrize(
    "x0, sigma, x_min, x_max, alpha, expected_value, uncertainty",
    [
        # (),
    ],
)
def test_calc_cvar_gauss(x0, sigma, x_min, x_max, alpha, expected_value, uncertainty):
    def gaussian(x, _x0, _sigma):
        a = 1 / np.sqrt(2 * np.pi * _sigma)
        e = -((x - _x0) ** 2) / (2 * _sigma ** 2)

        return a * np.exp(e)

    x_ls = np.linspace(x_min, x_max)
    y_ls = gaussian(x_ls, x0, sigma)

    cvar = CVaR._calc_cvar(x_ls, y_ls, alpha=alpha)

    assert np.abs(cvar - expected_value) <= uncertainty


@pytest.mark.parametrize(
    "probabilities, energies, alpha, expected_result, uncertainty",
    [
        ([1], [15], 1, 15, 0),
    ],
)
def test_calc_cvar(probabilities, energies, alpha, expected_result, uncertainty):
    if not isinstance(probabilities, np.ndarray):
        probabilities = np.array(probabilities)
    if not isinstance(energies, np.ndarray):
        energies = np.array(energies)
    cvar = CVaR._calc_cvar(probabilities, energies, alpha=alpha)

    assert np.abs(cvar - expected_result) <= uncertainty

@pytest.mark.parametrize(
    "model, state, alpha, expected_result, uncertainty",
    [
        (
            Ising("GridQubit", 
                  [1,2], 
                  np.ones((0, 2)), 
                  np.ones((1, 1)), 
                  np.ones((1, 2)),
                  "X"),
            np.array([np.sqrt(2)/2,0,0,np.sqrt(2)/2], dtype=np.complex64), 
            1, 
            -1/4, 
            2e-7,
         ),
         (
            Ising("GridQubit", 
                  [1,2], 
                  np.ones((0, 2)), 
                  np.ones((1, 1)), 
                  np.ones((1, 2)),
                  "X"),
            np.array([np.sqrt(2)/2,0,0,np.sqrt(2)/2], dtype=np.complex64), 
            1/2, 
            -3/4, 
            2e-7,
         ),
         (
            Ising("GridQubit", 
                  [1,2], 
                  np.ones((0, 2)), 
                  np.ones((1, 1)), 
                  np.ones((1, 2)),
                  "X"),
            np.array([1,0,0,0], dtype=np.complex64), 
            1, 
            -1/8, 
            2e-7,
         ),
         (
            Ising("GridQubit", 
                  [1,2], 
                  np.ones((0, 2)), 
                  np.ones((1, 1)), 
                  np.ones((1, 2)),
                  "X"),
            np.array([np.sqrt(2)/2,0,np.sqrt(2)/2,0], dtype=np.complex64), 
            1/2, 
            -3/4, 
            2e-7,
         ),
    ],
)
def test_evaluate_X(model, state, alpha, expected_result, uncertainty):
    #model.diagonalise("numpy")
    print(model.energy())
    
    objective = CVaR(model, alpha=alpha, field="X") 
    print(state)
    result = objective.evaluate(state)
    print(result)

    assert np.abs(result - expected_result) <= uncertainty

@pytest.mark.parametrize(
    "probabilities, energies, alpha",
    [
        ([1], [15], 1),
        ([0.1, 0.2, 0.3, 0.3, 0.2], [1, 4, 5, 6, 8, 9], 0.05),
    ],
)
def test_calc_cvar_min(probabilities, energies, alpha):
    if not isinstance(probabilities, np.ndarray):
        probabilities = np.array(probabilities)
    if not isinstance(energies, np.ndarray):
        energies = np.array(energies)
    cvar = CVaR._calc_cvar(probabilities, energies, alpha=alpha)

    assert cvar == energies[0]


@pytest.mark.parametrize(
    "alpha, field",
    [
        (0.1, "Y"),
        (0.5, "A"),
        (-0.1, "Z"),
        (1.01, "X"),
    ],
)
def test_init_error(alpha, field):
    model = MockModel()

    with pytest.raises(AssertionError):
        CVaR(model, field=field, alpha=alpha)


@pytest.mark.parametrize(
    "alpha, field",
    [
        (0.1, "Z"),
        (0.5, "X"),
        (0.1, "X"),
        (0.5, "Z"),
    ],
)
def test_cvar_repr(alpha, field):
    model = MockModel()

    cvar = CVaR(model, alpha=alpha, field=field)
    assert repr(cvar) == "<cVaR field={} alpha={}>".format(field, alpha)
