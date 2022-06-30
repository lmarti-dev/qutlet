"""
    This file tests DrivenModel(AbstractModel)
"""
import imp
import cirq
import numpy as np
import pytest
import sympy

from fauvqe.models.drivenmodel import DrivenModel
from fauvqe.models.ising import Ising
from fauvqe.models.isingxy import IsingXY

@pytest.mark.parametrize(
    "models, drives, T, t0, t, j_max, dv_hamiltonian",
    [
        (
            [
                Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "Z" ),
                Ising(  "GridQubit",
                        [1,2],
                        0*np.ones((1-1,2)),
                        0*np.ones((1,2-1)),
                        1*np.ones((1,2)),
                        "X" ),
            ],
            [
                lambda t:  1,
                lambda t: sympy.sin((10)*t)
            ],
            2*sympy.pi/10, 0, None, 1, cirq.PauliSum()
        ),
        (
            Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "Z" ),
            lambda t:  1,
            2*sympy.pi/10, 0, None, 1, cirq.PauliSum()
        ),
        (
            IsingXY(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "Z" ),
            lambda t: sympy.sin((10)*t),
            2*sympy.pi/10, 0, np.pi/20, 1, 
            cirq.PauliSum.from_pauli_strings(-cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1))-cirq.Y.on(cirq.GridQubit(0,0))*cirq.Y.on(cirq.GridQubit(0,1)))
        ),
        (
            [
                Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "Z" ),
                IsingXY(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "Z" ),
                Ising(  "GridQubit",
                        [1,2],
                        0*np.ones((1-1,2)),
                        0*np.ones((1,2-1)),
                        1*np.ones((1,2)),
                        "X" ),
            ],
            [
                lambda t:  1,
                lambda t: sympy.sin((10)*t),
                lambda t: sympy.sin((5)*t)
            ],
            2*sympy.pi/10, 0, None, 2, cirq.PauliSum()
        ),
    ],
)
def test_constructor(models, drives, T, t0, t, j_max, dv_hamiltonian):
    driven_model = DrivenModel( models, 
                                drives,
                                T = T,
                                t0 = t0,
                                t=t,
                                j_max = j_max)

    if not isinstance(models, list): models = [models]
    if not isinstance(drives, list): drives = [drives]

    assert driven_model.models == models
    assert driven_model.drives == drives
    assert driven_model.qubits == models[-1].qubits   
    assert driven_model._hamiltonian == dv_hamiltonian
    assert driven_model._t == t
    assert driven_model.T == T
    assert driven_model.t0 == t0
    assert driven_model.j_max == j_max
    assert all(driven_model.n == models[-1].n)
    assert driven_model.circuit == cirq.Circuit()
    assert driven_model.circuit_param == []
    assert driven_model.circuit_param_values == None

@pytest.mark.parametrize(
    "models, drives, t, equivalent_model",
    [
        (
            [
                Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "X" ),
                Ising(  "GridQubit",
                        [1,2],
                        0*np.ones((1-1,2)),
                        0*np.ones((1,2-1)),
                        1*np.ones((1,2)),
                        "X" ),
            ],
            [
                lambda t:  1,
                lambda t: t
            ],
            1,
            Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        1*np.ones((1,2)),
                        "X" ),
         ),
         (
            [
                Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "X" ),
                Ising(  "GridQubit",
                        [1,2],
                        0*np.ones((1-1,2)),
                        0*np.ones((1,2-1)),
                        1*np.ones((1,2)),
                        "X" ),
            ],
            [
                lambda t:  1,
                lambda t: t
            ],
            0,
            Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "X" ),
         ),
         (
            [
                Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "X" ),
                Ising(  "GridQubit",
                        [1,2],
                        0*np.ones((1-1,2)),
                        0*np.ones((1,2-1)),
                        1*np.ones((1,2)),
                        "Z" ),
            ],
            [
                lambda t:  1,
                lambda t: t
            ],
            0,
            Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "X" ),
         ),
         (
            [
                Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "X" ),
                Ising(  "GridQubit",
                        [1,2],
                        0*np.ones((1-1,2)),
                        0*np.ones((1,2-1)),
                        1*np.ones((1,2)),
                        "X" ),
            ],
            [
                lambda t: t,
                lambda t: t
            ],
            0.5,
            Ising(  "GridQubit",
                        [1,2],
                        0.5*np.ones((1-1,2)),
                        0.5*np.ones((1,2-1)),
                        0.5*np.ones((1,2)),
                        "X" ),
         ),
    ],
)           
def test_energy_filter(models, drives, t, equivalent_model):
    driven_model = DrivenModel( models, 
                                drives)
    assert (np.array(driven_model.energy(t)) == np.array(equivalent_model.energy())).all()

def test_H0():
    #assert driven_model.H0 == models[0]._hamiltonian
    pass

def test_Heff():
    pass

def test_Kt():
    pass

def test_Vjs():
    pass

@pytest.mark.parametrize(
    "models, drives, drive_names, true_repr",
    [
        (
            [
                Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0*np.ones((1,2)),
                        "Z" ),
                Ising(  "GridQubit",
                        [1,2],
                        0*np.ones((1-1,2)),
                        0*np.ones((1,2-1)),
                        1*np.ones((1,2)),
                        "X" ),
            ],
            [
                lambda t:  1,
                lambda t: sympy.sin((10)*t)
            ],
            [
                "f(t) = 1",
                "f(t) = sin(10*t)"
            ],
            "< DrivenModel\nModel 0 < IsingModel, Hamiltonian=-1.000*Z((0, 0))*Z((0, 1)) >\nDrive 0 f(t) = 1\nModel 1 < IsingModel, Hamiltonian=-1.000*X((0, 0))-1.000*X((0, 1)) >\nDrive 1 f(t) = sin(10*t) >"
        ),
        (
            [
                Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0.5*np.ones((1,2)),
                        "X" ),
            ],
            [
                lambda t:  1,
            ],
            [
                "",
            ],
            "< DrivenModel\nModel 0 < IsingModel, Hamiltonian=-1.000*Z((0, 0))*Z((0, 1))-0.500*X((0, 0))-0.500*X((0, 1)) >\nDrive 0  >"
        ),
    ],
)
def test__repr(models, drives, drive_names, true_repr):
    for i in range(len(drives)): drives[i].__name__ = drive_names[i]
    driven_model = DrivenModel( models, 
                                drives)
    print(repr(driven_model) )
    assert repr(driven_model) == true_repr
#####################################
#                                   #
#           Asssert tests           #
#                                   #
#####################################
@pytest.mark.parametrize(
    "models, drives",
    [
        (
            Ising(  "GridQubit",
                        [1,2],
                        1*np.ones((1-1,2)),
                        1*np.ones((1,2-1)),
                        0.5*np.ones((1,2)),
                        "X" ),
            [
                lambda t:  1,
                lambda t: sympy.sin((10)*t)
            ],
        ),
        (
            [
                    Ising(  "GridQubit",
                            [1,3],
                            1*np.ones((1-1,3)),
                            1*np.ones((1,3-1)),
                            0*np.ones((1,3)),
                            "Z" ),
                    Ising(  "GridQubit",
                            [1,2],
                            0*np.ones((1-1,2)),
                            0*np.ones((1,2-1)),
                            1*np.ones((1,2)),
                            "X" ),
                ],
                [
                    lambda t:  1,
                    lambda t: sympy.sin((10)*t)
                ],
        ),
    ],
)
def test_constructor_asserts(models, drives):
    with pytest.raises(AssertionError):
        driven_model = DrivenModel( models, 
                                    drives)