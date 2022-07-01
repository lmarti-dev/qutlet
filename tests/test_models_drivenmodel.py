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

@pytest.mark.parametrize(
    "models, drives, H0",
    [
        (
            [
                Ising(  "GridQubit",
                        [2,2],
                        1*np.ones((2-1,2)),
                        1*np.ones((2,2-1)),
                        0*np.ones((2,2)),
                        "X" ),
                Ising(  "GridQubit",
                        [2,2],
                        0*np.ones((2-1,2)),
                        0*np.ones((2,2-1)),
                        1*np.ones((2,2)),
                        "Z" ),
            ],
            [
                lambda t: 1,
                lambda t: t
            ],
             Ising(  "GridQubit",
                        [2,2],
                        1*np.ones((2-1,2)),
                        1*np.ones((2,2-1)),
                        0*np.ones((2,2)),
                        "X" )._hamiltonian
         ),
         (
            [
                Ising(  "GridQubit",
                        [2,2],
                        1*np.ones((2-1,2)),
                        1*np.ones((2,2-1)),
                        0*np.ones((2,2)),
                        "X" ),
                Ising(  "GridQubit",
                        [2,2],
                        0*np.ones((2-1,2)),
                        0*np.ones((2,2-1)),
                        1*np.ones((2,2)),
                        "X" ),
            ],
            [
                lambda t: sympy.sin(t),
                lambda t: t
            ],
             Ising(  "GridQubit",
                        [2,2],
                        0*np.ones((2-1,2)),
                        0*np.ones((2,2-1)),
                        0*np.ones((2,2)),
                        "X" )._hamiltonian
         ),
         (
            [
                Ising(  "GridQubit",
                        [2,2],
                        1*np.ones((2-1,2)),
                        1*np.ones((2,2-1)),
                        0*np.ones((2,2)),
                        "X" ),
                Ising(  "GridQubit",
                        [2,2],
                        0*np.ones((2-1,2)),
                        0*np.ones((2,2-1)),
                        1*np.ones((2,2)),
                        "Z" ),
            ],
            [
                lambda t: 1,
                lambda t: 1
            ],
             Ising(  "GridQubit",
                        [2,2],
                        1*np.ones((2-1,2)),
                        1*np.ones((2,2-1)),
                        1*np.ones((2,2)),
                        "Z" )._hamiltonian
         ),
    ],
)        
def test_H0(models, drives, H0):
    driven_model = DrivenModel( models, 
                                drives)
    #print("driven_model.H0: {}\nH0:{}".format(driven_model.H0 , H0))
    #print(driven_model.Vjs[0])
    assert driven_model.H0 == H0

@pytest.mark.parametrize(
    "models, drives, Heff",
    [
        (
            [
                Ising(  "GridQubit",
                        [2,2],
                        1*np.ones((2-1,2)),
                        1*np.ones((2,2-1)),
                        0*np.ones((2,2)),
                        "X" ),
                Ising(  "GridQubit",
                        [2,2],
                        0*np.ones((2-1,2)),
                        0*np.ones((2,2-1)),
                        1*np.ones((2,2)),
                        "Z" ),
            ],
            [
                lambda t: 1,
                lambda t: sympy.sin(10*t),
            ],
            Ising(  "GridQubit",
                        [2,2],
                        1*np.ones((2-1,2)),
                        1*np.ones((2,2-1)),
                        0*np.ones((2,2)),
                        "X" )._hamiltonian
         ),
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
                lambda t: 1,
                lambda t: sympy.sin(10*t),
            ],
            cirq.PauliSum.from_pauli_strings( \
                -(1-2*0.01)*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1)) \
                -2*0.01*cirq.Y.on(cirq.GridQubit(0,0))*cirq.Y.on(cirq.GridQubit(0,1)))
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
                lambda t: 1,
                lambda t: sympy.sin(100*t),
            ],
            cirq.PauliSum.from_pauli_strings( \
                -(1-2*0.01)*cirq.Z.on(cirq.GridQubit(0,0))*cirq.Z.on(cirq.GridQubit(0,1)) \
                -2*0.01*cirq.Y.on(cirq.GridQubit(0,0))*cirq.Y.on(cirq.GridQubit(0,1)))
         ),
    ],
)    
def test_Heff(models, drives, Heff):
    # For 1 TFIM:
    # Heff =  -J\left(\sum_{k=1}^n (1- 2\frac{h^2}{\omega^2}) Z_kZ_{k+1} +2 \frac{h^2}{\omega^2} Y_kY_{k+1}) \right)
    driven_model = DrivenModel( models, 
                                drives)
    if driven_model.Heff == Heff:
        assert True
    else:
        print(Heff.matrix())
        print(driven_model.Heff.matrix())
        np.testing.assert_allclose( Heff.matrix() - driven_model.Heff.matrix(),
                                    np.zeros((2**np.product(driven_model.n), 2**np.product(driven_model.n))),rtol=1e-15,atol=1e-15)
    
@pytest.mark.parametrize(
    "models, drives, integrated_Vt",
    [
        (
            [
                Ising(  "GridQubit",
                        [2,2],
                        1*np.ones((2-1,2)),
                        1*np.ones((2,2-1)),
                        0*np.ones((2,2)),
                        "X" ),
                Ising(  "GridQubit",
                        [2,2],
                        0*np.ones((2-1,2)),
                        0*np.ones((2,2-1)),
                        #np.ones((2,2)),
                        2*(np.random.rand(2,2)- 0.5),
                        "X" ),
            ],
            [
                lambda t: 1,
                lambda t: sympy.cos(10*t),
            ],
            lambda t: (1/10)*np.sin(10*t),
         ),
    ],
)    
def test_Kt(models, drives, integrated_Vt):
    # make t random, make hamiltian random
    # give correctly integrated v(t) w.o. Hamiltonian to test
    driven_model = DrivenModel( models, 
                                drives)
    t = np.random.random_sample()
    H_V = (sum([models[i]._hamiltonian for i in range(len(models))]) - driven_model.H0)
    print(H_V.matrix())
    print(integrated_Vt(t))
    print(driven_model.K(t))
    print(driven_model.K(t) - integrated_Vt(t) * H_V)
    if driven_model.K(t) == integrated_Vt(t) * H_V:
        assert True
    else:
        np.testing.assert_allclose( driven_model.K(t).matrix() - integrated_Vt(t) * H_V.matrix(),
                                    np.zeros((2**np.product(driven_model.n), 2**np.product(driven_model.n))),rtol=1e-15,atol=1e-15)

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