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

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified={}
    for o in shared_keys:
        if not isinstance(d1[o], type(None)) and not isinstance(d2[o], type(None)):
            try:
                if d1[o] != d2[o]:
                    modified.update({o : (d1[o], d2[o])})
                    print("o:\t{}\nd1[o]:\t{}\nd2[o]:\t{}\n".format(o, d1[o], d2[o]))
            except:
                if d1[o].any() != d2[o].any():
                    modified.update({o : (d1[o], d2[o])})
                    print("o:\t{}\nd1[o]:\t{}\nd2[o]:\t{}\n".format(o, d1[o], d2[o]))

    #modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o].any() != d2[o].any()}
    #same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified

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

@pytest.mark.parametrize(
    "models, drives, ground_truth_Kt",
    [
        (
            [
                Ising(  "GridQubit",
                        [1,3],
                        1*np.ones((1-1,3)),
                        1*np.ones((1,3-1)),
                        0*np.ones((1,3)),
                        "X" ),
                Ising(  "GridQubit",
                        [1,3],
                        0*np.ones((1-1,3)),
                        0*np.ones((1,3-1)),
                        np.ones((1,3)),
                        "X" ),
            ],
            [
                lambda t: 1,
                lambda t: sympy.cos(10*t),
            ],
            lambda t: -(1/10)*np.sin(10*t)*cirq.PauliSum.from_pauli_strings([cirq.X.on(cirq.GridQubit(0,i)) for i in range(3)])
                      -(1*1/(10**2))*2*np.sin(10*t)*( \
                            cirq.PauliSum.from_pauli_strings([cirq.Z.on(cirq.GridQubit(0,i))*cirq.Y.on(cirq.GridQubit(0,i+1)) for i in range(3-1)]) \
                            + cirq.PauliSum.from_pauli_strings([cirq.Y.on(cirq.GridQubit(0,i))*cirq.Z.on(cirq.GridQubit(0,i+1)) for i in range(3-1)]))
         ),
         (
            [
                Ising(  "GridQubit",
                        [1,3],
                        1*np.ones((1-1,3)),
                        1*np.ones((1,3-1)),
                        0*np.ones((1,3)),
                        "X" ),
                Ising(  "GridQubit",
                        [1,3],
                        0*np.ones((1-1,3)),
                        0*np.ones((1,3-1)),
                        3.14*np.ones((1,3)),
                        "X" ),
            ],
            [
                lambda t: 1,
                lambda t: sympy.cos(10*t),
            ],
            lambda t: -(3.14/10)*np.sin(10*t)*cirq.PauliSum.from_pauli_strings([cirq.X.on(cirq.GridQubit(0,i)) for i in range(3)])
                      -(3.14/(10**2))*2*np.sin(10*t)*( \
                            cirq.PauliSum.from_pauli_strings([cirq.Z.on(cirq.GridQubit(0,i))*cirq.Y.on(cirq.GridQubit(0,i+1)) for i in range(3-1)]) \
                            + cirq.PauliSum.from_pauli_strings([cirq.Y.on(cirq.GridQubit(0,i))*cirq.Z.on(cirq.GridQubit(0,i+1)) for i in range(3-1)]))
         ),
         (
            [
                Ising(  "GridQubit",
                        [1,3],
                        1.17*np.ones((1-1,3)),
                        1.17*np.ones((1,3-1)),
                        0*np.ones((1,3)),
                        "X" ),
                Ising(  "GridQubit",
                        [1,3],
                        0*np.ones((1-1,3)),
                        0*np.ones((1,3-1)),
                        np.ones((1,3)),
                        "X" ),
            ],
            [
                lambda t: 1,
                lambda t: sympy.cos(10*t),
            ],
            lambda t: -(1/10)*np.sin(10*t)*cirq.PauliSum.from_pauli_strings([cirq.X.on(cirq.GridQubit(0,i)) for i in range(3)])
                      -(1.17/(10**2))*2*np.sin(10*t)*( \
                            cirq.PauliSum.from_pauli_strings([cirq.Z.on(cirq.GridQubit(0,i))*cirq.Y.on(cirq.GridQubit(0,i+1)) for i in range(3-1)]) \
                            + cirq.PauliSum.from_pauli_strings([cirq.Y.on(cirq.GridQubit(0,i))*cirq.Z.on(cirq.GridQubit(0,i+1)) for i in range(3-1)]))
         ),
         (
            [
                Ising(  "GridQubit",
                        [1,3],
                        1.17*np.ones((1-1,3)),
                        1.17*np.ones((1,3-1)),
                        0*np.ones((1,3)),
                        "X" ),
                Ising(  "GridQubit",
                        [1,3],
                        0*np.ones((1-1,3)),
                        0*np.ones((1,3-1)),
                        3.14*np.ones((1,3)),
                        "X" ),
            ],
            [
                lambda t: 1,
                lambda t: sympy.cos(10*t),
            ],
            lambda t: -(3.14/10)*np.sin(10*t)*cirq.PauliSum.from_pauli_strings([cirq.X.on(cirq.GridQubit(0,i)) for i in range(3)])
                      -(3.14*1.17/(10**2))*2*np.sin(10*t)*( \
                            cirq.PauliSum.from_pauli_strings([cirq.Z.on(cirq.GridQubit(0,i))*cirq.Y.on(cirq.GridQubit(0,i+1)) for i in range(3-1)]) \
                            + cirq.PauliSum.from_pauli_strings([cirq.Y.on(cirq.GridQubit(0,i))*cirq.Z.on(cirq.GridQubit(0,i+1)) for i in range(3-1)]))
         ),
    ],
)    
def test_Kt_order_2(models, drives, ground_truth_Kt):
    # make t random, make hamiltian random
    # give correctly integrated v(t) w.o. Hamiltonian to test
    driven_model = DrivenModel( models, 
                                drives)
    t = np.random.random_sample()
    
    print(driven_model.K(t, order=2))
    print(ground_truth_Kt(t))
    if driven_model.K(t, order=2) == ground_truth_Kt(t):
        assert True
    else:
        np.testing.assert_allclose( driven_model.K(t, order=2).matrix() - ground_truth_Kt(t).matrix(),
                                    np.zeros((2**np.product(driven_model.n), 2**np.product(driven_model.n))),rtol=1e-15,atol=1e-15)

@pytest.mark.parametrize(
    "models, drives, T",
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
            2*sympy.pi/10
         ),
         # Note: only difference to previous tests is 
         # sympy.cos(10*t) -> sympy.sin(10*t)
         # Seems like the current implementation has somehow a problem with using sin
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
                lambda t: sympy.sin(10*t),
            ],
            2*sympy.pi/10
         ),
         (
            [
                Ising(  "GridQubit",
                        [2,3],
                        1*np.ones((2-1,3)),
                        1*np.ones((2,3-1)),
                        0*np.ones((2,3)),
                        "X" ),
                Ising(  "GridQubit",
                        [2,3],
                        0*np.ones((2-1,3)),
                        0*np.ones((2,3-1)),
                        #np.ones((2,3)),
                        2*(np.random.rand(2,3)- 0.5),
                        "X" ),
            ],
            [
                lambda t: 1,
                lambda t: (sympy.cos(10*t) + sympy.cos(20*t)),
            ],
            2*sympy.pi/10
         ),
         # Note: only difference to previous tests is 
         # sympy.cos(20*t) -> sympy.sin(20*t)
         #(
         #   [
         #       Ising(  "GridQubit",
         #               [2,3],
         #               1*np.ones((2-1,3)),
         #               1*np.ones((2,3-1)),
         #               0*np.ones((2,3)),
         #               "X" ),
         #       Ising(  "GridQubit",
         #               [2,3],
         #               0*np.ones((2-1,3)),
         #               0*np.ones((2,3-1)),
         #               #np.ones((2,3)),
         #               2*(np.random.rand(2,3)- 0.5),
         #               "X" ),
         #   ],
         #   [
         #       lambda t: 1,
         #       lambda t: (sympy.cos(10*t) + sympy.sin(20*t)),
         #   ],
         #   2*sympy.pi/10
         #),
    ],
)    
def test_Vt(models, drives, T):
    # make t random, make hamiltian random
    # give correctly integrated v(t) w.o. Hamiltonian to test
    driven_model = DrivenModel( models, 
                                drives,
                                T = T)
    t = np.random.random_sample()

    print("t: {}\tdrive[1](t): {}".format(t, drives[1](t)))
    print(driven_model.V(t))
    print(drives[1](t) * models[1]._hamiltonian)

    if driven_model.V(t) == drives[1](t) * models[1]._hamiltonian:
        assert True
    else:
        np.testing.assert_allclose( driven_model.V(t).matrix() - drives[1](t) * models[1]._hamiltonian.matrix(),
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
            "< DrivenModel\nModel 0 < Ising, Hamiltonian=-1.000*Z(q(0, 0))*Z(q(0, 1)) >\nDrive 0 f(t) = 1\nModel 1 < Ising, Hamiltonian=-1.000*X(q(0, 0))-1.000*X(q(0, 1)) >\nDrive 1 f(t) = sin(10*t) >"
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
            "< DrivenModel\nModel 0 < Ising, Hamiltonian=-1.000*Z(q(0, 0))*Z(q(0, 1))-0.500*X(q(0, 0))-0.500*X(q(0, 1)) >\nDrive 0  >"
        ),
    ],
)
def test__repr(models, drives, drive_names, true_repr):
    for i in range(len(drives)): drives[i].__name__ = drive_names[i]
    driven_model = DrivenModel( models, 
                                drives)
    print("repr(driven_model):\n{}".format(repr(driven_model)) )
    print("true_repr:\n{}".format(true_repr) )
    assert repr(driven_model) == true_repr


@pytest.mark.parametrize(
    "models, drives, drive_names, T, t0, t, j_max",
    [
        (
            [
                Ising(  "GridQubit",
                        [2,2],
                        np.around(2*(np.random.rand(2-1,2)- 0.5), decimals=3) ,
                        np.around(2*(np.random.rand(2,2-1)- 0.5), decimals=3) ,
                        np.around(2*(np.random.rand(2,2)- 0.5), decimals=3) ,
                        "Z" ),
                Ising(  "GridQubit",
                        [2,2],
                        np.around(2*(np.random.rand(2-1,2)- 0.5), decimals=3) ,
                        np.around(2*(np.random.rand(2,2-1)- 0.5), decimals=3) ,
                        np.around(2*(np.random.rand(2,2)- 0.5), decimals=3) ,
                        "X" ),
            ],
            [
                lambda t:   1,
                lambda t:   sympy.sin((10)*t)
            ],
             [
                "f(t) = 1",
                "f(t) = sin(10*t)"
            ],
            5,
            3,
            7,
            9
        ),
    ],
)
@pytest.mark.higheffort
def test_copy(models, drives, drive_names, T, t0, t, j_max):
    for i in range(len(drives)): drives[i].__name__ = drive_names[i]
    driven_model = DrivenModel( models, 
                                drives,
                                T,t0,t,j_max)
    driven_model2 = DrivenModel( models, 
                                drives,
                                T,t0,t,j_max)
    copy_driven_model = driven_model.copy()
    del driven_model

    added, removed, modified = dict_compare(driven_model2.__dict__, copy_driven_model.__dict__)
    print("Added:\n{}\nRemoved:\n{}\nModified\n{}\n".format(added, removed, modified))
    assert copy_driven_model == driven_model2

@pytest.mark.parametrize(
    "models, drives, drive_names, T, t0, t, j_max",
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
                lambda t: sympy.cos((10)*t)
            ],  
            [
                "f(t) = 1",
                "f(t) = cos(10*t)"
            ],
            5,
            3,
            7,
            9,
        ),
        (
            [
                Ising(  "GridQubit",
                        [2,2],
                        np.around(2*(np.random.rand(2-1,2)- 0.5), decimals=3) ,
                        np.around(2*(np.random.rand(2,2-1)- 0.5), decimals=3) ,
                        np.around(2*(np.random.rand(2,2)- 0.5), decimals=3) ,
                        "Z" ),
                Ising(  "GridQubit",
                        [2,2],
                        np.around(2*(np.random.rand(2-1,2)- 0.5), decimals=3) ,
                        np.around(2*(np.random.rand(2,2-1)- 0.5), decimals=3) ,
                        np.around(2*(np.random.rand(2,2)- 0.5), decimals=3) ,
                        "X" ),
            ],
            [
                lambda t:   1,
                lambda t:   sympy.sin((10)*t)
            ],
             [
                "f(t) = 1",
                "f(t) = sin(10*t)"
            ],
            5,
            3,
            7,
            9,
        ),
    ]
) 
def test_json(models, drives, drive_names, T, t0, t, j_max):
    for i in range(len(drives)): drives[i].__name__ = drive_names[i]
    driven_model = DrivenModel( models, 
                                drives,
                                T,t0,t,j_max)
    json = driven_model.to_json_dict()
    
    driven_model2 = DrivenModel.from_json_dict(json)

    added, removed, modified = dict_compare(driven_model.__dict__, driven_model2.__dict__)
    print("Added:\n{}\nRemoved:\n{}\nModified\n{}\n".format(added, removed, modified))

    # Comparing driven_models sometimes fails due to numerical errors instabilities in Vjs
    assert (driven_model == driven_model2)

@pytest.mark.parametrize(
    "driven_model, non_driven_model",
    [
        (
            DrivenModel([Ising(  "GridQubit", 
                                    [3,2], 
                                    1 * np.ones((2, 2)), 
                                    1 * np.ones((3, 1)), 
                                    0 * np.ones((3, 2)),
                                    "X"),
                        Ising(  "GridQubit", 
                                    [3,2], 
                                    0 * np.ones((2, 2)), 
                                    0 * np.ones((3, 1)), 
                                    1 * np.ones((3, 2)),
                                    "X")],
                        [lambda t : 1, lambda t : 1]),
            Ising(  "GridQubit", 
                        [3,2], 
                        1 * np.ones((2, 2)), 
                        1 * np.ones((3, 1)), 
                        1 * np.ones((3, 2)),
                        "X")
        ),
        (
            DrivenModel([Ising(  "GridQubit", 
                                    [1,2], 
                                    1 * np.ones((0, 2)), 
                                    1 * np.ones((1, 1)), 
                                    0 * np.ones((1, 2)),
                                    "X"),
                        Ising(  "GridQubit", 
                                    [1,2], 
                                    0 * np.ones((0, 2)), 
                                    0 * np.ones((1, 1)), 
                                    1 * np.ones((1, 2)),
                                    "X")],
                        [lambda t : 1, lambda t : 2*t]),
            Ising(  "GridQubit", 
                        [1,2], 
                        1 * np.ones((0, 2)), 
                        1 * np.ones((1, 1)), 
                        2 * np.ones((1, 2)),
                        "X")
        ),
    ],
)
def test_hamiltonian(driven_model, non_driven_model):
    #print(driven_model.__dict__)
    #print("non_driven_model.hamiltonian: \t{}\ndriven_model.hamiltonian(t=1):\t{}".format(non_driven_model.hamiltonian(), driven_model.hamiltonian(t=1)))
    assert non_driven_model.hamiltonian() == driven_model.hamiltonian(t=1)

@pytest.mark.parametrize(
    "driven_model, non_driven_model, m,q,tf",
    [
        (
            DrivenModel([Ising(  "GridQubit", 
                                    [3,2], 
                                    1 * np.ones((2, 2)), 
                                    1 * np.ones((3, 1)), 
                                    0 * np.ones((3, 2)),
                                    "X"),
                        Ising(  "GridQubit", 
                                    [3,2], 
                                    0 * np.ones((2, 2)), 
                                    0 * np.ones((3, 1)), 
                                    1 * np.ones((3, 2)),
                                    "X")],
                        [lambda t : 1, lambda t : 1]),
            Ising(  "GridQubit", 
                        [3,2], 
                        1 * np.ones((2, 2)), 
                        1 * np.ones((3, 1)), 
                        1 * np.ones((3, 2)),
                        "X"),
            3,
            1,
            1.352
        ),
        (
            DrivenModel([Ising(  "GridQubit", 
                                    [1,2], 
                                    1 * np.ones((0, 2)), 
                                    1 * np.ones((1, 1)), 
                                    0 * np.ones((1, 2)),
                                    "X"),
                        Ising(  "GridQubit", 
                                    [1,2], 
                                    0 * np.ones((0, 2)), 
                                    0 * np.ones((1, 1)), 
                                    1 * np.ones((1, 2)),
                                    "X")],
                        [lambda t : 1, lambda t : 1.352]),
            Ising(  "GridQubit", 
                        [1,2], 
                        1 * np.ones((0, 2)), 
                        1 * np.ones((1, 1)), 
                        1.352 * np.ones((1, 2)),
                        "X"),
            3,
            2,
            1.352
        ),
    ],
)
def test_trotter(driven_model, non_driven_model, m,q,tf):
    driven_model.set_circuit("trotter",
                                {"trotter_number": m,
                                "trotter_order": q,
                                "tf": tf})
    non_driven_model.set_circuit("trotter",
                                {"trotter_number": m,
                                "trotter_order": q,
                                "tf": tf})
    print("driven_model.circuit:\n{}\n\nnon_driven_model.circuit:\n{}".format(driven_model.circuit, non_driven_model.circuit))
    assert driven_model.circuit == non_driven_model.circuit

#Add more examples:
@pytest.mark.parametrize(
    "models, drives, qalgorithm, options, final_circuit",
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
            "basics",
            {"start": "hadamard"},
            cirq.Circuit(cirq.H.on(cirq.GridQubit(0,0)),cirq.H.on(cirq.GridQubit(0,1)))
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
                lambda t:  1,
                lambda t: sympy.sin((10)*t)
            ],  
            "hea",
            {"p": 2, "parametrisation" : 'layerwise'},
            cirq.Circuit(cirq.PhasedXZGate(x_exponent=sympy.Symbol('x0'), z_exponent=sympy.Symbol('z0'), axis_phase_exponent=sympy.Symbol('a0')).on(cirq.GridQubit(0,0)),
                         cirq.PhasedXZGate(x_exponent=sympy.Symbol('x0'), z_exponent=sympy.Symbol('z0'), axis_phase_exponent=sympy.Symbol('a0')).on(cirq.GridQubit(0,1)),
                         cirq.FSimGate(phi=sympy.Symbol('phi0_1'), theta=sympy.Symbol('theta0_1')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),
                         cirq.PhasedXZGate(x_exponent=sympy.Symbol('x1'), z_exponent=sympy.Symbol('z1'), axis_phase_exponent=sympy.Symbol('a1')).on(cirq.GridQubit(0,0)),
                         cirq.PhasedXZGate(x_exponent=sympy.Symbol('x1'), z_exponent=sympy.Symbol('z1'), axis_phase_exponent=sympy.Symbol('a1')).on(cirq.GridQubit(0,1)),
                         cirq.FSimGate(phi=sympy.Symbol('phi1_1'), theta=sympy.Symbol('theta1_1')).on(cirq.GridQubit(0,0),cirq.GridQubit(0,1)),)
        ),
    ]
) 
def test_set_circuit(models, drives, qalgorithm, options, final_circuit):
    model = DrivenModel( models, drives)
    model.set_circuit(qalgorithm, options)

    print(model.circuit )
    assert model.circuit == final_circuit

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
@pytest.mark.parametrize(
    "models, drives",
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
        ),
    ]
)        
def test_Kt_asserts(models, drives):
    driven_model = DrivenModel( models, 
                                    drives)
    with pytest.raises(NotImplementedError):
        tmp= driven_model.K(t=1,order=3)

@pytest.mark.parametrize(
    "models, drives",
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
        ),
    ]
)        
def test_set_circuit_asserts(models, drives):
    driven_model = DrivenModel( models, 
                                    drives)
    with pytest.raises(NotImplementedError):
        driven_model.set_circuit("magic_algorithm")