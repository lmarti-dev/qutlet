# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import Ising, HeisenbergFC, CoolingNA, Cooling1A, AbstractExpectationValue

@pytest.mark.parametrize(
    "n, boundaries, field, options, solution",
    [
        (
            [2, 1], 
            [1, 1], 
            "X",
            {   "K":1,
                "time_steps":1,
            },
            cirq.Circuit(
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Y.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.Y.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(3,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Z.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.Z.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(3,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0))),
        ),
        (
            [2, 1], 
            [1, 1], 
            "X",
            {   "K":1,
                "time_steps":2,
            },
            cirq.Circuit(
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Y.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.Y.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(3,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Z.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.Z.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(3,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Y.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.Y.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(3,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Z.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.Z.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(3,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0))),
        ),
        (
            [2, 1], 
            [1, 1], 
            "X",
            {   "K":2,
                "time_steps":1,
                 "m":1,
                 "emax":2,
                 "emin":0.1
            },
            cirq.Circuit(
                        (cirq.ZZ**(-0.665)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-0.665)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(-0.665)).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        (cirq.X**(-0.665)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-0.665)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.665)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-0.665)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        (cirq.ZZ**(-0.707)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-0.707)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.XX**(-0.707)).on(cirq.GridQubit(1, 0), cirq.GridQubit(3, 0)),
                        (cirq.X**(-0.707)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-0.707)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.707)).on(cirq.GridQubit(1, 0)),
                        (cirq.Z**(-0.707)).on(cirq.GridQubit(3, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        cirq.reset(cirq.GridQubit(3,0)),
                        ),
        ),
    ]
)
def test_set_circuit_na(n, boundaries, field, options, solution):
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])), field)
    m_anc = Ising("GridQubit", n, np.zeros((n[0], n[1])), np.zeros((n[0], n[1])), np.ones((n[0], n[1])), "Z")
    j_int = np.ones((1, n[0], n[1]))
    
    model = CoolingNA(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("cooling", options)
    model.set_simulator("dm", dtype=np.complex128)
    print("model.circuit: \n{}".format(cirq.align_left(model.circuit)))
    print("solution: \n{}".format(cirq.align_left(solution)))
    initial = 2**(-2*np.size(n)) * np.eye(2**(2*np.size(n))).astype(np.complex128)
    
    wf = model.simulator.simulate(model.circuit, initial_state=initial).final_density_matrix
    sol_wf = model.simulator.simulate(solution, initial_state=initial).final_density_matrix
    
    assert np.linalg.norm(wf - sol_wf) < 1e-7
    
    #energy = AbstractExpectationValue(m_sys)
    #energy_initial = energy.evaluate(model.cooling.ptrace(initial, range(np.size(n), 2*np.size(n))))
    #energy_final = energy.evaluate(model.cooling.ptrace(wf, range(np.size(n), 2*np.size(n))))
    #print(energy_initial, "vs.", energy_final)
    #assert energy_initial > energy_final

@pytest.mark.parametrize(
    "n, boundaries, field, options, solution",
    [
        (
            [2, 1], 
            [1, 1], 
            "X",
            {   "K":1,
                "time_steps":1,
            },
            cirq.Circuit(
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Y.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Y.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Z.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Z.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        ),
        ),
        (
            [2, 1], 
            [1, 1], 
            "X",
            {   "K":1,
                "time_steps":2,
                 "m":1
            },
            2*cirq.Circuit(
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-13/15)).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Y.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Y.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Z.on(cirq.GridQubit(0,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-13/15)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.Z.on(cirq.GridQubit(1,0))*cirq.X.on(cirq.GridQubit(2,0)))**(-13/15),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-13/15)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-13/15)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0))
                ),
        ),
        (
            [2, 1], 
            [1, 1], 
            "X",
            {   "K":2,
                "time_steps":1,
                 "m":1,
                 "emax":2,
                 "emin":0.1
            },
            cirq.Circuit(
                        (cirq.ZZ**(-0.665)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-0.665)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.665)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-0.665)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.665)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-0.665)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-0.665)).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.665)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-0.665)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.665)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-0.707)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-0.707)).on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.707)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-0.707)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.707)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        (cirq.ZZ**(-0.707)).on(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
                        (cirq.XX**(-0.707)).on(cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.707)).on(cirq.GridQubit(0, 0)),
                        (cirq.Z**(-0.707)).on(cirq.GridQubit(2, 0)),
                        (cirq.X**(-0.707)).on(cirq.GridQubit(1, 0)),
                        cirq.reset(cirq.GridQubit(2,0)),
                        ),
        ),
    ]
)
def test_set_circuit_1a(n, boundaries, field, options, solution):
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])), field)
    m_anc = Ising("GridQubit", [1, n[1]], np.zeros((1, n[1])), np.zeros((1, n[1])), np.ones((1, n[1])), "Z")
    j_int = np.ones((1, n[0], n[1]))
    
    model = Cooling1A(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("cooling", options)
    model.set_simulator("dm", dtype=np.complex128)
    print("model.circuit: \n{}".format(cirq.align_left(model.circuit)))
    print("solution: \n{}".format(cirq.align_left(solution)))
    initial = 2**(-(np.size(n)+n[1])) * np.eye(2**(np.size(n) + n[1])).astype(np.complex128)
    wf = model.simulator.simulate(model.circuit, initial_state=initial).final_density_matrix
    sol_wf = model.simulator.simulate(solution, initial_state=initial).final_density_matrix
    
    assert np.linalg.norm(wf - sol_wf) < 1e-7
    
    #energy = AbstractExpectationValue(m_sys)
    #energy_initial = energy.evaluate(model.cooling.ptrace(initial, range(np.size(n), np.size(n)+1)))
    #energy_final = energy.evaluate(model.cooling.ptrace(wf, range(np.size(n), np.size(n)+1)))
    #print(energy_initial, "vs.", energy_final)
    #assert energy_initial > energy_final 

def test_set_K():
    n=[2,1]
    boundaries=[1,1]
    m_sys = Ising("GridQubit", n, np.ones((n[0]-boundaries[0], n[1])), np.ones((n[0], n[1]-boundaries[1])), np.ones((n[0], n[1])), "X")
    m_anc = Ising("GridQubit", [1, n[1]], np.zeros((1, n[1])), np.zeros((1, n[1])), np.ones((1, n[1])), "Z")
    j_int = np.ones((1, n[0], n[1]))
    
    model = Cooling1A(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    model.set_circuit("cooling")
    model.cooling.set_K(model, 2)
    assert model.cooling.options["K"] == 2

@pytest.mark.parametrize(
    "rho, ind, solution",
    [
        (
            1/4*np.eye(4),
            1,
            1/2*np.eye(2)
        ),
        (
            0.25*np.array(
                        [[1, 1, 0, 0], 
                        [1, 1, 0, 0], 
                        [0, 0, 1, 1], 
                        [0, 0, 1, 1]]
            ),
            0,
            0.5*np.array(
                        [[1, 1], 
                        [1, 1]]
            ),
        ),
        (
            1/8*np.eye(8),
            range(1, 3),
            1/2*np.eye(2)
        ),
    ]
)
def test_ptrace(rho, ind, solution):
    n=[2,1]
    boundaries=[1,1]
    m_sys = Ising("GridQubit", n)
    m_anc = Ising("GridQubit", [1, n[1]])
    j_int = np.ones((1, n[0], n[1]))
    model = Cooling1A(
                    m_sys,
                    m_anc,
                    [lambda q1, q2: cirq.X(q1)*cirq.X(q2)],
                    j_int
    )
    assert np.linalg.norm(model.cooling.ptrace(rho, ind) - solution) < 1e-7

def test_errors():
    n=[1, 2]
    m_sys = HeisenbergFC("GridQubit", n, np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])), np.ones((n[0], n[1], n[0], n[1])) )
    with pytest.raises(AssertionError):
        m_sys.set_circuit("cooling", {'K': 1})
    with pytest.raises(AssertionError):
        m_sys.set_circuit("cooling", {'K': 2})