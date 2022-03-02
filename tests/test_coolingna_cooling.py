# external imports
import pytest
import numpy as np
import cirq
import sympy

# internal imports
from fauvqe import Ising, CoolingNA

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
    ]
)
def test_set_circuit(n, boundaries, field, options, solution):
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
    #assert (cirq.align_left(model.circuit)) == (cirq.align_left(solution))