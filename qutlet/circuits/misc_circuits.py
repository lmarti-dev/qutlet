import cirq
from cirq.circuits import InsertStrategy

import sympy
from qutlet.models.fermionic_model import FermionicModel
from qutlet.models.qubit_model import QubitModel
from qutlet.utilities import flatten
import itertools
from qutlet.circuits.ansatz import Ansatz

"""This file defines a standard way to smoothly implement new simple "circuit" ansaetze with the class system. 
It hinges on the generic_circuit function which requires an circuit function itself.
THe circuit function should contain the description of the circuit for any number of layers
the layers part could have been left in generic_circuit,
but I wanted to have the ability to implement non-identical layers
and not simply repeat one single layer.

"""


def circuit_ansatz(model: QubitModel, layers, circuit: callable) -> Ansatz:
    circuit, symbols = circuit(model=model, layers=layers)
    symbols = list(flatten(symbols))
    ansatz = Ansatz(circuit=circuit, symbols=symbols)
    return ansatz


def brickwall_ansatz(
    model: FermionicModel, layers: int = 1, shared_layer_parameter: bool = True
) -> Ansatz:
    def circuit(model: FermionicModel, layers):
        symbols = []
        qubits = model.qubits
        circuit = cirq.Circuit()

        for layer in range(layers):
            layer_symbols = []
            if shared_layer_parameter:
                layer_symbols.append(sympy.Symbol(f"theta_{layer}"))
                layer_symbols.append(sympy.Symbol(f"phi_{layer}"))
            for a in range(2):
                sub_op_tree = []
                for iii in range(a, len(qubits) - 1, 2):
                    if not shared_layer_parameter:
                        layer_symbols.append(
                            sympy.Symbol("theta_{}_{}_{}".format(layer, iii, a))
                        )
                        layer_symbols.append(
                            sympy.Symbol("phi_{}_{}_{}".format(layer, iii, a))
                        )
                    qi = qubits[iii]
                    qj = qubits[iii + 1]
                    sub_op_tree.append(
                        cirq.FSimGate(
                            theta=layer_symbols[-2], phi=layer_symbols[-1]
                        ).on(qi, qj)
                    )
                circuit.append(sub_op_tree, strategy=InsertStrategy.EARLIEST)
            symbols.append(layer_symbols)
        return circuit, symbols

    return circuit_ansatz(model=model, layers=layers, circuit=circuit)


def pyramid_ansatz(model: FermionicModel, layers=1) -> Ansatz:
    def circuit(model: FermionicModel, layers):
        symbols = []
        qubits = model.qubits
        circuit = cirq.Circuit()
        for layer in range(layers):
            layer_symbols = []
            layer_symbols.append(sympy.Symbol(f"theta_{layer}"))
            layer_symbols.append(sympy.Symbol(f"phi_{layer}"))
            sub_op_tree = []
            Nq = len(qubits) - 1
            for iii in range(0, Nq):
                for jjj in range(iii, Nq - iii, 2):
                    qi = qubits[jjj]
                    qj = qubits[jjj + 1]
                    sub_op_tree.append(
                        cirq.FSimGate(
                            theta=layer_symbols[-2], phi=layer_symbols[-1]
                        ).on(qi, qj)
                    )
            circuit.append(sub_op_tree, strategy=InsertStrategy.NEW_THEN_INLINE)
            symbols.append(layer_symbols)
        return circuit, symbols

    return circuit_ansatz(model=model, layers=layers, circuit=circuit)


def totally_connected_ansatz(
    model: FermionicModel, layers=1, spin_conserving: bool = False
) -> Ansatz:
    def circuit(model: FermionicModel, layers):
        symbols = []
        qubits = model.qubits
        Nq = len(qubits)
        circuit = cirq.Circuit()
        perms = list(itertools.combinations(range(Nq), 2))
        for layer in range(layers):
            layer_symbols = []

            sub_op_tree = []
            for ni, nj in perms:
                if ni % 2 == nj % 2 or not spin_conserving:
                    layer_symbols.append(
                        sympy.Symbol(
                            "theta_{l}_{ni}_{nj}".format(l=layer, ni=ni, nj=nj)
                        )
                    )
                layer_symbols.append(
                    sympy.Symbol("phi_{l}_{ni}_{nj}".format(l=layer, ni=ni, nj=nj))
                )
                qi = qubits[ni]
                qj = qubits[nj]
                if ni % 2 != nj % 2 and spin_conserving:
                    sub_op_tree.append(
                        cirq.FSimGate(theta=0, phi=layer_symbols[-1]).on(qi, qj)
                    )
                else:
                    sub_op_tree.append(
                        cirq.FSimGate(
                            theta=layer_symbols[-2], phi=layer_symbols[-1]
                        ).on(qi, qj)
                    )
            circuit.append(sub_op_tree, strategy=InsertStrategy.EARLIEST)
            symbols.append(layer_symbols)
        return circuit, symbols

    return circuit_ansatz(model=model, layers=layers, circuit=circuit)


def ludwig_ansatz(model: FermionicModel, layers=1) -> Ansatz:
    def circuit(model: FermionicModel, layers: int):
        symbols = []
        circuit = cirq.Circuit()
        for layer in range(layers):
            sub_op_tree = []
            layer_symbols = []
            for q1 in range(model.n_qubits):
                for q2 in range(q1 + 2, model.n_qubits, 2):
                    sym_giv = sympy.Symbol(f"givs_{layer}_{q1}_{q2}")
                    layer_symbols.append(sym_giv)
                    sub_op_tree.append(
                        cirq.givens(
                            angle_rads=sym_giv,
                        ).on(model.qubits[q1], model.qubits[q2])
                    )
            for q in range(0, model.n_qubits - 1, 2):
                sym_cz = sympy.Symbol(f"cz_{layer}_{q}")
                layer_symbols.append(sym_cz)
                sub_op_tree.append(
                    cirq.CZPowGate(
                        exponent=sym_cz,
                    ).on(model.qubits[q], model.qubits[q + 1])
                )
            circuit.append(sub_op_tree, strategy=InsertStrategy.EARLIEST)
            symbols.append(layer_symbols)
        return circuit, symbols

    return circuit_ansatz(model=model, layers=layers, circuit=circuit)


def stair_ansatz(model: FermionicModel, layers=1) -> Ansatz:
    def circuit(model: FermionicModel, layers: int):
        symbols = []
        qubits = model.qubits
        Nq = len(qubits)
        circuit = cirq.Circuit()
        for layer in range(layers):
            sub_op_tree = []
            layer_symbols = []
            for nq in range(Nq - 1):
                layer_symbols.append(
                    sympy.Symbol("theta_{l}_{nq}".format(l=layer, nq=nq))
                )
                layer_symbols.append(
                    sympy.Symbol("phi_{l}_{nq}".format(l=layer, nq=nq))
                )
                qi = qubits[nq]
                qj = qubits[nq + 1]
                sub_op_tree.append(
                    cirq.FSimGate(theta=layer_symbols[-2], phi=layer_symbols[-1]).on(
                        qi, qj
                    )
                )
            circuit.append(sub_op_tree, strategy=InsertStrategy.EARLIEST)
            symbols.append(layer_symbols)
        return circuit, symbols

    return circuit_ansatz(model=model, layers=layers, circuit=circuit)
