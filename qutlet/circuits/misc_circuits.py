import cirq
from cirq.circuits import InsertStrategy

import openfermion as of
import sympy
from qutlet.models.fermionicModel import FermionicModel
from qutlet.models.qubitModel import QubitModel
from qutlet.utilities import flatten, default_value_handler
import numpy as np
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
    circuit, symbols = circuit(model=model, symbols=symbols, layers=layers)
    symbols = list(flatten(symbols))
    ansatz = Ansatz(circuit=circuit, param_resolver=symbols)
    return ansatz


def brickwall_circuit(
    model: FermionicModel, layers: int = 1, shared_layer_parameter: bool = True
):
    def circuit(model: FermionicModel, symbols, layers):
        qubits = model.flattened_qubits
        circuit = cirq.Circuit()

        for layer in range(layers):
            layer_symbols = []
            if shared_layer_parameter:
                layer_symbols.append(sympy.Symbol("theta_{}".format(layer)))
                layer_symbols.append(sympy.Symbol("phi_{}".format(layer)))
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

    circuit_ansatz(model=model, layers=layers, circuit=circuit)


def pyramid_circuit(model: FermionicModel, layers=1):
    def circuit(model: FermionicModel, symbols, layers):
        qubits = model.flattened_qubits
        circuit = cirq.Circuit()
        for layer in range(layers):
            layer_symbols = []
            layer_symbols.append(sympy.Symbol("theta_{}".format(layer)))
            layer_symbols.append(sympy.Symbol("phi_{}".format(layer)))
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

    circuit_ansatz(model=model, layers=layers, circuit=circuit)


def totally_connected_circuit(
    model: FermionicModel, layers=1, spin_conserving: bool = False
):
    def circuit(model: FermionicModel, symbols, layers):
        qubits = model.flattened_qubits
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

    circuit_ansatz(model=model, layers=layers, circuit=circuit)


def stair_circuit(model: FermionicModel, layers=1):
    def circuit(model: FermionicModel, symbols: list, layers: int):
        qubits = model.flattened_qubits
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

    circuit_ansatz(model=model, layers=layers, circuit=circuit)
