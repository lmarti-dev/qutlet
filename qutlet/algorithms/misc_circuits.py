import cirq
from cirq.circuits import InsertStrategy

import openfermion as of
import sympy
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.models.fermionicModel import FermionicModel
from fauvqe.models.qubitModel import QubitModel
from fauvqe.utilities.generic import flatten
import numpy as np
import itertools

"""This file defines a standard way to smoothly implement new simple "circuit" ansaetze with our class system. 
It hinges on the generic_ansatz function which requires an ansatz function itself.
THe ansatz function should contain the description of the circuit for any number of layers
the layers part could have been left in generic_ansatz,
but I wanted to have the ability to implement non-identical layers
and not simply repeat one single layer.

"""


def generic_ansatz(model: QubitModel, layers, ansatz: callable):
    circuit = cirq.Circuit()
    symbols = []
    circuit, symbols = ansatz(model=model, symbols=symbols, layers=layers)
    model.circuit.append(circuit)
    model.circuit_param.extend(list(flatten(symbols)))
    if model.circuit_param_values is None:
        model.circuit_param_values = np.array([])
    model.circuit_param_values = np.concatenate(
        (model.circuit_param_values, np.zeros(np.size(symbols)))
    )


def brickwall_ansatz(
    model: FermionicModel, layers: int = 1, shared_layer_parameter: bool = True
):
    def ansatz(model: FermionicModel, symbols, layers):
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

    generic_ansatz(model=model, layers=layers, ansatz=ansatz)


def pyramid_ansatz(model: FermionicModel, layers=1):
    def ansatz(model: FermionicModel, symbols, layers):
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

    generic_ansatz(model=model, layers=layers, ansatz=ansatz)


def totally_connected_ansatz(
    model: FermionicModel, layers=1, spin_conserving: bool = False
):
    def ansatz(model: FermionicModel, symbols, layers):
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

    generic_ansatz(model=model, layers=layers, ansatz=ansatz)


def stair_ansatz(model: FermionicModel, layers=1):
    def ansatz(model: FermionicModel, symbols: list, layers: int):
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

    generic_ansatz(model=model, layers=layers, ansatz=ansatz)
