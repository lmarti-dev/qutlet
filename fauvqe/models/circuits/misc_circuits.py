import cirq
from cirq.circuits import InsertStrategy

import openfermion as of
import sympy
from fauvqe.models.fermiHubbard import FermiHubbardModel
from fauvqe.models.fermionicModel import FermionicModel
from fauvqe.models.abstractmodel import AbstractModel
import fauvqe.utils as utils
import fauvqe.utils_cirq as cqutils
import numpy as np
import itertools

def generic_ansatz(model: AbstractModel,layers,ansatz: callable):
    circuit = cirq.Circuit()
    symbols = []
    circuit,symbols=ansatz(model=model,symbols=symbols,layers=layers)
    model.circuit.append(circuit)
    model.circuit_param.extend(list(utils.flatten(symbols)))
    if model.circuit_param_values is None:
        model.circuit_param_values = np.array([])
    model.circuit_param_values=np.concatenate((model.circuit_param_values,np.zeros(np.size(symbols))))

def potato_ansatz(model: FermionicModel,layers=1):
    def ansatz(model: FermionicModel,symbols,layers):
        qubits = model.flattened_qubits
        circuit=cirq.Circuit()
        for layer in range(layers):
            layer_symbols=[]
            layer_symbols.append(sympy.Symbol("theta_{}".format(layer)))
            layer_symbols.append(sympy.Symbol("phi_{}".format(layer)))
            for a in range(2):
                sub_op_tree=[]
                for iii in range(a,len(qubits)-1,2):
                    qi = qubits[iii]
                    qj = qubits[iii+1]
                    sub_op_tree.append(cirq.FSimGate(theta=layer_symbols[-2],phi=layer_symbols[-1]).on(qi,qj))
                circuit.append(sub_op_tree,strategy=InsertStrategy.EARLIEST)
            symbols.append(layer_symbols)
        return circuit,symbols
    generic_ansatz(model=model,layers=layers,ansatz=ansatz)

def triangle_potato_ansatz(model: FermionicModel,layers=1):
    def ansatz(model: FermionicModel,symbols,layers):
        qubits = model.flattened_qubits
        circuit=cirq.Circuit()
        for layer in range(layers):
            layer_symbols=[]
            layer_symbols.append(sympy.Symbol("theta_{}".format(layer)))
            layer_symbols.append(sympy.Symbol("phi_{}".format(layer)))
            sub_op_tree=[]
            Nq=len(qubits)-1
            for iii in range(0,Nq):
                for jjj in range(iii,Nq-iii,2):
                    qi = qubits[jjj]
                    qj = qubits[jjj+1]
                    sub_op_tree.append(cirq.FSimGate(theta=layer_symbols[-2],phi=layer_symbols[-1]).on(qi,qj))
            circuit.append(sub_op_tree,strategy=InsertStrategy.NEW_THEN_INLINE)
            symbols.append(layer_symbols)
        return circuit,symbols
    generic_ansatz(model=model,layers=layers,ansatz=ansatz)

def totally_connected_ansatz(model: FermionicModel,layers=1):
    def ansatz(model: FermionicModel,symbols,layers):
        qubits = model.flattened_qubits
        Nq = len(qubits)
        circuit=cirq.Circuit()
        perms = list(itertools.combinations(range(Nq),2))
        for layer in range(layers):
            layer_symbols=[]

            sub_op_tree=[]
            for ni,nj in perms:
                layer_symbols.append(sympy.Symbol("theta_{l}_{ni}_{nj}".format(l=layer,ni=ni,nj=nj)))
                layer_symbols.append(sympy.Symbol("phi_{l}_{ni}_{nj}".format(l=layer,ni=ni,nj=nj)))
                qi = qubits[ni]
                qj = qubits[nj]
                sub_op_tree.append(cirq.FSimGate(theta=layer_symbols[-2],phi=layer_symbols[-1]).on(qi,qj))
            circuit.append(sub_op_tree,strategy=InsertStrategy.EARLIEST)
            symbols.append(layer_symbols)
        return circuit,symbols
    generic_ansatz(model=model,layers=layers,ansatz=ansatz)