"""
    Implementation of the expectation value of the model hamiltonian as objective function for an AbstractModel object.
"""
from cirq import Circuit as cirq_Circuit
from cirq import Simulator as cirq_Simulator
from numbers import Integral
import numpy as np
from typing import Dict, List, Literal, Optional, Tuple

from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue
from fauvqe.models.abstractmodel import AbstractModel
from fauvqe.objectives.objective import Objective

class ExpectationValue(AbstractExpectationValue):
    """Energy expectation value objective

    This class implements as objective the expectation value of the energies
    of the linked model.

    Parameters
    ----------
    model: AbstractModel    The linked model
    
    Methods
    ----------
    __repr__() : str
        Returns
        ---------
        str:
            <ExpectationValue Energy fields=self.__energy_fields>
    """

    def __init__(   self, 
                    model: AbstractModel,
                    energy_filter: Optional[np.ndarray] = None):
        super().__init__(model)
        if energy_filter is None:
            self.__energy_fields: List[Literal["X", "Y", "Z"]] = model.energy_fields
            self.__energies: Tuple[np.ndarray, np.ndarray] = model.energy()
            assert len(self.__energy_fields) == len(self.__energies), "Length of Pauli types and energy masks do not match"
        else:
            self.__energy_fields = ["Z"]
            #The minus here is added to compensate a -1 later
            self.__energies = [-energy_filter]

        self.__n_qubits: Integral = np.log2(np.size(self.__energies[0]))
    
    def evaluate(self, wavefunction: np.ndarray, options: dict = {}) -> np.float64:
        if options.get("rotation_circuits") is None:
            i=0
            expectation_value=0
            for field in self.__energy_fields:
                if(field == "X"):
                    rot_wf = self._rotate_x(wavefunction)
                elif(field == "Y"):
                    rot_wf = self._rotate_y(wavefunction)
                elif(field == "Z"):
                    rot_wf = wavefunction
                else:
                    raise NotImplementedError()
                
                expectation_value += np.sum( np.abs(rot_wf) ** 2 * (-self.__energies[i]) ) / self.__n_qubits
                i+=1
            return expectation_value

        else:
            # This is the quick and easy but factor 2 slower cirq implementation
            # Of rotated basis expecation values
            # So far not quite clear if this does what we want
            # rot_circuit = identity already gives correct expectation value
            assert isinstance(self._model.simulator, cirq_Simulator), "Currently cirq simulator for rotated expectation values required"

            rotation_circuits = options.get("rotation_circuits")
            if options.get("qubit_order") is None:
                _qubit_order={self._model.qubits[k][l]: int(k*self._model.n[1] + l) for l in range(self._model.n[1]) for k in range(self._model.n[0])}
            else:
                _qubit_order=options.get("qubit_order") 
            print("_qubit_order: {}".format(_qubit_order))
            if isinstance(rotation_circuits, cirq_Circuit):
                return self._model.simulator.simulate_expectation_values(
                        rotation_circuits,
                        initial_state=wavefunction,
                        observables=self._observable) / self.__n_qubits
                        #qubit_order=_qubit_order) / self.__n_qubits
            elif all(isinstance(obj, cirq_Circuit) for obj in rotation_circuits):
                #Doing this potential does not make sense:
                expectation_value = 0
                for i in range(len(rotation_circuits)):                    
                    expectation_value += self._model.simulator.simulate_expectation_values(
                                            rotation_circuits[i],
                                            initial_state=wavefunction,
                                            observables=self._observable,
                                            qubit_order=_qubit_order) / self.__n_qubits/ len(rotation_circuits)
                    print(rotation_circuits[i])
                    print(expectation_value)
                return expectation_value
            else:
                assert False, "Given rotation circuit is not of type cirq.Circuit"


            #To Do: Implement this more efficently
            #Cirq implmentation about factor 2 slower compared to own implementation.
            #
            #Expect here List[cirq.Circuits]: rotation_circuits
            #Expect self.__energies[0] to be energies in Z basis
            #self.__energies[0] is interaction
            #self.__energies[1] is external field
            # 
            # Ideas for own implmentation
            #rotation_circuits = options.get("rotation_circuits")
            #if options.get("qubit_order") is None:
            #    _qubit_order={self._model.qubits[k][l]: int(k*self._model.n[1] + l) for l in range(self._model.n[1]) for k in range(self._model.n[0])}
            #else:
            #    _quit_order=options.get("qubit_order") 

            #if isinstance(rotation_circuits, cirq_Circuit):
            #    return np.sum( np.abs( self._model.simulator.simulate(
            #                            rotation_circuits,
            #                            qubit_order=_qubit_order,
            #                            initial_state=wavefunction,
            #                            ).state_vector())** 2 * (-self.__energies[0]) ) / self.__n_qubits
            #elif isinstance(rotation_circuits, List[cirq_Circuit]):
            #    expectation_value = 0
            #    for i in range(len(rotation_circuits)):                    
            #        expectation_value += np.sum( np.abs( self._model.simulator.simulate(
            #                            rotation_circuits[i],
            #                            qubit_order=_qubit_order,
            #                            initial_state=wavefunction,
            #                            ).state_vector())** 2 * (-self.__energies[0]) ) / self.__n_qubits
            #    return expectation_value
            #else:
            #    assert False, "Given rotation circuit is not of type cirq.Circuit"

    #Old Version
    """
    def evaluate(self, wavefunction: np.ndarray, options: dict = {}) -> np.float64:
        if self.__field == "X":
            wf_x = self._rotate_x(wavefunction)

            return (
                np.sum(
                    np.abs(wavefunction) ** 2 * (-self.__energies[0])
                    + np.abs(wf_x) ** 2 * (-self.__energies[1])
                )
                / self.__n_qubits
            )
        
        # field must be "Z"
        return (
            np.sum(np.abs(wavefunction) ** 2 * (-self.__energies[0] - self.__energies[1]))
            / self.__n_qubits
        )
    """

    def to_json_dict(self) -> Dict:
        return {
            "constructor_params": {
                "model": self._model,
            },
        }

    @classmethod
    def from_json_dict(cls, dct: Dict):
        return cls(**dct["constructor_params"])

    def __repr__(self) -> str:
        return "<ExpectationValue Energy field={}>".format(self.__energy_fields)

    def __eq__(self, other): 
        '''Temporary solution'''
        if not isinstance(other, self.__class__):
            # don't attempt to compare against unrelated types
            return False

        #Most general: avoid to define Attributes
        temp_bools = []
        for key in self.__dict__.keys():
            print(key)
            if(key == '_ExpectationValue__energies'):
                temp_bools.append((getattr(self, key)[0] == getattr(other, key)[0]).all())
                temp_bools.append((getattr(self, key)[1] == getattr(other, key)[1]).all())
                continue
            temp_bools.append(getattr(self, key) == getattr(other, key))
        return all(temp_bools)