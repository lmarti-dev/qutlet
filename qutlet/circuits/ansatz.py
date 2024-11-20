from cirq import Circuit, ParamResolver, SimulatorBase, Simulator
from qutlet.utilities import populate_empty_qubits, depth
import sympy
import uuid


def param_name_from_circuit(circ: Circuit, proptype="ops") -> str:
    if circ == Circuit():
        # quick fix if the circuit is empty
        num = uuid.uuid1()
    if proptype == "ops":
        num = sum(1 for _ in circ.all_operations())
    elif proptype == "depth":
        num = depth(circuit=circ)
    return "p_" + str(num)


class Ansatz:
    def __init__(
        self,
        circuit: Circuit = None,
        symbols: list = None,
        params: list = None,
        name: str = None,
    ) -> None:
        if circuit is None:
            circuit = Circuit()
        self.circuit = circuit
        self.name = name

        if symbols is None:
            if params is not None:
                self.params = params
                self.symbols = [
                    sympy.Symbol(name=param_name_from_circuit(self.circuit))
                    for x in range(self.n_params)
                ]
            else:
                self.params = []
                self.symbols = []
        else:
            self.symbols = symbols
            if params is None:
                self.params = [0 for x in range(self.n_symbols)]
            else:
                self.params = params

    def __iadd__(self, sympar: tuple[sympy.Symbol, float]):
        sym, par = sympar
        if sym in self.symbols:
            raise ValueError("Cannot overwrite symbols with iadd")
        self.symbols.append(sym)
        self.params.append(par)

    @property
    def n_qubits(self):
        return len(self.circuit.all_qubits())

    def param_resolver(self, override_params: list = None):
        if override_params is not None:
            return ParamResolver(
                {str(k): v for k, v in zip(self.symbols, override_params)}
            )
        return ParamResolver({str(k): v for k, v in zip(self.symbols, self.params)})

    @property
    def n_symbols(self):
        return len(self.symbols)

    @property
    def n_params(self):
        return len(self.params)

    @property
    def __to_json__(self):
        return {
            "circuit": self.circuit,
            "symbols": self.symbols,
            "params": self.params,
            "name": self.name,
        }

    def simulate(
        self,
        *args,
        simulator: SimulatorBase = None,
        override_params: list = None,
        **kwargs
    ):
        if simulator is None:
            simulator = Simulator()
        if "initial_state" in kwargs.keys() and "state_qubits" in kwargs.keys():
            circuit = populate_empty_qubits(
                circuit=self.circuit, qubits=kwargs["state_qubits"]
            )
            del kwargs["state_qubits"]
        else:
            circuit = self.circuit
        if isinstance(simulator, Simulator):
            return simulator.simulate(
                *args,
                **kwargs,
                param_resolver=self.param_resolver(override_params=override_params),
                program=circuit,
            ).final_state_vector
