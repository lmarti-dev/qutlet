from cirq import Circuit, ParamResolver, SimulatorBase, Simulator
from typing import Union


class Ansatz:
    def __init__(
        self,
        circuit: Circuit,
        simulator: SimulatorBase = None,
        param_resolver: Union[ParamResolver, list, tuple[list, list]] = None,
    ) -> None:
        self.circuit = circuit

        # we allow three ways of setting the param res:
        # 1. ParamResolver directly
        # 2. list of symbols (params initialized at None)
        # 3. tuple of (symbols, params)
        if isinstance(param_resolver, ParamResolver):
            self.param_resolver = param_resolver
        elif isinstance(param_resolver, tuple):
            self.param_resolver = ParamResolver(
                {k: v for k, v in zip(param_resolver[0], param_resolver[1])}
            )
        elif isinstance(param_resolver, list):
            self.param_resolver = ParamResolver({k: None for k in param_resolver})
        else:
            raise TypeError(
                f"Expected one of the allowed types, got: {type(param_resolver)}"
            )
        if simulator is None:
            self.simulator = Simulator()
        else:
            self.simulator = simulator

    @property
    def n_qubits(self):
        return len(self.circuit.all_qubits())

    @property
    def symbols(self):
        return list(self.param_resolver.param_dict.keys())

    @property
    def n_symbols(self):
        return len(self.symbols)

    @property
    def params(self):
        return list(self.param_resolver.param_dict.values())

    def simulate(self, *args, **kwargs):
        if isinstance(self.simulator, Simulator):
            return self.simulator.simulate(
                *args, **kwargs, param_resolver=self.param_resolver
            ).final_state_vector
