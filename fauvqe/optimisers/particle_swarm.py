from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.local_best import LocalBestPSO
from fauvqe.optimisers.optimiser import Optimiser
from fauvqe.objectives.objective import Objective
from fauvqe.optimisers.optimisation_result import OptimisationResult
import fauvqe.utils as utils
import fauvqe.utils_cirq as cqutils

import numpy as np
from typing import Dict
import cirq


class ParticleSwarm(Optimiser):
    def __init__(
        self,
        minimize_options={
            "method": "GlobalBestPSO",
            "c1": 0.5,
            "c2": 0.3,
            "w": 0.9,
            "iters": 1000,
            "nparticles": 10,
        },
        initial_state=None,
    ):
        self.minimize_options = minimize_options
        self.initial_state = initial_state
        if self.minimize_options["method"] == "GlobalBestPSO":
            self.pso_method = GlobalBestPSO
        elif self.minimize_options["method"] == "LocalBestPSO":
            self.pso_method = LocalBestPSO
        else:
            raise ValueError(
                "Expected method to be either GlobalBestPSO or LocalBestPSO but got: {}".format(
                    self.minimize_options["method"]
                )
            )
        super().__init__()

    def simulate(self, x):
        nparticles = self.minimize_options["nparticles"]
        nparams = len(self._objective.model.circuit_param)
        if x.shape == (
            nparticles,
            nparams,
        ):
            for xk in x:
                wf = self._objective.simulate(
                    param_resolver=cqutils.get_param_resolver(
                        model=self._objective.model, param_values=xk
                    ),
                    initial_state=self.initial_state,
                )
        elif x.shape == (nparams,):
            wf = self._objective.simulate(
                param_resolver=cqutils.get_param_resolver(
                    model=self._objective.model, param_values=x
                ),
                initial_state=self.initial_state,
            )
        else:
            raise ValueError(
                "Expected the shape of x to be either ({nparticles},{nparams}) or ({nparams}) but got: {xshape}".format(
                    nparticles=nparticles, nparams=nparams, xshape=x.shape
                )
            )
        return wf

    def fun(self, x):
        wf = self.simulate(x)
        return self._objective.evaluate(wavefunction=wf)

    def optimise(self, objective: Objective):
        self._objective = objective
        self._fauvqe_res = OptimisationResult(self._objective)

        dimensions = len(self._objective.model.circuit_param)
        optimizer = self.pso_method(
            n_particles=self.minimize_options["nparticles"],
            dimensions=dimensions,
            options=self.minimize_options,
            init_pos=None,
        )

        cost, pos = optimizer.optimize(self.fun, iters=self.minimize_options["iters"])
        best_particle_ind = optimizer.swarm.pbest_cost.argmin().astype(int)
        # pos_history has a shape of (iters, n_particles, dimensions)
        # add all history of the particle that found the best objective

        # for now there is no way of adding the best position (xk) to the history without prohibitve
        # computations.
        # We need to wait for the pull request that add callbacks to the optimize routine.
        pos_history = np.array(optimizer.pos_history)
        best_path = np.reshape(
            (pos_history)[:, best_particle_ind, :], (self.minimize_options["iters"], dimensions)
        )
        cost_history = optimizer.cost_history
        for objective, xk in zip(cost_history, best_path):
            # For now add none and none to params and wavefunctions, since we don't know which one is best
            self._fauvqe_res.add_step(params=None, wavefunction=None, objective=objective)
        # add last step
        self._fauvqe_res.add_step(
            params=pos, wavefunction=self.simulate(np.array(pos)), objective=cost
        )

        return self._fauvqe_res

    def from_json_dict(self) -> Dict:
        pass

    def to_json_dict(self) -> Dict:
        pass
