"""
Objectives used in the `fauvqe` optimisation.
"""

from fauvqe.objectives.abstractexpectationvalue import AbstractExpectationValue
from fauvqe.objectives.correlation import Correlation
from fauvqe.objectives.cvar import CVaR
from fauvqe.objectives.entanglement import Entanglement
from fauvqe.objectives.expectationvalue import ExpectationValue
from fauvqe.objectives.fidelity import Fidelity
from fauvqe.objectives.magnetisation import Magnetisation
from fauvqe.objectives.matrixcost import MatrixCost
from fauvqe.objectives.objective import Objective
from fauvqe.objectives.tracedistance import TraceDistance
from fauvqe.objectives.utcost import UtCost
from fauvqe.objectives.variance import Variance