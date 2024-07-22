from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from math import factorial
import numpy as np
from openfermion import get_fermion_operator, get_sparse_operator
from qutlet.utilities import jw_eigenspectrum_at_particle_number

if TYPE_CHECKING:
    from qutlet.models import FermionicModel


def plot_spectrum_interaction_sweep(
    model: "FermionicModel", n_steps: int = 100, which: str = "nonquad"
):

    # t \sum a*i aj + strength * J \sum a*i a*j ak al

    strengths = np.linspace(0, 1, n_steps)

    # in fermionic model, unlikely to have odd n_qubits
    n_levels = (
        factorial(model.n_qubits // 2)
        / (
            factorial(model.n_electrons[0])
            * (factorial(model.n_qubits // 2 - model.n_electrons[0]))
        )
    ) * (
        factorial(model.n_qubits // 2)
        / (
            factorial(model.n_electrons[1])
            * (factorial(model.n_qubits // 2 - model.n_electrons[1]))
        )
    )

    energy_levels = np.zeros((len(strengths), int(n_levels)))

    for ind, strength in enumerate(strengths):
        print(f"strength: {strength:.3f}/{strengths[-1]:.3f}")

        if which == "nonquad":
            fop = (
                get_fermion_operator(model.quadratic_terms)
                + strength * model.non_quadratic_terms
            )
        elif which == "quad":
            fop = (
                strength * get_fermion_operator(model.quadratic_terms)
                + model.non_quadratic_terms
            )
        sparse_op = get_sparse_operator(fop)

        eigvals, _ = jw_eigenspectrum_at_particle_number(
            sparse_operator=sparse_op,
            particle_number=model.n_electrons,
            expanded=False,
        )
        energy_levels[ind, :] = eigvals

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("turbo", len(energy_levels.T))
    for ind, ei in enumerate(np.transpose(energy_levels)):
        ax.plot(strengths, ei, linewidth=0.5, color=cmap(ind))

    ax.set_title(
        rf"${model.n_electrons[0]}\uparrow, {model.n_electrons[-1]}\downarrow$"
    )

    ax.set_ylabel("Energy")
    if which == "quad":
        ax.set_xlabel(
            r"$s: \ s\sum_{i,j} t_{ij} a^{\dagger}_i a_j + \sum_{ijk\ell} a^{\dagger}_i a^{\dagger}_j a_k a_\ell$"
        )
    elif which == "nonquad":
        ax.set_xlabel(
            r"$s: \ \sum_{i,j} t_{ij} a^{\dagger}_i a_j + s\sum_{ijk\ell} a^{\dagger}_i a^{\dagger}_j a_k a_\ell$"
        )

    return fig
