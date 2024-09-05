from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from math import factorial
import numpy as np
from openfermion import get_fermion_operator, get_sparse_operator
from qutlet.utilities.fermion import (
    jw_eigenspectrum_at_particle_number,
    jw_get_true_ground_state_at_particle_number,
)
from qutlet.utilities.complexity import (
    global_entanglement_wallach,
    stabilizer_renyi_entropy,
)

from qutlet.utilities.circuit import pauli_neighbour_order

import io
from cirq.contrib.svg import circuit_to_svg
from cirq import Circuit

if TYPE_CHECKING:
    from qutlet.models import FermionicModel


def plot_ham_spectrum_non_quadratic_sweep(
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


def plot_ham_complexity_non_quadratic_sweep(
    model: "FermionicModel",
    n_steps: int = 100,
    which_sweep: str = "lin",
    which: str = "add",
):

    n_sorts = 2
    entropies = np.zeros((n_steps, n_sorts))

    if which_sweep == "lin":
        strengths = np.linspace(0, 1, n_steps)
    elif which_sweep == "log":
        strengths = np.logspace(-10, 0, n_steps)

    for ind, strength in enumerate(strengths):
        if which == "add":
            fop = (
                get_fermion_operator(model.quadratic_terms)
                + strength * model.non_quadratic_terms
            )
        elif which == "linterp":
            fop = (1 - strength) * get_fermion_operator(
                model.quadratic_terms
            ) + strength * model.non_quadratic_terms
        ground_energy, ground_state = jw_get_true_ground_state_at_particle_number(
            get_sparse_operator(fop), particle_number=model.n_electrons
        )

        entropies[ind, 0] = global_entanglement_wallach(
            state=ground_state, n_qubits=model.n_qubits
        )
        entropies[ind, 1] = stabilizer_renyi_entropy(
            state=ground_state, n_qubits=model.n_qubits
        )
        print(
            f"{strength:.3f} {ind} ge:{entropies[ind,0]:.3f} sre:{entropies[ind,1]:.3f} {ground_energy:.3f}"
        )

    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.plot(
        strengths,
        entropies[:, 0],
        label="Global entanglement",
    )
    ax.plot(
        strengths,
        entropies[:, 1],
        label="Stabilizer Rényi entropy",
    )
    if which == "linterp":
        prefac = "(1-t)"
    else:
        prefac = ""
    ax.set_xlabel(
        rf"$t: \ {prefac}\sum_{{i,j}} c_{{ij}} a^{{\dagger}}_i a_j  + t \sum_{{i,j,k,\ell}} h_{{ijk\ell}} a^{{\dagger}}_i a^{{\dagger}}_j a_k a_{{\ell}}$"
    )
    ax.set_ylabel("Global entanglement")
    if which_sweep == "log":
        ax.set_xscale("log")
    ax.legend()

    return fig


def save_circuit_svg(circuit: Circuit, filepath: str):
    svg = circuit_to_svg(circuit)
    fstream = io.open(filepath, "w+", encoding="utf8")
    fstream.write(svg)
    fstream.close()


def get_integer_bins(data: np.ndarray) -> np.ndarray:

    d = np.min(np.diff(np.unique(data)))
    left_of_first_bin = np.min(data) - float(d) / 2
    right_of_last_bin = np.max(data) + float(d) / 2

    bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)
    return bins


def plot_model_weighted_locality_histogram(model: "FermionicModel"):
    p_locs = []
    k_locs = []
    w_p_locs = []
    w_k_locs = []
    for pstr in model.hamiltonian:
        p_loc = pauli_neighbour_order(pstr)
        if p_loc != 0:
            p_locs.append(p_loc)
            w_p_locs.append(np.abs(pstr.coefficient))
        if len(pstr.qubits):
            k_locs.append(len(pstr.qubits))
            w_k_locs.append(np.abs(pstr.coefficient))

    no_counts, no_bins = np.histogram(
        p_locs, bins=get_integer_bins(p_locs), density=True, weights=w_p_locs
    )

    k_counts, k_bins = np.histogram(
        k_locs, bins=get_integer_bins(k_locs), density=True, weights=w_k_locs
    )

    fig, axes = plt.subplots(ncols=2)

    axes[0].stairs(no_counts, no_bins, fill=True)
    axes[0].set_xlabel("Neighbour order")
    axes[0].set_ylabel("Weighted normalized amount")

    axes[1].stairs(k_counts, k_bins, fill=True)
    axes[1].set_xlabel("K-locality")
    return fig
