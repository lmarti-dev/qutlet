import pytest
import models.fermiHubbardModel as fh
import fauvqe.models.circuits.hva as hva
import openfermion as of


def test_init():
    fermi_hubbard = fh.FermiHubbardModel(
        x_dimension=2,
        y_dimension=2,
        tunneling=2,
        coulomb=2,
        fock_maps=(0, 2, 1, 3, 4, 6, 5, 7),
        Z_snake=(0, 1, 2, 3, 7, 6, 5, 4),
    )
    hva.set_circuit(fermi_hubbard)
