import pytest
import fauvqe.models.fermiHubbard as fh
import fauvqe.models.circuits.hva as hva


def test_init():
    fermi_hubbard = fh.FermiHubbardModel(2,2,1,1,qubit_map="sectors")
    fermi_hubbard.set_initial_state_circuit(name="slater")
    hva.set_circuit(fermi_hubbard)




def main():
    test_init()

if __name__ == "__main__":
    main()