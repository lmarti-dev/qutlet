import pytest 
import openfermion as of
import fauvqe.utils_cirq as cqutils


@pytest.mark.parametrize("coeff, indices, correct",
    [
        (.5,[1,2],[(1,1),("1^ 2", "2^ 1")]),
        (.666,[1,2,3,4],[(1,1),(("1^ 2^ 3 4", "3^ 4^ 1 2"))]),
        (.65,[1,2,3,4,5,6],[(1,-1),("1^ 2^ 3^ 4 5 6", "4^ 5^ 6^ 1 2 3")]),
        (.65,[1,2,3,4,5,6,7,8],[(1,1),("1^ 2^ 3^ 4^ 5 6 7 8", "5^ 6^ 7^ 8^ 1 2 3 4")]),
        (.65,[2,10,33,21,12,3],[(1,-1),("2^ 10^ 33^ 21 12 3", "21^ 12^ 3^ 2 10 33")]),
    ]
)
def test_even_excitation(coeff,indices,correct):
    correct_fop = sum([coeff*parity*of.FermionOperator(s) for parity,s in zip(correct[0],correct[1])])
    assert(cqutils.even_excitation(coeff=coeff,indices=indices)==correct_fop)