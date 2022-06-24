#external imports:
import numpy as np
import pytest
from scipy import stats

# internal imports
from fauvqe import uniform

@pytest.mark.parametrize(
    "n, m",
    [
        (
            2, 100
        )
    ]
)
def test_uniform(n, m):
    # Test whether norms are 1
    # test for uniform distribution:
    # https://www.eg.bucknell.edu/~xmeng/Course/CS6337/Note/master/node43.html
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    # Still need to figure good test
    
    random_states = uniform(n,m)
    probabilities = abs(random_states)**2
    #print(stats.kstest(probabilities, "uniform", N=m, alternative='two-sided', mode='auto'))
    assert False

