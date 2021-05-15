# fauvqe

Cirq-based custom VQE python package

## Installation

Both the production and the development require python3.6+, pip and the requirements as defined in `requirements.txt`.
To install those, run

```bash
pip install -r requirements.txt
```

### Production

TODO: Expose setup.py to install this as a pip package.

### Development

In addition to those defined in `requirements.txt`, the development setup requires more dependencies.
To install those, run

```bash
pip install --upgrade \
  black \
  sphinx \
  pre-commit
```

## Development guide

### Linting

This module uses [black code style](https://pypi.org/project/black/).
It is automatically run before a commit, but can be manually invoked by running `black` in the project root.
Its configuration can be found in the `pyproject.toml` file under `[tools.black]`.

### git hooks

There are a few git hooks which are run before committing.
Those hooks are handled by [pre-commit](https://pre-commit.com/).
Currently, they include:

- black (code style linting)

To install more pre-commit hooks, modify `.pre-commit-config.yaml` and run `pre-commit install`.

### Continuous integration (CI)

CI is done with GitHub actions.

## Documentation

[sphinx](https://www.sphinx-doc.org/en/master/) is used to generate a html/pdf documentation from docstrings.
Additional documentation can easily be added by editing the files in [docs](./tree/main/docs/source).

To generate the documentation including the `autodoc` files for automatic docstring documentation run

```bash
cd ./docs/
make generate
make html # alternatively: latex
$BROWSER ./build/html/index.html
```

Refer to the documentation of sphinx for more information, for example on how to add custom documentation to the `.rst` files.

## Design ideas

### General ideas

- Avoid unnecessary dependencies, if locally needed load in class via importlib
- Reasoning: Update of dependencies can crash with existing code hence keep dependencies to minimum
- Abolish legancy code, use pytest for testing and pytest-cov to test test coverage
- Reasoning:
  - After a given time, usually only a few month, we don't know anymore that exactly parts of the code are doing; testing ensures that it still does what we expected, when we understood/wrote the code
  - If you not have written part of the code you are using, but it was written by another group member you would like to know whether it does what was expected by the person writting it. This way you don't need to understand every line of the entire code.
- Avoid > 1000 line files; rather put some functions in submodules e.g. one can put different circuit parts/ideas in submodules and call/load/include them in a 1 line load, which is much easier to understand.
- Use inheritance and Abstract/superclasses to ensure compatibility e.g. there a some properties every optimiser needs to have. With inheritance the same objects will always be name the same and have the same abstract structure

### Further ideas

- use abstract class (via python abstract base class: from abc import ABC, abstractmethod) Abstract classes can be a better way of interfacing, maybe simply by leeting intialiser() and Optimiser() be subclasses of abc.ABCMETA?
- use cython and explicit type declaration for higher performance
- Add multiprocessing/MPI4py expecially when differnt f(x) have to be evaluated/the objective function has to be evaluated at different points.

## Single parts

- Initialiser() defines/ensures abstract compatibility (e.g. but not limited to):
  - qubits (how are these stored, relevant e.g. for optimiser structure)
  - circuit (every subclass should have a cirq.Circuit() object)
  - circuit_param (list of sympy symbols to parametries cirq.circuit)
  - cicuit_param_values ()
  - simulator (to sample/simulate wf of circuit)
- Individual projects in sub-modules:
  - e.g. isings
    - Ising() class has
      - model related functions such as energy()
      - function to persue/sets different circuits
      - should use the general Optimiser()-class and its sub-classes/modules such as GradientDescent() to optimise parametrised circuit

## Next steps

1. 100 % test coverage
2. Include periodic boundary conditions
3. Check again analytic result, maybe include analytic 1D TFIM wavefunction?
4. Add an alternative optimiser/adaptive step size/gradient shift rule
5. Add/Use MPI4py|Multiprocessing|Threatening|joblib within optimiser to achieve speed-up
