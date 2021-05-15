# fauvqe

Cirq-based custom VQE python package

## Installation

Both the production and the development require python3.6+, pip and the requirements as defined in `requirements.txt`.
To install those, run

```shell
pip install -r requirements.txt
```

### Production

TODO: Expose setup.py to install this as a pip package.

### Development

In addition to those defined in `requirements.txt`, the development setup requires more dependencies.
To install those, run

```shell
pip install --upgrade \
  pylint \
  black \
  sphinx \
  pytest \
  pytest-cov \
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

### Tests

Test are implemented using `pytest` and can be found in the [tests](./tests) subdirectory.
The tests can be run by running `pytest` in the project root.

For coverage reports, use the `pytest-cov` plugin:

```shell
pytest --cov=fauvqe --cov-report=html
```

This will generate html output in a new htmlcov subdirectory.
For more options, refer to the [pytest-cov documentation](https://pytest-cov.readthedocs.io/en/latest/config.html).

### Docstring format

This module is documented with numpy-style docstrings.
The sphinx documentation provides an [elaborate example on how to use it](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html).
For a more technical approach, refer to [the numpydoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html).

## Documentation

### Sphinx

[sphinx](https://www.sphinx-doc.org/en/master/) is used to generate a html/pdf documentation from docstrings.
Additional documentation can easily be added by editing the files in [docs](./docs/source).

To generate the documentation including the `autodoc` files for automatic docstring documentation run

```shell
cd ./docs/
make generate
make html # alternatively: latex
$BROWSER ./build/html/index.html
```

Refer to the documentation of sphinx for more information, for example on how to add custom documentation to the `.rst` files.

### UML diagrams

`pylint` comes with a UML diagram generator called `pyreverse`.
With `pylint` installed, you should also be able to access `pyreverse` via CLI.

In the repository, run

```shell
pyreverse . -o png
```

for a simple UML diagram.
The generated files will be called `classes.png` and `packages.png`.
To also visualise private and protected members, add the flag `-f ALL`.

For more documentation, refer to `pyreverse --help`.
