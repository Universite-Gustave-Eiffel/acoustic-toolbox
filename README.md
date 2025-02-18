<p align="center">
    <img height="200px" src="https://raw.githubusercontent.com/Universite-Gustave-Eiffel/acoustic-toolbox/main/docs/images/logo_txt.png" />
    <!-- <h1 align="center">Acoustic Toolbox</h1> -->
</p>

<p align="center">
    <img src="https://badgen.net/static/science/enabled/green" />
    <a href="https://pypi.org/project/acoustic-toolbox/"><img src="https://badgen.net/pypi/v/acoustic-toolbox" /></a>
    <a href="LICENSE"><img src="https://badgen.net/github/license/Universite-Gustave-Eiffel/acoustic-toolbox" /></a>
    <img src="https://badgen.net/github/checks/Universite-Gustave-Eiffel/acoustic-toolbox" />
    <img src="https://readthedocs.org/projects/acoustic-toolbox/badge/?version=latest&style=flat">
</p>

The `acoustic-toolbox` module is a Python module with useful tools for acousticians.

## Installation

The latest release can be found on PyPI and installed with `pip install acoustic-toolbox`.

Otherwise, you can clone this repository and install with `pip install` or `pip install -e` when you want an editable install.

## Examples

Several examples can be found in the `docs/examples` folder.

## Tests

The test suite can be run with

`uv run pytest`

## Documentation

Documentation can be found [online](http://acoustic-toolbox.readthedocs.io/).

## License

`acoustic-toolbox` is distributed under the BSD 3-clause license. See `LICENSE` for more information.

## Contributing

Contributors are always welcome.

Setting up the development environment and dependency management is done with `uv`. `uv` can be installed [from source](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) or with `pip install uv`.

To install the development environment, run `uv sync --all-extras` in the root of the repository. This will setup a `.venv` and install all dependencies including dev and docs dependencies.

Documentation and examples are stored in the `docs/` folder and built with `uv run mkdocs serve` or `uv run mkdocs build` from .

## Origin project

This project is based on the amazing work initially done here: [python-acoustics](https://github.com/python-acoustics/python-acoustics)
