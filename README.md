# Kelvin
Quantum chemistry at non-zero temperature

[![Tests](https://github.com/awhite862/kelvin/workflows/CI/badge.svg)](https://github.com/awhite862/kelvin/actions/workflows/ci.yml)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/awhite862/kelvin/master/LICENSE)

## Features
The kelvin package contains some implementations of finite-temperature quantum chemical methods. In particular reference implementations of finite-temperature coupled cluster (FT-CC) methods are provided.

## Examples
see the [examples](../master/examples)

## Tests
The tests can be run individually from the `kelvin/tests` subdirectory. All tests can be run at once by running any of the following commands:
  - `python kelvin/tests/test_suites.py all`
  - `python -m unittest test.py`
  - `python -m pytest test.py`
  - `pytest test.py`

Note that running all tests will take several hours. A subset of the tests can be run with `python kelvin/tests/test_suites.py`
and an even smaller subset of of the tests can be run with `python test_ci.py`.
