"""Shim for editable installs on old pip/setuptools (pre-PEP 660).

All project metadata lives in pyproject.toml. This file exists so that
`pip install -e .` works on the Jetson's older Python 3.6 / pip 20 /
setuptools 59 combo, which predates the PEP 660 'build_editable' hook.
Modern environments ignore this file and use pyproject.toml directly.
"""
from setuptools import setup

setup()
