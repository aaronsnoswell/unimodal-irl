#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="unimodal_irl",
    version="0.1.0",
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "mdp_extras @ git+https://github.com/aaronsnoswell/mdp-extras.git",
    ],
    packages=find_packages(),
)
