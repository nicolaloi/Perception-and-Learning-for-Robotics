#!/usr/bin/env python
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = '0.0.3'

setup(
    name='real_lsd',
    version=version,
    install_requires=requirements,
    )   
