# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:50:28 2018

@author: mmilkovits
"""
import os
import re

from ez_setup import use_setuptools
use_setuptools()


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, path), encoding='utf-8') as f:
        version_file = f.read()
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

VERSION = version('emat/__init__.py')

from setuptools import setup, find_packages

setup(
    name='emat',
    version=VERSION,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'appdirs',
        'pydot',
        'plotly',
        'scipy',
        'seaborn',
        'ema_workbench',
    ],
)

