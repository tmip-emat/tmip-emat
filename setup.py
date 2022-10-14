# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup, find_packages

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

with open('requirements.txt') as f:
    requirements_lines = f.readlines()
install_requires = [r.strip() for r in requirements_lines]

with open('README.md') as f:
    README = f.read()

def recursive_glob(front, back):
    return [os.path.join(front, *(['*']*i), back) for i in range(10)]

setup(
    name='emat',
    version=VERSION,
    packages=find_packages(),
    package_data={
        # If sub-package contains these types of files, include them:
        'emat': [
            'model/tests/*.yaml',
            'scope/tests/*.yaml',
            'database/sqlite/*.sql',
            *recursive_glob('examples', '*.yaml'),
            *recursive_glob('examples', '*.csv'),
            *recursive_glob('examples', '*.xlsx'),
            *recursive_glob('examples', '*.db.gz'),
        ],
    },
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'emat-road-test-demo = emat.model.core_python.core_python_examples:_Road_Capacity_Investment_CmdLine',
        ],
    },
    long_description_content_type="text/markdown",
    long_description=README,
)

