#!/usr/bin/env python3

import setuptools
import glob


with open('mri_project/_version.py') as f:
    exec(f.read())

with open('README.rst') as f:
    readme = f.read()

setuptools.setup(
    name='mri_project',

    # Read in from above.
    version=__version__,

    author= 'Behnam Rasoolian',

    description='Additive',
    long_description=readme,

    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False,

    python_requires='>= 3.6',
    install_requires=[
    ],
    setup_requires=[
       'pytest-runner == 2.10'
    ],
    tests_require=[
       'pytest == 3.2.1'
    ],

    scripts=glob.glob('bin/*')
)
