#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = []

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest>=3',
]

setup(
    author="Adelle Goodwin",
    author_email='adelle.goodwin@curtin.edu.au',
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Package to fit TDE radio spectra to determine the spectral index, p, and the peak flux desnity and frequency of the spectrum",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='tde_spectra_fit',
    name='tde_spectra_fit',
    packages=find_packages(include=['tde_spectra_fit', 'tde_spectra_fit.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/adellej/tde_spectra_fit',
    version='0.1.0',
    zip_safe=False,
)
