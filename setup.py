#!/usr/bin/env python3

from setuptools import setup, find_packages
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))



version = {}
with open(os.path.join(_here, 'qq', 'version.py')) as f:
    exec(f.read(), version)
    
    
setup(
    name='qq',
    version=version['__version__'],
    url='https://github.com/ffsedd/qq/',
    author='ffsedd',
    author_email='ffsedd@gmail.com',
    description='python tools library',
    packages=['qq'],
    #scripts=['qq'],
    install_requires=['send2trash', 'pillow'],
    include_package_data=True,
)
