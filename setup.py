#!/usr/bin/env python3

from setuptools import setup, find_packages




version = {}
with open(os.path.join(_here, 'somepackage', 'version.py')) as f:
    exec(f.read(), version)
    
    
setup(
    name='qq',
    version=version['__version__'],
    url='https://github.com/ffsedd/qq/',
    author='ffsedd',
    author_email='ffsedd@gmail.com',
    description='python tools library',
    packages=['qq'],
    scripts=['qq'],
    install_requires=['send2trash', 'pillow'],
)
