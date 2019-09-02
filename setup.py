#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='qq',
    version='1.0.0',
    url='https://github.com/ffsedd/qq.git',
    author='ffsedd',
    author_email='ffsedd@gmail.com',
    description='python tools library',
    packages=find_packages(),    
    install_requires=['send2trash', 'pillow'],
)
