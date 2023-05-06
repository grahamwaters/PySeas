# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='sample',
    version='0.1.0',
    description='PySeas Setup',
    long_description=readme,
    author='Graham Waters',
    author_email='gtxdatascientist@gmail.com',
    url='https://github.com/grahamwaters/PySeas',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
