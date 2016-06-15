import os
from setuptools import setup


def read(fname):
    """Read the name relative to current directory"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="safemdp",
    version="1.0",
    author="Matteo Turchetta, Felix Berkenkamp",
    author_email="matteotu@ethz.ch, befelix@ethz.ch",
    description=("Safe exploration in MDPs"),
    license="MIT",
    url="http://packages.python.org/an_example_pypi_project",
    packages=['safemdp'],
    long_description=read('README.md'),
    install_requires=[
        'GPy >= 0.8.0',
        'numpy >= 1.7.2',
        'scipy >= 0.16',
        'matplotlib >= 1.5.0',
        'networkx >= 1.1',
    ],
)
