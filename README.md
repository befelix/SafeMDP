# SafeMDP

[![Build Status](https://travis-ci.org/befelix/SafeMDP.svg?branch=master)](https://travis-ci.org/befelix/SafeMDP)
[![Documentation Status](https://readthedocs.org/projects/safemdp/badge/?version=latest)](http://safemdp.readthedocs.io/en/latest/?badge=latest)

Code for safe exploration in Markov Decision Processes (MDPs). This code accompanies the paper

M. Turchetta, F. Berkenkamp, A. Krause, "Safe Exploration in Finite Markov Decision Processes with Gaussian Processes", Proc. of the Conference on Neural Information Processing Systems (NIPS), 2016, <a href="http://arxiv.org/abs/1606.04753" target="_blank">[PDF]</a>

# Installation

The easiest way to install use the library is to install the <a href="https://www.continuum.io/downloads" target="_blank">Anaconda<a/> Python distribution. Then, run the following commands in the root directory of this repository:
```
pip install GPy
python setup.py install
```

# Usage

The documentation of the library is available on <a href="http://safemdp.readthedocs.io/en/latest/" target="_blank">Read the Docs</a>

The file `examples/sample.py` implements a simple examples that samples a random world from a Gaussian process and shows exploration results.

The code for the experiments in the paper can be found in the `examples/mars/` directory.
