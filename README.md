# SafeMDP

Code for safe exploration in Markov Decision Processes (MDPs). This code accompanies the paper

M. Turchetta, F. Berkenkamp, A. Krause, "Safe Exploration in Finite Markov Decision Processes with Gaussian Processes", ArXiv, 2016.

# Installation

The easiest way to install use the library is to install the <a href="https://www.continuum.io/downloads" target="_blank">Anaconda<a/> Python distribution. Then, run the following commands in the root directory of this repository:
```
pip install GPy
python setup.py install
```

# Usage

The file `examples/sample.py` implements a simple examples that samples a random world from a Gaussian process and shows exploration results.

The code for the experiments in the paper can be found in the `examples/mars/` directory.
