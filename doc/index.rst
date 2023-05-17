.. tnsgrt documentation master file, created by
   sphinx-quickstart on Sat May  6 16:31:07 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tnsgrt's documentation!
======================================

This package contains classes and functions that facilitate the design and analysis of tensegrity structures.
It started as a matlab toolbox many years ago written by Mauricio de Oliveira and Joseph Cessna, which is now been
ported to Python.

It relies heavily on `numpy <https://numpy.org>`_, `scipy <https://scipy.org>`_, `pandas <https://pandas.org>`_, and
`cvxpy <https://cvxpy.org>`_ for computations.

Visualization is supported via `matplotlib <https://matplotlib.org>`_ and
`vispy <https://vispy.org>`_.

.. toctree::
   :maxdepth: 3
   :caption: User guide

   usage/quickstart
   usage/structure
   usage/properties
   usage/transformations
   usage/examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   usage/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
