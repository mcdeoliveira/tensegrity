Quick start
===========

This package contains classes and functions that facilitate the design and analysis of tensegrity structures.
It started as a matlab toolbox many years ago written by Mauricio de Oliveira and Joseph Cessna, which has now been
ported to Python.

It relies heavily on `numpy <https://numpy.org>`_, `scipy <https://scipy.org>`_, `pandas <https://pandas.org>`_, and
`cvxpy <https://cvxpy.org>`_ for computations.

Visualization is supported via `matplotlib <https://matplotlib.org>`_ and
`vispy <https://vispy.org>`_.

Hello World
-----------

Try the following code::

    from tensegrity.prism import Prism
    s = Prism()

The object ``s`` is a :class:`tensegrity.structure.Structure` representing a **Snelson Tensegrity Prism** with three
bars and nine strings such as the one in the following figure:

.. image:: /images/snelson.png
   :scale: 50%

The structure can be visualized using `matplotlib <https://matplotlib.org>`_::

    from matplotlib import pyplot as plt
    # add your favorite matplotlib magic below
    %matplotlib widget

and :class:`tensegrity.plotter.MatplotlibPlotter`::

    from tensegrity.plotter.matplotlib import MatplotlibPlotter

to produce a 3D plot like the one in the figure above::

    plotter = MatplotlibPlotter()
    plotter.plot(s)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

The underlying ``figure`` and ``axis`` can be retrieved and used to manipulate the plot::

    fig, ax = plotter.get_handles()
    ax.view_init(elev=20, azim=45, roll=0)
    plt.show()

Prisms, lots of prisms
----------------------

It is possible to construct Snelson prisms with different number of bars::

    prisms = [Prism(n) for n in (3, 4, 6, 12)]

All prisms are constructed centered at the origin. They can be translated::

    import numpy as np
    prisms = [prism.translate(np.array([3*i,0,0])) for i, prism in enumerate(prisms)]

before plotting::

    plotter = MatplotlibPlotter()
    plotter.plot(prisms)
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    fig, ax = plotter.get_handles()
    ax.view_init(elev=30, azim=60)
    ax.set_box_aspect(None, zoom=1.8)

to produce a plot such as:

.. image:: /images/prisms.png
  :scale: 50%

Plotting with VisPy
-------------------

It is also possible to plot structures using `VisPy <https://vispy.org/>`_.
Certain users might need to tweak their installations.
See `installation instructions <https://vispy.org/installation.html>`_ for details.

The above prisms can be plotted with VisPy::

    from IPython.display import display
    import jupyter_rfb
    from tensegrity.plotter.vispy import VisPyPlotter
    plotter = VisPyPlotter(scene={'size': (800,200), 'app': 'jupyter_rfb'},
                           camera={'scale_factor': 6, 'center': (4.5,2,0)})
    plotter.plot(prisms)
    plotter.get_canvas()
