Quick start
===========

Installation
------------

Install using pip

.. code-block:: python

    pip install tnsgrt

See `matplotlib <https://matplotlib.org>`_ and `vispy <https://vispy.org>`_ for tweaking visualization settings.

Source code
-----------

Code corresponding to the sections in this guide are distributed in the form of Jupiter notebooks and can be found at
the `examples directory <https://github.com/mcdeoliveira/tensegrity/tree/main/examples>`_
in the `source code repo <https://github.com/mcdeoliveira/tensegrity>`_.

Hello World
-----------

Try the following code

.. code-block:: python

    from tnsgrt.prism import Prism
    s = Prism()

The object ``s`` is a :class:`tnsgrt.structure.Structure` representing a **Snelson Tensegrity Prism** with three
bars and nine strings such as the one in the following figure:

.. image:: /images/snelson1.png
   :scale: 50%

The structure can be visualized using `matplotlib <https://matplotlib.org>`_

.. code-block:: python

    from matplotlib import pyplot as plt
    # add your favorite matplotlib magic below
    %matplotlib widget

and :class:`tnsgrt.plotter.MatplotlibPlotter`

.. code-block:: python

    from tnsgrt.plotter.matplotlib import MatplotlibPlotter

to produce a 3D plot like the one in the figure above

.. code-block:: python

    plotter = MatplotlibPlotter()
    plotter.plot(s)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

The underlying ``figure`` and ``axis`` can be retrieved and used to manipulate the plot

.. code-block:: python

    fig, ax = plotter.get_handles()
    ax.view_init(elev=20, azim=45, roll=0)
    plt.show()

which in this case rotates the plot to obtain the better viewpoint

.. image:: /images/snelson2.png
  :scale: 50%

Prisms, lots of prisms
----------------------

It is possible to construct Snelson prisms with different number of bars

.. code-block:: python

    prisms = [Prism(n) for n in (3, 4, 6, 12)]

All prisms are constructed centered at the origin. They can be translated

.. code-block:: python

    import numpy as np
    prisms = [prism.translate(np.array([3*i,0,0])) for i, prism in enumerate(prisms)]

before plotting

.. code-block:: python

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

The above prisms can be plotted with VisPy

.. code-block:: python

    from IPython.display import display
    import jupyter_rfb
    from tnsgrt.plotter.vispy import VisPyPlotter
    plotter = VisPyPlotter(scene={'size': (800,200), 'app': 'jupyter_rfb'},
                           camera={'scale_factor': 6, 'center': (4.5,2,0)})
    plotter.plot(prisms)
    plotter.get_canvas()

