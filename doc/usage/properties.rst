The structure of Structures
===========================

Start by building the following simple planar structure

.. code-block:: python

    import numpy as np
    from tnsgrt.structure import Structure

    nodes = np.array([[0,0,0], [0,1,0], [2,1,0], [2,0,0]]).transpose()
    members = np.array([[0,1], [1,2], [2,3], [3,0], [0,2], [1,3]]).transpose()
    s = Structure(nodes, members, number_of_strings=4)

which is visualized as

.. image:: /images/planar1.png
  :scale: 50%

by the following code:

.. code-block:: python

    from matplotlib import pyplot as plt
    from tnsgrt.plotter.matplotlib import MatplotlibPlotter
    %matplotlib widget

    plotter = MatplotlibPlotter()
    plotter.plot(s)
    fig, ax = plotter.get_handles()
    ax.view_init(90,-90)
    ax.axis('equal')
    ax.axis('off')
    plt.show()

Accessing data
--------------

Quite often, it is useful to view and modify the data stored in a
Structure. The following examples show several ways of doing that.

Nodes and members
^^^^^^^^^^^^^^^^^

A string representation of a Structure displays its number of nodes,
number of bars, and number of strings. Typing

.. code-block:: python

    print(s)

produces::

     Structure with 4 nodes, 2 bars and 4 strings


A Structure has as attributes ``nodes``,

.. code-block:: python

    s.nodes

which is a 3 x n numpy array ::

     array([[0., 0., 2., 2.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.]])

storing the Structure's nodes in its columns, and ``members``

.. code-block:: python

    s.members

which is a 2 x m numpy array::

     array([[0, 1, 2, 3, 0, 1],
            [1, 2, 3, 0, 2, 3]], dtype=int64)

storing the Structure's members. Each column of ``member`` stores the indices of the nodes to which the
ends of a bar or of a string is connected to.

Properties
^^^^^^^^^^

Node and member properties are stored as pandas' Dataframes. In this
example

.. code-block:: python

    s.node_properties

returns the dataframe::

        radius  visible constraint          facecolor          edgecolor
     0   0.002     True       None  (0, 0.447, 0.741)  (0, 0.447, 0.741)
     1   0.002     True       None  (0, 0.447, 0.741)  (0, 0.447, 0.741)
     2   0.002     True       None  (0, 0.447, 0.741)  (0, 0.447, 0.741)
     3   0.002     True       None  (0, 0.447, 0.741)  (0, 0.447, 0.741)

and

.. code-block:: python

    s.member_properties

returns::

        lambda_  force  stiffness  volume  radius  inner_radius  mass  rest_length
     0      0.0    0.0        0.0     0.0   0.005           0.0   1.0          0.0  \
     1      0.0    0.0        0.0     0.0   0.005           0.0   1.0          0.0
     2      0.0    0.0        0.0     0.0   0.005           0.0   1.0          0.0
     3      0.0    0.0        0.0     0.0   0.005           0.0   1.0          0.0
     4      0.0    0.0        0.0     0.0   0.010           0.0   1.0          0.0
     5      0.0    0.0        0.0     0.0   0.010           0.0   1.0          0.0

                yld  density       modulus  visible             facecolor
     0  250000000.0   7850.0  2.000000e+11     True  (0.85, 0.325, 0.098)  \
     1  250000000.0   7850.0  2.000000e+11     True  (0.85, 0.325, 0.098)
     2  250000000.0   7850.0  2.000000e+11     True  (0.85, 0.325, 0.098)
     3  250000000.0   7850.0  2.000000e+11     True  (0.85, 0.325, 0.098)
     4  250000000.0   7850.0  2.000000e+11     True     (0, 0.447, 0.741)
     5  250000000.0   7850.0  2.000000e+11     True     (0, 0.447, 0.741)

                   edgecolor  linewidth linestyle
     0  (0.85, 0.325, 0.098)          2         -
     1  (0.85, 0.325, 0.098)          2         -
     2  (0.85, 0.325, 0.098)          2         -
     3  (0.85, 0.325, 0.098)          2         -
     4     (0, 0.447, 0.741)          2         -
     5     (0, 0.447, 0.741)          2         -

The DataFrames can be manipulated directly or one can use some
convenience methods that will be discussed later.

Properties are populated with default values taken from
``Structure.NodeProperty``, ``Structure.MemberProperty``, and the
values in the dictionary

.. code-block:: python

    Structure.member_defaults

returns the dictionary::

     {'bar': {'facecolor': (0, 0.447, 0.741), 'edgecolor': (0, 0.447, 0.741)},
      'string': {'facecolor': (0.85, 0.325, 0.098),
       'edgecolor': (0.85, 0.325, 0.098),
       'radius': 0.005}}

The keys in this dictionary are *tags*, which we discuss next.

Tags
^^^^

Nodes and members can be assigned and manipulated via *tags*. Nodes
do not have any default tag, as

.. code-block:: python

    s.node_tags

returns::

     {}

but members are automatically assigned either ``bar`` or ``string``
as a tag. Typing

.. code-block:: python

    s.member_tags

returns::

     {'bar': array([4, 5], dtype=int64), 'string': array([0, 1, 2, 3], dtype=int64)}

This association happens at the constructor time by passing the
parameter ``number_of_strings``, which tags the first
``number_of_strings`` members as ``strings`` and the remaining as
``bars``. Alternatively, one can pass tags at construction time in
the form of a dictionary with tags as keys and a numpy array of node
or string indices as values.

It is always recommended to manipulate tags using the convenience
methods of :class:`tnsgrt.structure.Structure`, which take care of keeping
the member and node indices unique and sorted.

Additional member tags can be assigned using
:meth:`tnsgrt.structure.Structure.add_member_tag`. For example

.. code-block:: python

    s.add_member_tag('vertical', [0, 2])

creates a new tag ``vertical`` and associated the two members with
indices ``0`` and ``2`` to it. Conversely,
``tnsgrt.structure.Structure.get_members_by_tag`` retrieves the
member indices associated with a given tag, as in

.. code-block:: python

    s.get_members_by_tag('vertical')

which returns::

     array([0, 2])

while ``tnsgrt.structure.Structure.get_member_tags`` retrieve all
tags associated with a given member index:

.. code-block:: python

    s.get_member_tags(2)

which returns::

     ['string', 'vertical']

Similar methods exist to manipulate node tags.

Retrieving and setting properties
---------------------------------

Even though it is possible to manipulate the property
``DataFrame``\ s directly, it is sometimes easier to use some
convenience methods.

For example

.. code-block:: python

    s.get_member_properties(s.get_members_by_tag('vertical'), 'radius', 'facecolor', 'edgecolor')

retrieves a view of the member's properties::

        radius             facecolor             edgecolor
     0   0.005  (0.85, 0.325, 0.098)  (0.85, 0.325, 0.098)
     2   0.005  (0.85, 0.325, 0.098)  (0.85, 0.325, 0.098)

for all members that have ``vertical`` as a tag.

Conversely

.. code-block:: python

    s.set_member_properties(s.get_members_by_tag('vertical'), 'radius', 0.04)

sets the ``radius`` property of the members that have ``vertical``

:meth:`tnsgrt.structure.Structure.set_member_properties` can also be used
to set multiple values at the same time, as in

.. code-block:: python

    from tnsgrt.utils import Colors

    s.set_member_properties(s.get_members_by_tag('vertical'),
                          'facecolor', Colors.GREEN.value,
                          'edgecolor', Colors.GREEN.value,
                          'mass', 2)

Retrieving the properties confirm the changes:

.. code-block:: python

    s.get_member_properties(s.get_members_by_tag('vertical'), 'radius', 'mass', 'facecolor', 'edgecolor')

in the dataframe::

        radius  mass              facecolor              edgecolor
     0    0.04   2.0  (0.466, 0.674, 0.188)  (0.466, 0.674, 0.188)
     2    0.04   2.0  (0.466, 0.674, 0.188)  (0.466, 0.674, 0.188)

These changes are also reflected on the structure's plot generated by
the following code.

.. code-block:: python

    plotter = MatplotlibPlotter()
    plotter.plot(s)
    fig, ax = plotter.get_handles()
    ax.view_init(90,-90)
    ax.axis('equal')
    ax.axis('off')
    plt.show()

.. image:: /images/planar2.png
  :scale: 50%