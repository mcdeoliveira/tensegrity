Working with Structures
=======================

The prisms seen in the quick start are members of :class:`tnsgrt.structure.Structure`. In fact,
:class:`tnsgrt.prism.Prism` overloads only the constructor of :class:`tnsgrt.structure.Structure`, which provides most
of the functionality.

Construction
------------

Object of class :class:`tnsgrt.structure.Structure` are composed of *nodes* and *members*.
Nodes are `3 x n` numpy arrays:

.. code-block:: python

    import numpy as np
    nodes = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]]).transpose()

Members are *bars* or *strings*, which are specified by a `2 x m` array with the indices of the nodes that define the
members, as in:

.. code-block:: python

    members = np.array([[0,1],[1,2],[2,3],[3,0],[0,2],[1,3]]).transpose()

Nodes and members are then combined to build a :class:`tnsgrt.structure.Structure`:

.. code-block:: python

    from tnsgrt import Structure
    s = Structure(nodes, members, number_of_strings=4)

The parameter ``number_of_strings = 4`` means that the first four members are to be considered strings.

As before, the resulting structure can be plotted using :class:`tnsgrt.plotter.MatplotlibPlotter`:

.. code-block:: python

    from tnsgrt.plotter.matplotlib import MatplotlibPlotter
    plotter = MatplotlibPlotter()
    plotter.plot(s)
    fig, ax = plotter.get_handles()
    ax.view_init(-90,0)

to visualize the resulting planar tensegrity structure below:

.. image:: /images/planar1.png
  :scale: 50%

Equilibrium
-----------

Once a structure is built, one can perform various calculations. For example one can determine the member forces so that
the structure is in equilibrium in various cases.

The key method is :meth:`tnsgrt.structure.Structure.equilibrium`, which calculates the internal forces required to
maintain a structure in equilibrium. It is assumed that all nodes act as *ball joints*.

Unloaded
^^^^^^^^

In the *unloaded* case, no external forces are applied to the structure, and equilibrium is achieved by *pretensioning*
the structure. The result of calling :meth:`tnsgrt.structure.Structure.equilibrium`, as in

.. code-block:: python

    s.equilibrium()

is a set of *force coefficients*, which are forces divided by member length.

The result of the equilibrium calculation can be found in the member properties
``lambda_`` (the force coefficient) and ``force``:

.. code-block:: python

    s.member_properties[['lambda_', 'force']]

which, in this example, returns:

.. csv-table::
   :file: /tables/tab1.csv
   :header-rows: 1

Pretension is set so that the average of the magnitude of the force coefficient on all bars is equal to the parameter
``lambda_bar``, which is by default equal to one.

Loaded
^^^^^^

In this case equilibrium is sought in the presence of external forces, given as a `3 x n` array as the following one:

.. code-block:: python

    f = 0.125*np.array([[0,1,0],[0,-1,0],[0,-2,0],[0,2,0]]).transpose()

Each column is to be interpreted as a force vector to be applied at the corresponding node.

The external force array ``f`` can then be passed on to the method :meth:`tnsgrt.structure.Structure.equilibrium`:

.. code-block:: python

    s.equilibrium(f)

resulting in the new set of member forces and force coefficients:

.. code-block:: python

    s.member_properties[['lambda_', 'force']]

that returns:

.. csv-table::
   :file: /tables/tab2.csv
   :header-rows: 1

The following code produces a visualization of the applied forces superimposed on the structure:

.. code-block:: python

    plotter = MatplotlibPlotter()
    plotter.plot(s)
    fig, ax = plotter.get_handles()
    ax.quiver(s.nodes[0,:], s.nodes[1,:], s.nodes[2,:], f[0,:], f[1,:], f[2,:], arrow_length_ratio=0.2, color='g')
    ax.view_init(90,-90)
    ax.axis('off')
    plt.show()

resulting in a figure like

.. image:: /images/loaded.png
   :scale: 50%

The forces are represented by the green arrows.

When it is not possible to find a set of internal forces that satisfy the equilibrium conditions an Exception with a
message "could not find equilibrium" is produced. For example:

.. code-block:: python

    f = 0.125*np.array([[0,1,0],[0,-1,0],[0,-1,0],[0,2,0]]).transpose()
    s.equilibrium(f)

can not be in equilibrium.

Stiffness
---------

Once a structure is in equilibrium, its response to forces can be calculated in terms of its *stiffness matrix*. For
that it is necessary to characterize the members' geometry and material properties. The fundamental properties are the
member radius, and elasticity modulus:

.. code-block:: python

    s.member_properties[['radius', 'inner_radius', 'modulus']]

The current default values for such properties are:

.. csv-table::
   :file: /tables/tab3.csv
   :header-rows: 1

For calculating the stiffness matrix of a pretensioned structure, it is also necessary to know the member's force
coefficient and the derived member stiffness property. As seen before, the force coefficient and the force are obtained
during the equilibrium calculation:

.. code-block:: python

    s.equilibrium()
    s.member_properties[['lambda_', 'force', 'stiffness']]

which returns:

.. csv-table::
   :file: /tables/tab4.csv
   :header-rows: 1

Because the stiffness is a "derived" property, it does not get automatically populated, which can be done by calling
:meth:`tnsgrt.structure.Structure.update_member_properties`:

.. code-block:: python

    s.update_member_properties('stiffness')
    s.member_properties[['stiffness']]

to obtain:

.. csv-table::
   :file: /tables/tab5.csv
   :header-rows: 1

After setting the material properties, one can calculate the stiffness model associated with the current equilibrium:

.. code-block:: python

    stiffness, _, _ = s.stiffness()

For large models, the stiffness is stored and calculated as sparse arrays. However, for small models, such as this one,
the model is stored in dense arrays. The warning message can be suppressed by explicitly setting the parameter
``storage=dense``:

.. code-block:: python

    stiffness, _, _ = s.stiffness(storage='dense')

**WARNING:** setting ``storage='dense'`` for large models is not advised.

Rigid-body constraints
^^^^^^^^^^^^^^^^^^^^^^

The stiffness model can be used to calculate various quantities of interest. For example:

.. code-block:: python

    d, v = stiffness.eigs()

returns the eigenvalues and eigenvectors of the stiffness matrix. In this case, because there are no constraints in the
structure, we should expect to encounter various eigenvalues numerically close to zero:

.. code-block:: python

    d

returns::

    -6.237207e-09
    -4.329203e-10
     1.415459e-11
     9.183017e-10
     4.478545e-09
     7.290895e-09
     4.000000e+00
     3.141592e+07
     3.141593e+07
     3.141593e+07


Six of these are the so-called "rigid body modes," associated to rigid translations and rotations of the structure.
They can be "removed" by applying certain constraints to the set of allowed displacements. Enforcement of these
constraints can be done by passing the parameter ``apply_rigid_body_constraint=True`` when calculating the stiffness
model:

.. code-block:: python

    stiffness, _, _ = s.stiffness(storage='dense', apply_rigid_body_constraint=True)

To see that the six near zero eigenvalues of the stiffness matrix have been removed by the rigid body constraints
recalculate:

.. code-block:: python

    d, v = stiffness.eigs()
    d

to obtain::

    4.000000e+00
    3.141592e+07
    3.141593e+07
    3.141593e+07
    8.885766e+07
    1.202736e+08

Interestingly, in this case, there still remains one eigenvalue that is much smaller than the rest.
We shall deal with this small eigenvalue later.

For now, even though the smallest eigenvalue is small, the resulting stiffness matrix is not singular, and therefore
suitable for computing displacements. This time:

.. code-block:: python

    x = stiffness.displacements(f)
    x

successfully calculates the resulting approximate displacements::

    -2.20468248e-09, -2.20468248e-09,  2.20468248e-09,  2.20468248e-09
     1.77419161e-09, -1.77419161e-09, -5.75306493e-09,  5.75306493e-09
     4.02657501e-18, -4.02657481e-18,  4.02657460e-18, -4.02657419e-18

which can be visualized, after much enlargement, along with the applied forces in the figure:

.. image:: /images/stiffness1.png
    :scale: 50%

in which the forces are in green and the vectors indicating the resulting displacement are in yellow.
This figure is generated by the code:

.. code-block:: python

    X = f
    Y = 5e7*x

    plotter = MatplotlibPlotter()
    plotter.plot(s)
    fig, ax = plotter.get_handles()
    ax.quiver(s.nodes[0,:], s.nodes[1,:], s.nodes[2,:], X[0,:], X[1,:], X[2,:], arrow_length_ratio=.2, color='g')
    ax.quiver(s.nodes[0,:], s.nodes[1,:], s.nodes[2,:], Y[0,:], Y[1,:], Y[2,:], arrow_length_ratio=.2, color='y')
    ax.view_init(90,-90)
    ax.axis('off')
    plt.show()


Planar constraints
^^^^^^^^^^^^^^^^^^

Back to the small eigenvalue, which is sometimes associated with what is called a *soft mode*, in this case it appeared
because the structure is planar, and its ball joints offer little resistance to out-of-plane forces. Indeed, the
eigenvector associated with the eigenvalue is:

.. code-block:: python

    v[:,0].reshape((3, 4), order='F')

which equals::

     2.27657232e-16,  6.97069602e-17, -6.05872973e-17, -1.78599980e-16
    -1.50805456e-16,  4.09366810e-18,  1.79720993e-17,  5.96054627e-17
    -5.00000000e-01,  5.00000000e-01, -5.00000000e-01,  5.00000000e-01

which constitutes a pair of "couples" in the out-of-plane z-direction.

As with rigid body modes, constraining the node displacements to be planar "eliminates" such mode, as in:

.. code-block:: python

    stiffness, _, _ = s.stiffness(storage='dense', apply_rigid_body_constraint=True, apply_planar_constraint=True)

Resulting in a structure in which:

.. code-block:: python

    d, v = stiffness.eigs()
    d

equals::

    3.141592e+07
    3.141593e+07
    3.141593e+07
    8.885766e+07
    1.202736e+08

indicating that there are no soft modes.

Of course one should expect no impact in the displacements if the forces do not have out-of-plane components and:

.. code-block:: python

    x = stiffness.displacements(f)
    x

indeed returns displacements that are very similar to the ones calculated before.
