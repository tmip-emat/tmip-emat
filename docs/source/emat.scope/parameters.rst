.. py:currentmodule:: emat

.. _input_parameters:

Parameters
----------

A :class:`Parameter` is used to provide scoping information for
a single model input.  This can be an exogenous
uncertainty, or a policy lever.  Invariant input (i.e., constants)
can be represented with a :class:`Constant`, which exposes a very
similar set of attributes and methods, but doesn't allow the
value to vary. Neither class object should be instantiated directly,
but instead use the :func:`make_parameter` function, which
will create an object of the appropriate (sub)class.


.. autofunction:: emat.make_parameter


.. autoclass:: emat.Parameter
    :show-inheritance:
    :members:
    :inherited-members:

.. autoclass:: emat.Constant
    :show-inheritance:
    :members:
    :inherited-members:


Float-Valued Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: emat.scope.parameter.RealParameter
    :show-inheritance:
    :members:
    :inherited-members:

Integer Parameters
~~~~~~~~~~~~~~~~~~

.. autoclass:: emat.scope.parameter.IntegerParameter
    :show-inheritance:
    :members:
    :inherited-members:

Boolean Parameters
~~~~~~~~~~~~~~~~~~

.. autoclass:: emat.scope.parameter.BooleanParameter
    :show-inheritance:
    :members:
    :inherited-members:

Categorical Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: emat.scope.parameter.CategoricalParameter
    :show-inheritance:
    :members:
    :inherited-members:

