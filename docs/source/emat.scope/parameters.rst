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
    :exclude-members: from_dist

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
    :exclude-members: from_dist

Continuous Distributions
++++++++++++++++++++++++

Float-valued parameters can use any continuous distribution available
in :any:`scipy.stats`.  For convenience, a few extra (and simplified)
distributions are available in the ``emat.util.distributions``
module.

.. toctree::
    continuous_distributions

Integer Parameters
~~~~~~~~~~~~~~~~~~

.. autoclass:: emat.scope.parameter.IntegerParameter
    :show-inheritance:
    :members:
    :inherited-members:
    :exclude-members: from_dist

Discrete Distributions
++++++++++++++++++++++

Integer-valued parameters can use any discrete distribution available
in :any:`scipy.stats`.  Note that actually using a discrete distribution
is required, one cannot use a continuous distribution that loosely
approximates a discrete distribution.  The only exception to this rule
is for "uniform", which is technically a continuous distribution,
but is transparently interpreted by EMAT as an equivalently-bounded "randint".

Boolean Parameters
~~~~~~~~~~~~~~~~~~

.. autoclass:: emat.scope.parameter.BooleanParameter
    :show-inheritance:
    :members:
    :inherited-members:
    :exclude-members: from_dist

Categorical Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: emat.scope.parameter.CategoricalParameter
    :show-inheritance:
    :members:
    :inherited-members:
    :exclude-members: from_dist

