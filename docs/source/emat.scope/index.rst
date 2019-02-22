
.. py:currentmodule:: emat

Exploratory Scoping
===================

.. toctree::
    :hidden:

    scope
    parameters
    measures
    boxes



.. rubric:: :doc:`Scope <scope>`

The exploratory |Scope| provides a high-level
group of instructions for what inputs and outputs a model
provides, and what ranges and/or distributions of these inputs
will be considered in an exploratory analysis.


.. rubric:: :doc:`Parameters <parameters>`

A |Parameter| is used to provide scoping information for
a single model input.  This can be a constant, an exogenous
uncertainty, or a policy lever.


.. rubric:: :doc:`Performance Measures <measures>`

A |Measure| is used to provide scoping information for
a single model output.


.. rubric:: :doc:`Boxes <boxes>`

A |Box| represents a constrained subset of experiments, containing
only those :term:`cases <case>` that meet certain restrictions.  These
restrictions are expressed as limited ranges on a particular set of named
|parameters| or |measures|.  A |Box| can also
designate a set of relevant features, which are not themselves constrained
but should be considered in any analytical report developed based on that
|Box|.




.. |Scope| replace:: :class:`Scope`
.. |Parameter| replace:: :class:`Parameter`
.. |Measure| replace:: :class:`Measure`
.. |parameters| replace:: :term:`parameters <parameter>`
.. |measures| replace:: :term:`measures <measure>`
.. |Box| replace:: :class:`Box`
