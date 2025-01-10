.. py:currentmodule:: emat


Boxes
=====

A |Box| represents a constrained subset of experiments, containing
only those :term:`cases <case>` that meet certain restrictions.  These
restrictions are expressed as limited ranges on a particular set of named
|parameters| or |measures|.  A |Box| can also
designate a set of relevant features, which are not themselves constrained
but should be considered in any analytical report developed based on that
|Box|.

A |Box| can be created manually as part of the scoping process. For example,
the initially defined scope may be too broad or may include more uncertainties
or policy levers than an analyst wishes to study at a particular time.  For this
situation, a |Box| can be used to define a more limited scope for study, by
limiting the ranges of some set of |parameters|.  It is possible to effectively
remove parameters from analysis by setting the minimum and maximum values to
the same value (for scalar parameters) or by limiting the allowed set of values
to a single value (for categorical parameters).  It is also possible to include
|measures| in the set of restrictions, which results in a different kind of
exploratory analysis, where the analyst can undertake scenario discovery, to
analyze the variety of input values that might lead to certain kinds of output
values.

Box
---
.. autoclass:: Box
    :member-order: bysource
    :members: scope,
        uncertainty_thresholds, lever_thresholds,
        measure_thresholds

.. automethod:: Box.set_lower_bound
.. automethod:: Box.set_upper_bound
.. automethod:: Box.set_bounds
.. automethod:: Box.add_to_allowed_set
.. automethod:: Box.remove_from_allowed_set
.. automethod:: Box.replace_allowed_set

Boxes
-----

.. autoclass:: Boxes
    :show-inheritance:
    :members:


ChainedBox
----------

.. autoclass:: ChainedBox
    :member-order: bysource
    :members:
        uncertainty_thresholds, lever_thresholds,
        measure_thresholds

.. automethod:: ChainedBox.uncertainty_thresholds
.. automethod:: ChainedBox.lever_thresholds
.. automethod:: ChainedBox.measure_thresholds

.. autoclass:: Bounds
    :show-inheritance:
    :members:




.. |Scope| replace:: :class:`Scope`
.. |Parameter| replace:: :class:`Parameter`
.. |Measure| replace:: :class:`Measure`
.. |parameters| replace:: :term:`parameters <parameter>`
.. |measures| replace:: :term:`measures <measure>`
.. |Box| replace:: :class:`Box`
