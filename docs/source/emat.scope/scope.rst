

.. py:currentmodule:: emat


Scope
=====

The exploratory :class:`Scope` provides a high-level
group of instructions for what inputs and outputs a model
provides, and what ranges and/or distributions of these inputs
will be considered in an exploratory analysis.

.. autoclass:: Scope

Read / Write
------------

.. automethod:: Scope.store_scope
.. automethod:: Scope.delete_scope
.. automethod:: Scope.dump
.. automethod:: Scope.duplicate


Feature Access
--------------

.. automethod:: Scope.get_constants
.. automethod:: Scope.get_uncertainties
.. automethod:: Scope.get_levers
.. automethod:: Scope.get_parameters
.. automethod:: Scope.get_measures

Names
~~~~~

.. automethod:: Scope.get_constant_names
.. automethod:: Scope.get_uncertainty_names
.. automethod:: Scope.get_lever_names
.. automethod:: Scope.get_parameter_names
.. automethod:: Scope.get_measure_names
.. automethod:: Scope.get_all_names

Other Attributes
~~~~~~~~~~~~~~~~

.. automethod:: Scope.get_dtype
.. automethod:: Scope.get_cat_values

Utilities
---------

.. automethod:: Scope.info
.. automethod:: Scope.n_factors
.. automethod:: Scope.n_sample_factors
.. automethod:: Scope.design_experiments
.. automethod:: Scope.ensure_dtypes


