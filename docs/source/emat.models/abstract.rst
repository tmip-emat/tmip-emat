
.. py:currentmodule:: emat.model

Basic EMAT Model API
--------------------

.. autoclass:: emat.model.AbstractCoreModel
    :show-inheritance:
    :members:
    :inherited-members:
    :exclude-members: create_metamodel_from_data,
        create_metamodel_from_design,
        setup, run,
        run_experiments,
        robust_optimize, read_experiments,
        read_experiment_parameters, read_experiment_measures,
        load_measures, post_process, get_experiment_archive_path, archive,
        ensure_dtypes, get_feature_scores, design_experiments


Abstract Methods
~~~~~~~~~~~~~~~~

The interface for these methods is defined in this abstract base class,
but any implementation must provide implementation-specific overrides
for each of these methods.

.. automethod:: AbstractCoreModel.setup
.. automethod:: AbstractCoreModel.run
.. automethod:: AbstractCoreModel.load_measures
.. automethod:: AbstractCoreModel.post_process
.. automethod:: AbstractCoreModel.get_experiment_archive_path
.. automethod:: AbstractCoreModel.archive


Data Management
~~~~~~~~~~~~~~~

.. automethod:: AbstractCoreModel.read_experiments
.. automethod:: AbstractCoreModel.read_experiment_parameters
.. automethod:: AbstractCoreModel.read_experiment_measures
.. automethod:: AbstractCoreModel.ensure_dtypes


Model Execution
~~~~~~~~~~~~~~~

Assuming that the abstract methods outlined above are properly implemented,
these model execution methods should not need to be overridden.

.. automethod:: AbstractCoreModel.design_experiments
.. automethod:: AbstractCoreModel.run_experiments
.. automethod:: AbstractCoreModel.robust_optimize
.. automethod:: AbstractCoreModel.robust_evaluate
.. automethod:: AbstractCoreModel.get_feature_scores


Meta-Model Construction
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: AbstractCoreModel.create_metamodel_from_data
.. automethod:: AbstractCoreModel.create_metamodel_from_design