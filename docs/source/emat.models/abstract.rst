
.. py:currentmodule:: emat.model

Basic EMAT Model API
--------------------

.. autoclass:: emat.model.AbstractCoreModel
    :show-inheritance:
    :members:
    :inherited-members:
    :exclude-members: create_metamodel_from_data,
        create_metamodel_from_design, create_metamodel_from_designs,
        setup, run, run_model, io_experiment,
        run_experiments, run_reference_experiment,
        robust_optimize, robust_evaluate, read_experiments,
        read_experiment_parameters, read_experiment_measures,
        load_measures, post_process, get_experiment_archive_path, archive,
        ensure_dtypes, get_feature_scores, feature_scores, design_experiments,
        reset_model, retrieve_output, optimize, model_init,
        model_init, initialized, as_dict, cleanup


Abstract Methods
~~~~~~~~~~~~~~~~

The interface for these methods is defined in this abstract base class,
but any implementation must provide implementation-specific overrides
for each of these methods.

.. note::
    An important feature of overriding these functions is that the
    function signature (what arguments and types each function accepts,
    and what types it returns) should not be changed, even though
    technically Python itself allows doing so.

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
.. automethod:: AbstractCoreModel.io_experiment
.. automethod:: AbstractCoreModel.run_reference_experiment
.. automethod:: AbstractCoreModel.optimize
.. automethod:: AbstractCoreModel.robust_optimize
.. automethod:: AbstractCoreModel.robust_evaluate
.. automethod:: AbstractCoreModel.feature_scores


Meta-Model Construction
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: AbstractCoreModel.create_metamodel_from_data
.. automethod:: AbstractCoreModel.create_metamodel_from_design
.. automethod:: AbstractCoreModel.create_metamodel_from_designs