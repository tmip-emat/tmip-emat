
.. py:currentmodule:: emat.database


Basic EMAT Database API
=======================

.. autoclass:: Database
    :show-inheritance:
    :members:
    :exclude-members:
        read_scope_names,
        write_scope,
        read_scope,
        delete_scope,
        read_uncertainties,
        read_levers,
        read_constants,
        read_measures,
        read_box,
        read_box_names,
        read_box_parent_name,
        read_box_parent_names,
        read_boxes,
        write_box,
        write_boxes,
        read_design_names,
        delete_experiments,
        write_experiment_parameters,
        write_experiment_parameters_1,
        write_experiment_measures,
        write_experiment_all,
        read_experiment_parameters,
        read_experiment_measures,
        read_experiment_all,
        read_experiment_ids,
        get_new_metamodel_id,
        write_metamodel,
        read_metamodel_ids,
        read_metamodel


Scopes
------

.. automethod:: Database.write_scope
.. automethod:: Database.read_scope
.. automethod:: Database.delete_scope
.. automethod:: Database.read_scope_names


Scope Features
~~~~~~~~~~~~~~

.. automethod:: Database.read_uncertainties
.. automethod:: Database.read_levers
.. automethod:: Database.read_constants
.. automethod:: Database.read_measures


Boxes
-----

.. automethod:: Database.write_box
.. automethod:: Database.write_boxes
.. automethod:: Database.read_box
.. automethod:: Database.read_boxes
.. automethod:: Database.read_box_names
.. automethod:: Database.read_box_parent_name
.. automethod:: Database.read_box_parent_names


Experiments
-----------

.. automethod:: Database.write_experiment_parameters
.. automethod:: Database.write_experiment_parameters_1
.. automethod:: Database.write_experiment_measures
.. automethod:: Database.write_experiment_all
.. automethod:: Database.read_experiment_parameters
.. automethod:: Database.read_experiment_measures
.. automethod:: Database.read_experiment_all
.. automethod:: Database.read_experiment_ids
.. automethod:: Database.read_design_names
.. automethod:: Database.delete_experiments



Meta-Models
-----------

.. automethod:: Database.get_new_metamodel_id
.. automethod:: Database.write_metamodel
.. automethod:: Database.read_metamodel_ids
.. automethod:: Database.read_metamodel

