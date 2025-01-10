
.. py:currentmodule:: emat.database

SQLite Database
===============

.. autoclass:: SQLiteDB
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

.. automethod:: SQLiteDB.write_scope
.. automethod:: SQLiteDB.read_scope
.. automethod:: SQLiteDB.delete_scope
.. automethod:: SQLiteDB.read_scope_names


Scope Features
~~~~~~~~~~~~~~

.. automethod:: SQLiteDB.read_uncertainties
.. automethod:: SQLiteDB.read_levers
.. automethod:: SQLiteDB.read_constants
.. automethod:: SQLiteDB.read_measures


Boxes
-----

.. automethod:: SQLiteDB.write_box
.. automethod:: SQLiteDB.write_boxes
.. automethod:: SQLiteDB.read_box
.. automethod:: SQLiteDB.read_boxes
.. automethod:: SQLiteDB.read_box_names
.. automethod:: SQLiteDB.read_box_parent_name
.. automethod:: SQLiteDB.read_box_parent_names


Experiments
-----------

.. automethod:: SQLiteDB.write_experiment_parameters
.. automethod:: SQLiteDB.write_experiment_parameters_1
.. automethod:: SQLiteDB.write_experiment_measures
.. automethod:: SQLiteDB.write_experiment_all
.. automethod:: SQLiteDB.read_experiment_parameters
.. automethod:: SQLiteDB.read_experiment_measures
.. automethod:: SQLiteDB.read_experiment_all
.. automethod:: SQLiteDB.read_experiment_ids
.. automethod:: SQLiteDB.read_design_names
.. automethod:: SQLiteDB.delete_experiments



Meta-Models
-----------

.. automethod:: SQLiteDB.get_new_metamodel_id
.. automethod:: SQLiteDB.write_metamodel
.. automethod:: SQLiteDB.read_metamodel_ids
.. automethod:: SQLiteDB.read_metamodel

