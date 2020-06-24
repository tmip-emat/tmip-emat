
.. py:currentmodule:: emat.model

Files-Based Model API
---------------------

The :class:`FilesCoreModel` class defines a common implementation system for
bespoke models that generate performance measure attributes that can be read from
one or more files on disk after a model run.  Many of the abstract methods
defined in the :class:`AbstractCoreModel` remain to be overloaded, but
a standard `load_measures` implementation is defined here, along with a
default implementation of `get_experiment_archive_path` and `post_process`.

.. autoclass:: FilesCoreModel
    :show-inheritance:
    :members:
    :exclude-members:
        setup, run, load_measures, post_process, get_experiment_archive_path, archive,
        run_model, model_init, load_archived_measures, add_parser

.. automethod:: FilesCoreModel.add_parser
.. automethod:: FilesCoreModel.load_measures
.. automethod:: FilesCoreModel.load_archived_measures

Parsing Files
~~~~~~~~~~~~~

The :meth:`FilesCoreModel.add_parser` method accepts :class:`FileParser` objects,
which can be used to read performance measures from individual files.  For an
illustration of how to use parsers, see the source code for the :class:`GBNRTCModel`.

.. autoclass:: emat.model.core_files.parsers.FileParser
    :members:

.. autoclass:: emat.model.core_files.parsers.TableParser
    :show-inheritance:
    :members:

.. autoclass:: emat.model.core_files.parsers.MappingParser
    :show-inheritance:
    :members:

.. autoclass:: emat.model.core_files.parsers.Getter

