
.. py:currentmodule:: emat.model

Files-Based Model
-----------------

The :class:`FilesCoreModel` class defines a common implementation system for
bespoke models that generate performance measure attributes that can be read from
one or more files on disk after a model run.  Many of the abstract methods
defined in the :class:`AbstractCoreModel` remain to be overloaded, but
a standard `load_measures` implementation is defined here.

You can connect any bespoke transportation model to TMIP-EMAT as long
as it satisfies these requirements:

- You can prepare a model run by manipulating files from within Python,
  including tasks such as copying or renaming files, editing the text
  of script or configuration files, or executing one or more "command line"
  programs (e.g. batch files) with defined arguments.
- You can execute a model run, including any necessary post-processing,
  either directly from Python, or by executing one or more "command line"
  programs (e.g. batch files) with defined arguments.
- You can extract individual performance measures of interest by reading
  and parsing one or more output files (which can be output from the main
  model or any post-processing routines) from within Python.

To actually create the linkage between TMIP-EMAT and your bespoke transportation
model, you must define a new Python `class` that derives from the
:class:`FilesCoreModel` class, and at a minimum write implementations for
these methods:

- :meth:`AbstractCoreModel.setup`
- :meth:`AbstractCoreModel.run`
- :meth:`AbstractCoreModel.post_process`
- :meth:`AbstractCoreModel.get_experiment_archive_path`
- :meth:`AbstractCoreModel.archive`



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

.. autoclass:: emat.model.core_files.parsers.Getter


.. toctree::

    table_parse_example
