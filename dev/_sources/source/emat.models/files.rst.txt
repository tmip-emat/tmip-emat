
.. py:currentmodule:: emat.model

Files-Based Model
-----------------

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

The ability to tap into other programs via the command line allows TMIP-EMAT
to be used with nearly any transportation modeling tool, including most
commercial software.

To actually create the linkage between TMIP-EMAT and your bespoke transportation
model, you must define a new Python `class` that derives from the
:class:`FilesCoreModel` class, and at a minimum write implementations for
these methods:

- :meth:`AbstractCoreModel.setup`
- :meth:`AbstractCoreModel.run`
- :meth:`AbstractCoreModel.archive`

The :class:`FilesCoreModel` class also includes default implementations for
`get_experiment_archive_path` and `post_process`, but it may be appropriate
to overload these methods into a custom class as well.

A walk-through for building and using a :class:`FilesCoreModel` subclass
in your own module is provided below, although with a more detailed
reference for the methods defined for the :class:`FilesCoreModel` class.

.. toctree::

    interface-walkthrough
    table_parse_example
    mapping_parse_example
    files-api
