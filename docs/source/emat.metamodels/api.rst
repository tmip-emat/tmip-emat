.. py:currentmodule:: emat


Meta-Model API
==============

The :class:`MetaModel` wraps the code required for automatically implementing
metamodels.  The resulting object is callable in the expected manner for a
PythonCoreModel (i.e., accepting keyword arguments to set inputs, and returning
a dictionary of named performance measure outputs).

.. autoclass:: emat.MetaModel
    :members:
    :special-members: __call__

