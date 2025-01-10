.. py:currentmodule:: emat


Meta-Model API
==============

The :class:`MetaModel` wraps the code required for automatically implementing
metamodels.  The resulting object is callable in the expected manner for a
function that is attached to a
PythonCoreModel (i.e., accepting keyword arguments to set inputs, and returning
a dictionary of named performance measure outputs).

It is expected that a user will not instatiate a :class:`MetaModel` directly itself,
but instead create them implicitly through other tools.
MetaModels can be created from other existing core models using the
:meth:`create_metamodel_from_design <model.AbstractCoreModel.create_metamodel_from_design>`
or :meth:`create_metamodel_from_data <model.AbstractCoreModel.create_metamodel_from_data>`
methods of a core model, or by using the :func:`create_metamodel` function, which can
create a MetaModel directly from a scope and experimental results, without requiring
a core model instance.  Each of these functions returns a :class:`PythonCoreModel` that
already wraps the MetaModel in an interface ready for use with other TMIP-EMAT tools,
so that in typical cases the user does not need to interact with or know anything
about the :class:`MetaModel` class itself, unless they care to dive in to the underlying
core or mathematical structures.

.. autofunction:: create_metamodel

.. autoclass:: emat.MetaModel
    :members:
    :special-members: __call__

