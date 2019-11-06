.. py:currentmodule:: emat.model


Meta Models
===========

.. toctree::
    :hidden:

    creation
    api
    regression


.. rubric:: :doc:`Meta-Model Creation <creation>`

Meta-model objects are created automatically by TMIP-EMAT
using the :func:`create_metamodel` function.


.. rubric:: :doc:`Meta-Model API <api>`

The :class:`MetaModel` provides a basic interface structure
for interacting with meta-models used in exploratory analysis. The
class interface is designed to mimic a typical Python-based model,
passing values to and from the underlying regression model transparently.
Although the API for the meta-model class is documented here,
typically meta-model objects are created automatically by TMIP-EMAT
using the :func:`create_metamodel` function.


.. rubric:: :doc:`Meta-Model Regression <regression>`

Documentation is provided for the `scikit-learn` style de-trended
multi-target Gaussian process regression model, which is used by default
in TMIP-EMAT.
