.. py:currentmodule:: emat

Meta-Models
===========

The :class:`MetaModel` wraps the code required for automatically implementing
metamodels.  The resulting object is callable in the expected manner for a
PythonCoreModel (i.e., accepting keyword arguments to set inputs, and returning
a dictionary of named performance measure outputs).

.. autoclass:: emat.MetaModel
    :members:
    :special-members: __call__

MetaModels can be created directly from other core models using the
:meth:`create_metamodel_from_design <model.AbstractCoreModel.create_metamodel_from_design>`
or :meth:`create_metamodel_from_data <model.AbstractCoreModel.create_metamodel_from_data>`
methods of a core model, or by using the :func:`create_metamodel` function, which can
create a MetaModel directly from a scope and experimental results, without requiring
a core model instance.  Each of these functions returns a :class:`PythonCoreModel` that
already wraps the MetaModel in an interface ready for use with other TMIP-EMAT tools.

.. autofunction:: create_metamodel

Meta-Model Regression
---------------------

TMIP-EMAT currently implements meta-modeling by default using a de-trended multi-target
Gaussian process regression model.  The contents of this meta-model is provided
transparently for inspection and review as desired, and documention is provided
here to facilitate this.  However, future version of TMIP-EMAT may introduce
more sophisticated meta-models, and code to utilize meta-models should not
rely on the underlying regression model itself.

.. autoclass:: emat.multitarget.DetrendedMultipleTargetRegression

.. automethod:: emat.multitarget.DetrendedMultipleTargetRegression.fit
.. automethod:: emat.multitarget.DetrendedMultipleTargetRegression.predict
.. automethod:: emat.multitarget.DetrendedMultipleTargetRegression.detrend_predict
.. automethod:: emat.multitarget.DetrendedMultipleTargetRegression.residual_predict


.. automethod:: emat.multitarget.DetrendedMultipleTargetRegression.cross_val_scores
.. automethod:: emat.multitarget.DetrendedMultipleTargetRegression.detrend_scores


.. autofunction:: emat.learn.LinearAndGaussian