.. py:currentmodule:: emat

Meta-Models
===========

The :class:`MetaModel` wraps the code required for automatically implementing
metamodels.  MetaModels are created using the
:meth:`create_metamodel_from_design <model.AbstractCoreModel.create_metamodel_from_design>`
or :meth:`create_metamodel_from_data <model.AbstractCoreModel.create_metamodel_from_data>`
methods of a core model.

.. autoclass:: emat.MetaModel
    :members:
    :special-members: __call__


Meta-Model Regression
---------------------

TMIP-EMAT currently implements meta-modeling using a de-trended multi-target
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
