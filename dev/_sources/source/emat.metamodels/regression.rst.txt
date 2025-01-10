.. py:currentmodule:: emat


Meta-Model Regression
=====================

TMIP-EMAT currently implements meta-modeling by default using a de-trended multi-target
Gaussian process regression model.  The contents of this meta-model is provided
transparently for inspection and review as desired, and documentation is provided
here to facilitate this.  Alternate regressors can be used by passing any `scikit-learn`
compatible regressor object as the `regressor` argument in the `create_metamodel` function.

.. autofunction:: emat.learn.LinearAndGaussian