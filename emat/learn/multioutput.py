
import numpy as np
from abc import ABCMeta, abstractmethod

from sklearn.multioutput import MultiOutputRegressor as _MultiOutputRegressor
from .frameable import FrameableMixin
from .model_selection import CrossValMixin

from sklearn.multioutput import _partial_fit_estimator, _fit_estimator
from sklearn.base import BaseEstimator, RegressorMixin, is_classifier, clone
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted, has_fit_parameter
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils._joblib import Parallel, delayed

class CompositeCVMixin:

	def lock_best_estimator(self, dry_run=False):
		"""
		Convert GridSearchCV components to their best-fitted version.

		Parameters
		----------
		dry_run : bool, default False
			If true, only report the number of best_estimator_s
			found, but do not actually overwrite anything.

		Returns
		-------
		n_changes
			The number of modified component estimators.
		"""
		n_changes = 0
		estimators = []
		for e in self.estimators_:
			if hasattr(e, 'best_estimator_'):
				estimators.append(e.best_estimator_)
				n_changes += 1
			else:
				estimators.append(e)
		if not dry_run:
			self.estimators_ = estimators
			self.estimators = [clone(e) for e in self.estimators_]
		return n_changes


class MultiOutputRegressor(_MultiOutputRegressor, FrameableMixin, CrossValMixin):

	def fit(self, X, y, sample_weight=None):
		self._pre_fit(X,y)
		return super().fit(X, y, sample_weight=sample_weight)

	def predict(self, X):
		y = super().predict(X)
		y = self._post_predict(X,y)
		return y

	def take_best_estimators(self):
		"""
		Convert GridSearchCV components to their best-fitted version.

		Parameters
		----------
		dry_run : bool, default False
			If true, only report the number of best_estimator_s
			found, but do not actually overwrite anything.

		Returns
		-------
		n_changes
			The number of modified component estimators.
		"""
		n_changes = 0
		estimators = []
		for e in self.estimators_:
			if hasattr(e, 'best_estimator_'):
				estimators.append(e.best_estimator_)
				n_changes += 1
			else:
				estimators.append(e)
		result = MultiOutputRegressorDiverse([clone(e) for e in estimators], n_jobs=self.n_jobs)
		result.estimators_ = estimators
		return result

class MultiOutputEstimatorDiverse(BaseEstimator, CompositeCVMixin, metaclass=ABCMeta):

	@abstractmethod
	def __init__(self, estimators, n_jobs=None):
		self.estimators = estimators
		self.n_jobs = n_jobs

	@if_delegate_has_method('estimators')
	def partial_fit(self, X, y, classes=None, sample_weight=None):
		"""Incrementally fit the model to data.
		Fit a separate model for each output variable.

		Parameters
		----------
		X : (sparse) array-like, shape (n_samples, n_features)
			Data.

		y : (sparse) array-like, shape (n_samples, n_outputs)
			Multi-output targets.

		classes : list of numpy arrays, shape (n_outputs)
			Each array is unique classes for one output in str/int
			Can be obtained by via
			``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where y is the
			target matrix of the entire dataset.
			This argument is required for the first call to partial_fit
			and can be omitted in the subsequent calls.
			Note that y doesn't need to contain all labels in `classes`.

		sample_weight : array-like, shape = (n_samples) or None
			Sample weights. If None, then samples are equally weighted.
			Only supported if the underlying regressor supports sample
			weights.

		Returns
		-------
		self : object
		"""
		X, y = check_X_y(X, y,
						 multi_output=True,
						 accept_sparse=True)

		if y.ndim == 1:
			raise ValueError("y must have at least two dimensions for "
							 "multi-output regression but has only one.")

		for i in range(y.shape[1]):
			if (sample_weight is not None and
					not has_fit_parameter(self.estimators[i], 'sample_weight')):
				raise ValueError(f"Underlying estimator {i} does not support"
								 " sample weights.")

		first_time = not hasattr(self, 'estimators_')

		self.estimators_ = Parallel(n_jobs=self.n_jobs)(
			delayed(_partial_fit_estimator)(
				self.estimators_[i] if not first_time else self.estimators[i],
				X, y[:, i],
				classes[i] if classes is not None else None,
				sample_weight, first_time) for i in range(y.shape[1]))
		return self

	def fit(self, X, y, sample_weight=None):
		""" Fit the model to data.
		Fit a separate model for each output variable.

		Parameters
		----------
		X : (sparse) array-like, shape (n_samples, n_features)
			Data.

		y : (sparse) array-like, shape (n_samples, n_outputs)
			Multi-output targets. An indicator matrix turns on multilabel
			estimation.

		sample_weight : array-like, shape = (n_samples) or None
			Sample weights. If None, then samples are equally weighted.
			Only supported if the underlying regressor supports sample
			weights.

		Returns
		-------
		self : object
		"""
		for i in range(y.shape[1]):
			if not hasattr(self.estimators[i], "fit"):
				raise ValueError(f"The base estimator {i} should implement"
								 " a fit method")

		X, y = check_X_y(X, y,
						 multi_output=True,
						 accept_sparse=True)

		if is_classifier(self):
			check_classification_targets(y)

		if y.ndim == 1:
			raise ValueError("y must have at least two dimensions for "
							 "multi-output regression but has only one.")

		for i in range(y.shape[1]):
			if (sample_weight is not None and
					not has_fit_parameter(self.estimators[i], 'sample_weight')):
				raise ValueError(f"Underlying estimator {i} does not support"
								 " sample weights.")

		self.estimators_ = Parallel(n_jobs=self.n_jobs)(
			delayed(_fit_estimator)(
				self.estimators[i], X, y[:, i], sample_weight)
			for i in range(y.shape[1]))
		return self

	def predict(self, X):
		"""Predict multi-output variable using a model
		 trained for each target variable.

		Parameters
		----------
		X : (sparse) array-like, shape (n_samples, n_features)
			Data.

		Returns
		-------
		y : (sparse) array-like, shape (n_samples, n_outputs)
			Multi-output targets predicted across multiple predictors.
			Note: Separate models are generated for each predictor.
		"""
		check_is_fitted(self, 'estimators_')
		for i,e in enumerate(self.estimators):
			if not hasattr(e, "predict"):
				raise ValueError(f"The base estimator {i} should implement"
								 " a predict method")

		X = check_array(X, accept_sparse=True)

		y = Parallel(n_jobs=self.n_jobs)(
			delayed(e.predict)(X)
			for e in self.estimators_)

		return np.asarray(y).T

	def _more_tags(self):
		return {'multioutput_only': True}


class MultiOutputRegressorDiverse(
	MultiOutputEstimatorDiverse,
	RegressorMixin,
	FrameableMixin,
	CrossValMixin,
):
	"""
	Multi target regression

	This strategy consists of fitting one regressor per target.
	Unlike the default MultiOutputRegressor, the components
	do not need to have the same hyperparameters, or indeed even
	be the same Regressor type.

	Parameters
	----------
	estimators : estimator objects
		An collection of estimator objects implementing `fit` and `predict`.

	n_jobs : int or None, optional (default=None)
		The number of jobs to run in parallel for `fit`.
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>`
		for more details.

		When individual estimators are fast to train or predict
		using `n_jobs>1` can result in slower performance due
		to the overhead of spawning processes.

	Attributes
	----------
	estimators_ : list of ``n_output`` estimators
		Estimators used for predictions.
	"""

	def __init__(self, estimators, n_jobs=None):
		super().__init__(estimators, n_jobs)

	@if_delegate_has_method('estimators')
	def partial_fit(self, X, y, sample_weight=None):
		"""Incrementally fit the model to data.
		Fit a separate model for each output variable.

		Parameters
		----------
		X : (sparse) array-like, shape (n_samples, n_features)
			Data.

		y : (sparse) array-like, shape (n_samples, n_outputs)
			Multi-output targets.

		sample_weight : array-like, shape = (n_samples) or None
			Sample weights. If None, then samples are equally weighted.
			Only supported if the underlying regressor supports sample
			weights.

		Returns
		-------
		self : object
		"""
		super().partial_fit(
			X, y, sample_weight=sample_weight)

	def fit(self, X, y, sample_weight=None):
		self._pre_fit(X,y)
		return super().fit(X, y, sample_weight=sample_weight)

	def predict(self, X):
		y = super().predict(X)
		y = self._post_predict(X,y)
		return y
