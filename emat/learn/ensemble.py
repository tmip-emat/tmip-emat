
import numpy as np
from sklearn.ensemble import VotingRegressor as _VotingRegressor
from sklearn.utils.validation import check_is_fitted
from .frameable import FrameableMixin
from .model_selection import CrossValMixin

class VotingRegressor(
	_VotingRegressor,
	FrameableMixin,
	CrossValMixin,
):

	def fit(self, X, y, sample_weight=None):
		self._pre_fit(X, y)
		super().fit(X, y, sample_weight=sample_weight)

	def _predict(self, X):
		"""Collect results from clf.predict calls. """
		return np.asarray([np.asarray(clf.predict(X)) for clf in self.estimators_]).T

	def predict(self, X):
		"""
		Predict regression target for X.

		The predicted regression target of an input sample is computed as the
		mean predicted regression targets of the estimators in the ensemble.

		Parameters
		----------
		X : {array-like, sparse matrix} of shape (n_samples, n_features)
			The input samples.

		Returns
		-------
		y : array of shape (n_samples, n_targets)
			The predicted values.
		"""
		check_is_fitted(self, "estimators_")
		y_hat = np.average(
			self._predict(X).T,
			axis=0,
			weights=self._weights_not_none,
		)
		return self._post_predict(X, y_hat)

