

from typing import Sequence
from sklearn.base import RegressorMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict
from .frameable import FrameableMixin


class StackedRegressor(BaseEstimator, RegressorMixin, FrameableMixin):
	"""
	A stack of regressors.

	Each regressor is fit sequentially, and the residual
	"""

	def __init__(
			self,
			estimators,
			use_cv_predict=False,
	):
		self.estimators = estimators
		self.use_cv_predict = use_cv_predict

	def _use_cv_predict_n(self, n):
		if isinstance(self.use_cv_predict, Sequence):
			return self.use_cv_predict[n]
		return self.use_cv_predict

	def fit(self, X, Y, sample_weight=None):

		if sample_weight is not None:
			raise NotImplementedError
		self._pre_fit(X, Y)
		self.estimators_ = []
		Y_ = Y
		for n,e in enumerate(self.estimators):
			e_ = clone(e)
			e_.fit(X, Y_)
			self.estimators_.append(e_)
			if n+1 < len(self.estimators):
				if self._use_cv_predict_n(n):
					Y_ = Y_ - self._post_predict(X,cross_val_predict(e_,X))
				else:
					Y_ = Y_ - self._post_predict(X,e_.predict(X))

	def predict(self, X):
		Yhat = self.estimators_[0].predict(X)
		for e_ in self.estimators_[1:]:
			Yhat += e_.predict(X)
		Yhat = self._post_predict(X, Yhat)
		return Yhat

