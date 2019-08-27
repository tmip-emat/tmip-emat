

from typing import Sequence
from sklearn.base import RegressorMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict
from .frameable import FrameableMixin
from .model_selection import CrossValMixin

class StackedRegressor(BaseEstimator, RegressorMixin, FrameableMixin, CrossValMixin):
	"""
	A stack of regressors.

	Each regressor is fit sequentially, and the remaining residual
	is the target of the next model in the chain.
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
		return self

	def predict(self, X, tier=9999):
		"""
		Generate predictions from a set of exogenous data.

		Parameters
		----------
		X : array-like, prefer pandas.DataFrame
			Exogenous data.
		tier : int, default 9999


		"""
		Yhat = self.estimators_[0].predict(X)
		for n_, e_ in enumerate(self.estimators_[1:]):
			if n_ < tier:
				Yhat += e_.predict(X)
		Yhat = self._post_predict(X, Yhat)
		return Yhat


def LinearInteractAndGaussian(
		k=None,
		degree=2,
		fit_intercept=True,
		n_jobs=None,
		stats_on_fit=True,
		kernel_generator=None,
		alpha=1e-10,
		optimizer="fmin_l_bfgs_b",
		n_restarts_optimizer=250,
		normalize_y=False,
		standardize_before_fit=True,
		copy_X_train=True,
		random_state=None,
		use_cv_predict=False,
):
	from .linear_model import LinearRegression_KBestPoly
	from .anisotropic import AnisotropicGaussianProcessRegressor
	return StackedRegressor(
		[
			LinearRegression_KBestPoly(
				k=k,
				degree=degree,
				fit_intercept=fit_intercept,
				copy_X=False,
				n_jobs=n_jobs,
				stats_on_fit=stats_on_fit,
			),
			AnisotropicGaussianProcessRegressor(
				kernel_generator=kernel_generator,
				alpha=alpha,
				optimizer=optimizer,
				n_restarts_optimizer=n_restarts_optimizer,
				normalize_y=normalize_y,
				standardize_before_fit=standardize_before_fit,
				copy_X_train=copy_X_train,
				random_state=random_state,
			),
		],
		use_cv_predict=use_cv_predict,
	)

