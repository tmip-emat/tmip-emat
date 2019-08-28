
import pandas
from typing import Sequence
from sklearn.base import RegressorMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict
from .frameable import FrameableMixin
from .model_selection import CrossValMixin
from .multioutput import MultiOutputRegressor

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
			prediction_tier=9999,
	):
		self.estimators = estimators
		self.use_cv_predict = use_cv_predict
		self.prediction_tier = prediction_tier

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

	def _set_prediction_tier(self, tier):
		tier_ = tier
		if tier is not None:
			import numbers
			if not isinstance(tier, numbers.Integral):
				raise ValueError('tier must be integer')
			if tier == 0:
				tier = 9999
			if tier < 0:
				tier = len(self.estimators) + tier
			if tier <= 0:
				raise IndexError(f'invalid tier {tier_}')
			self.prediction_tier = tier

	def predict(self, X, tier=None):
		"""
		Generate predictions from a set of exogenous data.

		Parameters
		----------
		X : array-like, prefer pandas.DataFrame
			Exogenous data.
		tier : int, optional
			Limit the prediction to using only the first `tier`
			levels of stacking. For example, setting to 1 results
			in only using the very first level of the stack.  If not
			given, the existing value of `prediction_tier` is used.

		"""
		if tier is None:
			tier = self.prediction_tier
		Yhat = self.estimators_[0].predict(X)
		for n_, e_ in enumerate(self.estimators_[1:]):
			if n_+1 < tier:
				Yhat += e_.predict(X)
		Yhat = self._post_predict(X, Yhat)
		return Yhat

	def cross_val_scores(self, X, Y, cv=5, S=None, random_state=None, repeat=None, tier=None):
		"""
		Calculate the cross validation scores for this model.

		Unlike other scikit-learn scores, this method returns
		a separate score value for each output when the estimator
		is for a multi-output process.

		If the estimator includes a `sample_stratification`
		attribute, it is used along with

		Args:
			X, Y : array-like
				The independent and dependent data to use for
				cross-validation.
			cv : int, default 5
				The number of folds to use in cross-validation.
			S : array-like
				The stratification data to use for stratified
				cross-validation.  This data must be categorical
				(or convertible into such), and should be a
				vector of length equal to the first dimension
				(i.e. number of observations) in the `X` and `Y`
				arrays.
			repeat : int, optional
				Repeat the cross validation exercise this many
				times, with different random seeds, and return
				the average result.

		Returns:
			pandas.Series: The cross-validation scores, by output.

		"""
		self._set_prediction_tier(tier)

		if repeat is not None:
			ps = []
			for r in range(repeat):
				p_ = self._cross_validate(X, Y, cv=cv, S=S, random_state=r, cache_metadata=self.prediction_tier)
				ps.append(pandas.Series({j: p_[f"test_{j}"].mean() for j in self.Y_columns}))
			return pandas.concat(ps, axis=1).mean(axis=1)

		p = self._cross_validate(X, Y, cv=cv, S=S, random_state=random_state, cache_metadata=self.prediction_tier)
		try:
			return pandas.Series({j: p[f"test_{j}"].mean() for j in self.Y_columns})
		except:
			print("p=", p)
			print(len(self.Y_columns))
			print("self.Y_columns=", self.Y_columns)
			raise


def LinearAndGaussian(
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
	from .linear_model import LinearRegression
	from .anisotropic import AnisotropicGaussianProcessRegressor
	return StackedRegressor(
		[
			LinearRegression(
				fit_intercept=fit_intercept,
				copy_X=False,
				n_jobs=n_jobs,
				stats_on_fit=stats_on_fit,
			),
			MultiOutputRegressor(AnisotropicGaussianProcessRegressor(
				kernel_generator=kernel_generator,
				alpha=alpha,
				optimizer=optimizer,
				n_restarts_optimizer=n_restarts_optimizer,
				normalize_y=normalize_y,
				standardize_before_fit=standardize_before_fit,
				copy_X_train=copy_X_train,
				random_state=random_state,
			)),
		],
		use_cv_predict=use_cv_predict,
	)


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
			MultiOutputRegressor(AnisotropicGaussianProcessRegressor(
				kernel_generator=kernel_generator,
				alpha=alpha,
				optimizer=optimizer,
				n_restarts_optimizer=n_restarts_optimizer,
				normalize_y=normalize_y,
				standardize_before_fit=standardize_before_fit,
				copy_X_train=copy_X_train,
				random_state=random_state,
			)),
		],
		use_cv_predict=use_cv_predict,
	)

