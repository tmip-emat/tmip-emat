
import pandas
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from .cross_val import CrossValMixin
from .base import MultiOutputRegressor
from .select import SelectNAndKBest, feature_concat
from . import ignore_warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from .detrend import DetrendMixin
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ
from .frameable import FrameableMixin

class StackedSingleTargetRegression(
		BaseEstimator,
		RegressorMixin,
		CrossValMixin,
		FrameableMixin,
):

	def __init__(
			self,
			keep_other_features=3,
			step2_cv_folds=5,
			random_state=None,
			n_restarts_optimizer=250,
	):
		"""

		Parameters
		----------
		keep_other_features : int
			The number of other (derived) feature columns to keep. Keeping this
			number small help prevent overfitting problems if the number of
			output features is large.
		step2_cv_folds : int
			The step 1 cross validation predictions are used in step two.  How many
			CV folds?
		"""

		self.keep_other_features = keep_other_features
		self.step2_cv_folds = step2_cv_folds
		self.random_state = random_state
		self.n_restarts_optimizer = n_restarts_optimizer
		self._kernel_generator = lambda dims: C() * RBF([1.0] * dims)


	def fit(self, X, Y):
		"""
		Fit linear and gaussian model.

		Parameters
		----------
		X : numpy array or sparse matrix of shape [n_samples, n_features]
			Training data
		Y : numpy array of shape [n_samples, n_targets]
			Target values.

		Returns
		-------
		self : returns an instance of self.
		"""

		with ignore_warnings(DataConversionWarning):

			self._pre_fit(X,Y)

			self.step1 = MultiOutputRegressor(GaussianProcessRegressor(
				kernel=self._kernel_generator(X.shape[1]),
				random_state=self.random_state,
				n_restarts_optimizer=self.n_restarts_optimizer,
			))
			Y_cv = cross_val_predict(self.step1, X, Y, cv=self.step2_cv_folds)
			self.step1.fit(X, Y)

			step2_dims = X.shape[1] + min(self.keep_other_features, Y.shape[1])

			self.step2 = MultiOutputRegressor(
				make_pipeline(
					SelectNAndKBest(n=X.shape[1], k=self.keep_other_features),
					GaussianProcessRegressor(
						kernel=self._kernel_generator(step2_dims),
						random_state = self.random_state+1 if self.random_state else None,
						n_restarts_optimizer=self.n_restarts_optimizer,
					),
				)
			)

			self.step2.fit(feature_concat(X, Y_cv), Y)

			# if isinstance(Y, pandas.DataFrame):
			# 	self.Y_columns = Y.columns
			# elif isinstance(Y, pandas.Series):
			# 	self.Y_columns = Y.name
			# else:
			# 	self.Y_columns = None

		return self

	def predict(self, X, return_std=False, return_cov=False):
		"""Predict using the model

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = (n_samples, n_features)
			Samples.
		return_std, return_cov : bool
			Not implemented.

		Returns
		-------
		C : array, shape = (n_samples,)
			Returns predicted values.
		"""

		# if isinstance(X, pandas.DataFrame):
		# 	idx = X.index
		# else:
		# 	idx = None

		Yhat1 = self.step1.predict(X)
		Yhat2 = self.step2.predict(feature_concat(X, Yhat1))

		# cols = None
		# if self.Y_columns is not None:
		# 	if len(self.Y_columns) == Yhat2.shape[1]:
		# 		cols = self.Y_columns
		#
		# if idx is not None or cols is not None:
		# 	Yhat2 = pandas.DataFrame(
		# 		Yhat2,
		# 		index=idx,
		# 		columns=cols,
		# 	)

		Yhat2 = self._post_predict(X, Yhat2)

		return Yhat2

	def scores(self, X, Y, sample_weight=None):
		"""
		Returns the coefficients of determination R^2 of the prediction.

		The coefficient R^2 is defined as (1 - u/v), where u is the residual
		sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
		sum of squares ((y_true - y_true.mean()) ** 2).sum().
		The best possible score is 1.0 and it can be negative (because the
		model can be arbitrarily worse). A constant model that always
		predicts the expected value of y, disregarding the input features,
		would get a R^2 score of 0.0.

		Notes
		-----
		R^2 is calculated by weighting all the targets equally using
		`multioutput='raw_values'`.  See documentation for
		sklearn.metrics.r2_score for more information.

		Parameters
		----------
		X : array-like, shape = (n_samples, n_features)
			Test samples. For some estimators this may be a
			precomputed kernel matrix instead, shape = (n_samples,
			n_samples_fitted], where n_samples_fitted is the number of
			samples used in the fitting for the estimator.

		Y : array-like, shape = (n_samples, n_outputs)
			True values for X.

		sample_weight : array-like, shape = [n_samples], optional
			Sample weights.

		Returns
		-------
		score : ndarray
			R^2 of self.predict(X) wrt. Y.
		"""
		return r2_score(Y, self.predict(X), sample_weight=sample_weight,
						multioutput='raw_values')

class DetrendedStackedSingleTargetRegression(
	StackedSingleTargetRegression,
	DetrendMixin
):

	def fit(self, X, Y):
		return super().fit(X, self.detrend_fit(X,Y))

	def predict(self, X, return_std=False, return_cov=False):
		return self.detrend_predict(X) + super().predict(X)




class MultipleTargetRegression(
		BaseEstimator,
		RegressorMixin,
		CrossValMixin,
):

	def __init__(
			self,
	):
		self._kernel_generator = lambda dims: C() * RBF([1.0] * dims)

	def fit(self, X, Y):
		"""
		Fit linear and gaussian model.

		Parameters
		----------
		X : numpy array or sparse matrix of shape [n_samples, n_features]
			Training data
		T : numpy array of shape [n_samples, n_targets]
			Target values.

		Returns
		-------
		self : returns an instance of self.
		"""

		with ignore_warnings(DataConversionWarning):

			self.step1 = MultiOutputRegressor(GaussianProcessRegressor(
				kernel=self._kernel_generator(X.shape[1])
			))
			self.step1.fit(X, Y)

			if isinstance(Y, pandas.DataFrame):
				self.Y_columns = Y.columns
			elif isinstance(Y, pandas.Series):
				self.Y_columns = Y.name
			else:
				self.Y_columns = None

		return self

	def predict(self, X, return_std=False, return_cov=False):
		"""Predict using the model

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = (n_samples, n_features)
			Samples.
		return_std, return_cov : bool
			Not implemented.

		Returns
		-------
		C : array, shape = (n_samples,)
			Returns predicted values.
		"""

		Yhat1 = self.step1.predict(X)
		return Yhat1

	def scores(self, X, Y, sample_weight=None):
		"""
		Returns the coefficients of determination R^2 of the prediction.

		The coefficient R^2 is defined as (1 - u/v), where u is the residual
		sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
		sum of squares ((y_true - y_true.mean()) ** 2).sum().
		The best possible score is 1.0 and it can be negative (because the
		model can be arbitrarily worse). A constant model that always
		predicts the expected value of y, disregarding the input features,
		would get a R^2 score of 0.0.

		Notes
		-----
		R^2 is calculated by weighting all the targets equally using
		`multioutput='raw_values'`.  See documentation for
		sklearn.metrics.r2_score for more information.

		Parameters
		----------
		X : array-like, shape = (n_samples, n_features)
			Test samples. For some estimators this may be a
			precomputed kernel matrix instead, shape = (n_samples,
			n_samples_fitted], where n_samples_fitted is the number of
			samples used in the fitting for the estimator.

		Y : array-like, shape = (n_samples, n_outputs)
			True values for X.

		sample_weight : array-like, shape = [n_samples], optional
			Sample weights.

		Returns
		-------
		score : ndarray
			R^2 of self.predict(X) wrt. Y.
		"""
		return r2_score(Y, self.predict(X), sample_weight=sample_weight,
						multioutput='raw_values')


class DetrendedMultipleTargetRegression(
	MultipleTargetRegression,
	DetrendMixin
):

	def fit(self, X, Y):
		return super().fit(X, self.detrend_fit(X,Y))

	def predict(self, X, return_std=False, return_cov=False):
		return self.detrend_predict(X) + super().predict(X)

