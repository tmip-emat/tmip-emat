
import pandas
import numpy
from .multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_predict
from .feature_selection import SelectNAndKBest
from sklearn.pipeline import make_pipeline
from .frameable import FrameableMixin
from .model_selection import CrossValMixin
from sklearn.base import BaseEstimator, RegressorMixin

def feature_concat(*args):
	if all(isinstance(a, pandas.DataFrame) for a in args):
		return pandas.concat(args, axis=1)
	if any(isinstance(a, pandas.DataFrame) for a in args):
		ref = 0
		while not isinstance(args[ref], pandas.DataFrame):
			ref += 1
		ix = args[ref].index
		return pandas.concat([pandas.DataFrame(a, index=ix) for a in args], axis=1)
	return numpy.concatenate(args, axis=1)


class StackedSingleTargetRegressor(
		BaseEstimator,
		RegressorMixin,
		CrossValMixin,
		FrameableMixin,
):

	def __init__(
			self,
			estimator1,
			estimator2,
			keep_other_features=5,
			cv=5,
	):
		"""

		Parameters
		----------
		keep_other_features : int, default 5
			The number of other (derived) feature columns to keep. Keeping this
			number small help prevent overfitting problems if the number of
			output features is large.
		cv : int, default 5
			The step 1 cross validation predictions are used in fitting step two.
			This controls the number of folds in this internal cross validation.
		"""

		self.keep_other_features = keep_other_features
		self.cv = cv
		self.estimator1 = estimator1
		self.estimator2 = estimator2


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

		# with ignore_warnings(DataConversionWarning):

		self._pre_fit(X,Y)

		self.estimator1_ = MultiOutputRegressor(self.estimator1)
		Y_cv = cross_val_predict(self.estimator1_, X, Y, cv=self.cv)
		self.estimator1_.fit(X, Y)

		self.estimator2_ = MultiOutputRegressor(
			make_pipeline(
				SelectNAndKBest(n=X.shape[1], k=self.keep_other_features),
				self.estimator2,
			)
		)
		self.estimator2_.fit(feature_concat(X, Y_cv), Y)

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
		Yhat1 = self.estimator1_.predict(X)
		Yhat2 = self.estimator2_.predict(feature_concat(X, Yhat1))
		Yhat2 = self._post_predict(X, Yhat2)
		return Yhat2



def StackedLinearAndGaussian(
		keep_other_features=5,
		cv=5,
):
	from .boosting import LinearAndGaussian
	from .anisotropic import AnisotropicGaussianProcessRegressor

	return StackedSingleTargetRegressor(
		LinearAndGaussian(single_target=True),
		LinearAndGaussian(single_target=True),
		keep_other_features=keep_other_features,
		cv=cv,
	)

