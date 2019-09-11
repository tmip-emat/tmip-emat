
import pandas
import numpy
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ
from sklearn.multioutput import RegressorChain as _RegressorChain

from . import ignore_warnings
from .detrend import DetrendMixin
from .cross_val import CrossValMixin
from .base import MultiOutputRegressor
from .select import SelectNAndKBest, feature_concat
from .frameable import FrameableMixin

class RegressorChain(
	_RegressorChain,
	FrameableMixin,
):
	def fit(self, X, Y):
		self._pre_fit(X, Y)
		return super().fit(X,Y)

	def predict(self, X):
		Y = super().predict(X)
		Y = self._post_predict(X, Y)
		return Y

class ChainedTargetRegression(
		BaseEstimator,
		RegressorMixin,
		CrossValMixin,
):

	def __init__(self, keep_other_features=3, step2_cv_folds=5, randomize_chain=True):
		"""

		Parameters
		----------
		core_features
			feature columns to definitely keep for both LR and GPR

		"""

		self.keep_other_features = keep_other_features
		self.step1 = GaussianProcessRegressor()
		self.step2_cv_folds = step2_cv_folds
		self.randomize_chain = randomize_chain
		self._kernel_generator = lambda dims: C() * RBF([1.0] * dims)

	def fit(self, X, Y):
		"""
		Fit linear and gaussian model.

		Parameters
		----------
		X : numpy array or sparse matrix of shape [n_samples, n_features]
			Training data
		y : numpy array of shape [n_samples, n_targets]
			Target values.

		Returns
		-------
		self : returns an instance of self.
		"""

		with ignore_warnings(DataConversionWarning):

			if isinstance(Y, pandas.DataFrame):
				self.Y_columns = Y.columns
				Y_ = Y.values
			elif isinstance(Y, pandas.Series):
				self.Y_columns = [Y.name]
				Y_ = Y.values.reshape(-1,1)
			else:
				self.Y_columns = ["Untitled" * Y.shape[1]]
				Y_ = Y

			Yhat = pandas.DataFrame(
				data=0,
				index=X.index,
				columns=self.Y_columns,
			)

			self.steps = []

			self._chain_order = numpy.arange(Y.shape[1])
			if self.randomize_chain is not None and self.randomize_chain is not False:
				if self.randomize_chain is not True:
					numpy.random.seed(self.randomize_chain)
				numpy.random.shuffle(self._chain_order)

			for meta_n in range(Y.shape[1]):

				n = self._chain_order[meta_n]

				step_dims = X.shape[1] + min(self.keep_other_features, meta_n)

				self.steps.append(

					make_pipeline(
						SelectNAndKBest(n=X.shape[1], k=self.keep_other_features),
						GaussianProcessRegressor(
							kernel=self._kernel_generator(step_dims),
						),
					).fit(
						feature_concat(X, Yhat.iloc[:,:meta_n]),
						Y_[:,n]
					)
				)
				Yhat.iloc[:, meta_n] = cross_val_predict(
					self.steps[-1],
					feature_concat(X, Yhat.iloc[:,:meta_n]),
					Y_[:,n],
					cv=self.step2_cv_folds,
				)

		return self

	def predict(self, X, return_std=False, return_cov=False):
		"""Predict using the model

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = (n_samples, n_features)
			Samples.

		Returns
		-------
		C : array, shape = (n_samples,)
			Returns predicted values.
		"""

		if isinstance(X, (pandas.DataFrame, pandas.Series)):
			x_ix = X.index
		else:
			x_ix = pandas.RangeIndex(X.shape[0])

		Yhat = pandas.DataFrame(
			index=x_ix,
			columns=self.Y_columns[self._chain_order],
		)

		if return_std:
			Ystd = pandas.DataFrame(
				index=x_ix,
				columns=self.Y_columns[self._chain_order],
			)
			for meta_n in range(len(self.Y_columns)):
				y1, y2 = self.steps[meta_n].predict(
					feature_concat(X, Yhat.iloc[:, :meta_n]),
					return_std=True
				)
				Yhat.iloc[:, meta_n] = y1
				Ystd.iloc[:, meta_n] = y2
			return Yhat[self.Y_columns], Ystd[self.Y_columns]

		else:
			for meta_n in range(len(self.Y_columns)):
				y1 = self.steps[meta_n].predict(
					feature_concat(X, Yhat.iloc[:, :meta_n]),
				)
				Yhat.iloc[:, meta_n] = y1
			return Yhat[self.Y_columns]


	def cross_val_scores(self, X, Y, cv=3):
		p = self.cross_val_predicts(X, Y, cv=cv)
		return pandas.Series(
			r2_score(Y, p, sample_weight=None, multioutput='raw_values'),
			index=Y.columns
		)

	def cross_val_predicts(self, X, Y, cv=3, alt_y=None):
		with ignore_warnings(DataConversionWarning):
			p = cross_val_predict(self, X, Y, cv=cv)
		return pandas.DataFrame(p, columns=Y.columns, index=Y.index)


	def score(self, X, y, sample_weight=None):
		"""Returns the coefficient of determination R^2 of the prediction.

		The coefficient R^2 is defined as (1 - u/v), where u is the residual
		sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
		sum of squares ((y_true - y_true.mean()) ** 2).sum().
		The best possible score is 1.0 and it can be negative (because the
		model can be arbitrarily worse). A constant model that always
		predicts the expected value of y, disregarding the input features,
		would get a R^2 score of 0.0.

		Parameters
		----------
		X : array-like, shape = (n_samples, n_features)
			Test samples.

		y : array-like, shape = (n_samples) or (n_samples, n_outputs)
			True values for X.

		sample_weight : array-like, shape = [n_samples], optional
			Sample weights.

		Returns
		-------
		score : float
			R^2 of self.predict(X) wrt. y.
		"""

		return r2_score(y, self.predict(X), sample_weight=sample_weight,
						multioutput='raw_values').mean()

	def scores(self, X, y, sample_weight=None):
		"""Returns the coefficients of determination R^2 of the prediction.

		The coefficient R^2 is defined as (1 - u/v), where u is the residual
		sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
		sum of squares ((y_true - y_true.mean()) ** 2).sum().
		The best possible score is 1.0 and it can be negative (because the
		model can be arbitrarily worse). A constant model that always
		predicts the expected value of y, disregarding the input features,
		would get a R^2 score of 0.0.

		Parameters
		----------
		X : array-like, shape = (n_samples, n_features)
			Test samples.

		y : array-like, shape = (n_samples) or (n_samples, n_outputs)
			True values for X.

		sample_weight : array-like, shape = [n_samples], optional
			Sample weights.

		Returns
		-------
		score : float
			R^2 of self.predict(X) wrt. y.
		"""

		return r2_score(y, self.predict(X), sample_weight=sample_weight,
						multioutput='raw_values')


class DetrendedChainedTargetRegression(
	ChainedTargetRegression,
	DetrendMixin
):

	def fit(self, X, Y):
		return super().fit(X, self.detrend_fit(X,Y))

	def predict(self, X, return_std=False, return_cov=False):
		return self.detrend_predict(X) + super().predict(X)
