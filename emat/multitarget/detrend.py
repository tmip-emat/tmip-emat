
import pandas
from sklearn.metrics import r2_score

from .linear import LinearRegression

class DetrendMixin:

	def detrend_fit(self, X, Y):
		self.lr = LinearRegression()
		self.lr.fit(X, Y)
		residual = Y - self.lr.predict(X)
		return residual

	def detrend_predict(self, X):
		Yhat1 = self.lr.predict(X)
		return Yhat1

	def detrend_scores(self, X, Y, sample_weight=None):
		"""
		Returns the coefficients of determination R^2 of the prediction.

		The coefficient :math:`R^2` is defined as (1 - u/v), where u is the residual
		sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
		sum of squares ((y_true - y_true.mean()) ** 2).sum().
		The best possible score is 1.0 and it can be negative (because the
		model can be arbitrarily worse). A constant model that always
		predicts the expected value of y, disregarding the input features,
		would get a :math:`R^2` score of 0.0.

		Notes
		-----
		:math:`R^2` is calculated by weighting all the targets equally using
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
		return r2_score(Y, self.detrend_predict(X), sample_weight=sample_weight,
						multioutput='raw_values')


from sklearn.gaussian_process import GaussianProcessRegressor

class DetrendedGaussianProcessRegressor(
	GaussianProcessRegressor,
	DetrendMixin,
):

	"""
	De-trended Gaussian process regression (GPR).

	Parameters
	----------
	standardize_before_fit : bool, optional (default: True)
		Whether to rescale the columns of Y to have unit variance
		before fitting.  This may improve performance.  The standardization
		is inverted on prediction, so that predictions are still properly
		scaled.

	n_restarts_optimizer : int, optional (default: 0)
		The number of restarts of the optimizer for finding the kernel's
		parameters which maximize the log-marginal likelihood. The first run
		of the optimizer is performed from the kernel's initial parameters,
		the remaining ones (if any) from thetas sampled log-uniform randomly
		from the space of allowed theta-values. If greater than 0, all bounds
		must be finite. Note that n_restarts_optimizer == 0 implies that one
		run is performed.

	copy_X_train : bool, optional (default: True)
		If True, a persistent copy of the training data is stored in the
		object. Otherwise, just a reference to the training data is stored,
		which might cause predictions to change if the data is modified
		externally.

	random_state : int, RandomState instance or None, optional (default: None)
		The generator used to initialize the centers. If int, random_state is
		the seed used by the random number generator; If RandomState instance,
		random_state is the random number generator; If None, the random number
		generator is the RandomState instance used by `np.random`.

	Attributes
	----------

	lr : LinearRegression
		The de-trending linear regression model.



	"""

	def __init__(
			self,
			standardize_before_fit=True,
			n_restarts_optimizer=250,
			copy_X_train=True,
			random_state=None,
	):
		self.standardize_before_fit=standardize_before_fit
		super().__init__(
			n_restarts_optimizer=n_restarts_optimizer,
			copy_X_train=copy_X_train,
			random_state=random_state,
		)

	def fit(self, X, Y):
		"""
		Fit a linear and gaussian regression model.

		Parameters
		----------
		X : array-like or sparse matrix of shape [n_samples, n_features]
			Training data
		Y : array-like of shape [n_samples, ]
			Target values.

		Returns
		-------
		self : MultipleTargetRegression
			Returns an instance of self, the the sklearn standard practice.
		"""

		if self.standardize_before_fit:
			self.standardize_Y = Y.std(axis=0, ddof=0)
			Y = Y / self.standardize_Y
		else:
			self.standardize_Y = None

		super().fit(X, self.detrend_fit(X,Y))

		if isinstance(Y, pandas.DataFrame):
			self.Y_columns = Y.columns
		elif isinstance(Y, pandas.Series):
			self.Y_columns = Y.name
		else:
			self.Y_columns = None

		return self

	def residual_predict(self, X, return_std=False, return_cov=False):
		"""
		Predict using the only the Gaussian regression model

		This function will return a pandas DataFrame instead of
		a simple numpy array if there is information available
		to populate the index (if the X argument to this function
		is a DataFrame) or the columns (if the Y argument to `fit`
		was a DataFrame).

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = (n_samples, n_features)
			Samples.
		return_std, return_cov : bool
			Not implemented.

		Returns
		-------
		C : array-like, shape = (n_samples, n_targets)
			Returns predicted values. The n_targets dimension is
			determined in the `fit` method.
		"""
		return super().predict(X, return_std=return_std, return_cov=return_cov)

	def detrend_predict(self, X):
		"""
		Predict using the only the detrending linear regression model

		This function will return a pandas DataFrame instead of
		a simple numpy array if there is information available
		to populate the index (if the X argument to this function
		is a DataFrame) or the columns (if the Y argument to `fit`
		was a DataFrame).

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = (n_samples, n_features)
			Samples.

		Returns
		-------
		C : array-like, shape = (n_samples, n_targets)
			Returns predicted values. The n_targets dimension is
			determined in the `fit` method.
		"""

		Yhat = super().detrend_predict(X)

		if isinstance(X, pandas.DataFrame):
			idx = X.index
		else:
			idx = None

		cols = None
		if self.Y_columns is not None:
			try:
				Yhat_shape_1 = Yhat.shape[1]
			except IndexError:
				Yhat_shape_1 = 1
			if len(self.Y_columns) == Yhat_shape_1:
				cols = self.Y_columns

		if idx is not None or cols is not None:
			return pandas.DataFrame(
				Yhat,
				index=idx,
				columns=cols,
			)
		return Yhat

	def predict(self, X, return_std=False, return_cov=False):
		"""
		Predict using the model

		This function will return a pandas DataFrame instead of
		a simple numpy array if there is information available
		to populate the index (if the X argument to this function
		is a DataFrame) or the columns (if the Y argument to `fit`
		was a DataFrame).

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = (n_samples, n_features)
			Samples.
		return_std, return_cov : bool
			Not implemented.

		Returns
		-------
		C : array-like, shape = (n_samples, n_targets)
			Returns predicted values. The n_targets dimension is
			determined in the `fit` method.
		"""
		if return_std:
			rp, se = self.residual_predict(X, return_std=return_std)
			return super().detrend_predict(X) + rp, se
		else:
			return super().detrend_predict(X) + self.residual_predict(X)
