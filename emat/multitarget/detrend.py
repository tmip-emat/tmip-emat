
from .linear import LinearRegression
from sklearn.metrics import r2_score

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
