
import pandas, numpy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from .frameable import FrameableMixin

class AnisotropicGaussianProcessRegressor(
	GaussianProcessRegressor,
	FrameableMixin,
):

	def __init__(
			self,
			kernel_generator=None,
			alpha=1e-10,
			optimizer="fmin_l_bfgs_b",
			n_restarts_optimizer=250,
			normalize_y=False,
			standardize_before_fit=True,
			copy_X_train=True,
			random_state=None,
	):

		self.kernel_generator = kernel_generator
		self.standardize_before_fit = standardize_before_fit

		super().__init__(
			kernel=None,
			alpha=alpha,
			optimizer=optimizer,
			n_restarts_optimizer=n_restarts_optimizer,
			normalize_y=normalize_y,
			copy_X_train=copy_X_train,
			random_state=random_state,
		)

	def fit(self, X, y):
		"""
		Fit Gaussian process regression model.

		Parameters
		----------
		X : array-like, shape = (n_samples, n_features)
			Training data

		y : array-like, shape = (n_samples, [n_output_dims])
			Target values

		Returns
		-------
		self : returns an instance of self.
		"""

		if self.kernel_generator is None:
			if self.standardize_before_fit:
				kernel_generator = lambda dims: RBF([1.0] * dims)
			else:
				kernel_generator = lambda dims: C() * RBF([1.0] * dims)
		else:
			kernel_generator = self.kernel_generator
		self.kernel = kernel_generator(X.shape[1])

		self._pre_fit(X, y)

		if self.standardize_before_fit:
			y = numpy.copy(y)
			self.standardize_Y = y.std(axis=0, ddof=0)
			if isinstance(self.standardize_Y, numpy.float):
				if self.standardize_Y == 0:
					self.standardize_Y = 1
			else:
				self.standardize_Y[self.standardize_Y==0] = 1
			y /= self.standardize_Y
		else:
			self.standardize_Y = None

		return super().fit(X, y)

	def predict(self, X, return_std=False, return_cov=False):
		"""
		Predict using the Gaussian process regression model

		We can also predict based on an unfitted model by using the GP prior.
		In addition to the mean of the predictive distribution, also its
		standard deviation (return_std=True) or covariance (return_cov=True).
		Note that at most one of the two can be requested.

		Parameters
		----------
		X : array-like, shape = (n_samples, n_features)
			Query points where the GP is evaluated

		return_std : bool, default: False
			If True, the standard-deviation of the predictive distribution at
			the query points is returned along with the mean.

		return_cov : bool, default: False
			If True, the covariance of the joint predictive distribution at
			the query points is returned along with the mean

		Returns
		-------
		y_mean : array, shape = (n_samples, [n_output_dims])
			Mean of predictive distribution a query points

		y_std : array, shape = (n_samples,), optional
			Standard deviation of predictive distribution at query points.
			Only returned when return_std is True.

		y_cov : array, shape = (n_samples, n_samples), optional
			Covariance of joint predictive distribution a query points.
			Only returned when return_cov is True.
		"""

		if return_cov:
			y_hat, y_cov = super().predict(X, return_std=return_std, return_cov=return_cov)
			y_std = None
		elif return_std:
			y_hat, y_std = super().predict(X, return_std=return_std, return_cov=return_cov)
			y_cov = None
		else:
			y_hat = super().predict(X, return_std=return_std, return_cov=return_cov)
			y_std = None
			y_cov = None

		if self.standardize_Y is not None:
			try:
				y_hat *= self.standardize_Y[None, :]
			except (IndexError, TypeError):
				y_hat *= self.standardize_Y
			if y_std is not None:
				try:
					y_std *= self.standardize_Y[None, :]
				except IndexError:
					y_std *= self.standardize_Y
			if y_cov is not None:
				raise NotImplementedError()

		y_hat = self._post_predict(X, y_hat)
		if y_std is not None:
			y_std = self._post_predict(X, y_hat)

		if y_std is not None:
			return y_hat, y_std
		if y_cov is not None:
			return y_hat, y_cov

		return y_hat