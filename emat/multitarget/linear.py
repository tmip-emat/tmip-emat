
import pandas
import numpy
import warnings
import scipy.stats
from sklearn.linear_model import LinearRegression as _sklearn_LinearRegression
from .frameable import FrameableMixin
from .select import PolynomialFeatures, SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.base import clone

class LinearRegression(_sklearn_LinearRegression, FrameableMixin):
	"""
	Ordinary least squares Linear Regression.

	This class extends the LinearRegression provided in scikit-learn.

	Parameters
	----------
	fit_intercept : boolean, optional, default True
		whether to calculate the intercept for this model. If set
		to False, no intercept will be used in calculations
		(e.g. data is expected to be already centered).

	normalize : boolean, optional, default False
		This parameter is ignored when ``fit_intercept`` is set to False.
		If True, the regressors X will be normalized before regression by
		subtracting the mean and dividing by the l2-norm.
		If you wish to standardize, please use
		:class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
		an estimator with ``normalize=False``.

	copy_X : boolean, optional, default True
		If True, X will be copied; else, it may be overwritten.

	n_jobs : int or None, optional (default=None)
		The number of jobs to use for the computation. This will only provide
		speedup for n_targets > 1 and sufficient large problems.
		``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
		``-1`` means using all processors. See :term:`Glossary <n_jobs>`
		for more details.

	Attributes
	----------
	coef_ : array, shape (n_features, ) or (n_targets, n_features)
		Estimated coefficients for the linear regression problem.
		If multiple targets are passed during the fit (y 2D), this
		is a 2D array of shape (n_targets, n_features), while if only
		one target is passed, this is a 1D array of length n_features.

	intercept_ : array
		Independent term in the linear model.
	"""

	def __init__(
			self,
			fit_intercept=True,
			normalize=False,
			copy_X=True,
			n_jobs=None,
			frame_out=False,
	):
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.copy_X = copy_X
		from ..util import n_jobs_cap
		self.n_jobs = n_jobs_cap(n_jobs)
		self.frame_out = frame_out

	def fit(self, X, y, sample_weight=None):
		self._pre_fit(X, y)
		super().fit(X, y, sample_weight=sample_weight)

		if isinstance(X, pandas.DataFrame):
			self.x_names_ = X.columns.copy()

		if isinstance(y, pandas.DataFrame):
			self.y_names_ = y.columns.copy()
		elif isinstance(y, pandas.Series):
			self.y_names_ = [y.name,]
		else:
			try:
				y_shape_1 = y.shape[1]
			except IndexError:
				y_shape_1 = 1
			self.y_names_ = [f'y{i + 1}' for i in range(y_shape_1)]

		sse = numpy.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])

		if sse.shape == ():
			sse = sse.reshape(1,)

		self.sse_ = sse

		if self.fit_intercept:
			if not isinstance(X, pandas.DataFrame):
				X1 = pandas.DataFrame(X)
			else:
				X1 = X.copy(deep=True)
			X1['__constant__'] = 1.0
		else:
			X1 = X

		try:
			inv_X_XT = numpy.linalg.inv(numpy.dot(X1.T, X1))
		except numpy.linalg.LinAlgError:
			inv_X_XT = numpy.full_like(numpy.dot(X1.T, X1), fill_value=numpy.nan)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)

			try:
				se = numpy.array([
					numpy.sqrt(numpy.diagonal(sse[i] * inv_X_XT))
					for i in range(sse.shape[0])
				])
			except:
				print("sse.shape",sse.shape)
				print(sse)
				raise

			if self.fit_intercept:
				self.stderr_ = se[:,:-1]
				self.t_ = self.coef_ / se[:,:-1]
				self.stderr_intercept_ = se[:,-1]
				self.t_intercept_ = self.intercept_ / se[:,-1]
			else:
				self.stderr_ = se
				self.t_ = self.coef_ / se
			self.p_ = 2 * (1 - scipy.stats.t.cdf(numpy.abs(self.t_), y.shape[0] - X.shape[1]))
			if self.fit_intercept:
				self.p_intercept_ = 2 * (1 - scipy.stats.t.cdf(numpy.abs(self.t_intercept_), y.shape[0] - X.shape[1]))

		from sklearn.metrics import r2_score
		self.r2 = pandas.Series(
			r2_score(y, self.predict(X), sample_weight=sample_weight,
						multioutput='raw_values'),
			index=self.y_names_
		)

		return self

	def predict(self, X):
		y_hat = super().predict(X)
		if self.frame_out:
			y_hat = self._post_predict(X, y_hat)
		return y_hat

	def coefficients_summary(self):
		"""
		A summary DataFrame of the coefficients.

		Includes coefficient estimates, standard error,
		t-stats and p values.

		Returns:
			pandas.DataFrame
		"""

		beta = pandas.DataFrame(
			data=self.coef_,
			index=self.y_names_,
			columns=self.x_names_,
		)
		if self.fit_intercept:
			beta['_Intercept_'] = self.intercept_
		beta = beta.stack()
		beta.name = 'Coefficient'

		se = pandas.DataFrame(
			data=self.stderr_,
			index=self.y_names_,
			columns=self.x_names_,
		)
		if self.fit_intercept:
			se['_Intercept_'] = self.stderr_intercept_
		se = se.stack()
		se.name = 'StdError'

		t = pandas.DataFrame(
			data=self.t_,
			index=self.y_names_,
			columns=self.x_names_,
		)
		if self.fit_intercept:
			t['_Intercept_'] = self.t_intercept_
		t = t.stack()
		t.name = 't-Statistic'

		p = pandas.DataFrame(
			data=self.p_,
			index=self.y_names_,
			columns=self.x_names_,
		)
		if self.fit_intercept:
			p['_Intercept_'] = self.p_intercept_
		p = p.stack()
		p.name = 'p'

		return pandas.concat([beta, se, t, p], axis=1)



class LinearRegressionInteract(LinearRegression):

	def __init__(
			self,
			fit_intercept=True,
			normalize=False,
			copy_X=True,
			n_jobs=None,
			frame_out=True,
			n_interactions=None,
	):
		super().__init__(
			fit_intercept=fit_intercept,
			normalize=normalize,
			copy_X=copy_X,
			n_jobs=n_jobs,
			frame_out=frame_out,
		)

		self.n_interactions = n_interactions

	def fit(self, X, y, sample_weight=None):

		if len(y.shape) == 1 or y.shape[1] == 1:
			# Single Output Dim
			n_interactions = self.n_interactions
			if n_interactions is None:
				n_interactions = X.shape[1]
			self._poly = PolynomialFeatures(2, include_bias=False).fit(X)
			X1 = self._poly.transform(X).iloc[:, X.shape[1]:]
			self._kbest = SelectKBest(mutual_info_regression, k=n_interactions).fit(X1, y)
			X2 = self._kbest.transform(X1)
			X_ = pandas.concat([X, X2], axis=1)
			self._pre_fit(X, y)
			_sklearn_LinearRegression.fit(self, X_, y, sample_weight=sample_weight)

		else:
			# multiple output dim
			self._dim_models = [clone(self) for j in range(y.shape[1])]
			self._dim_names = y.columns
			for j in range(y.shape[1]):
				try:
					self._dim_models[j].fit(X, y.iloc[:,j], sample_weight=sample_weight)
				except:
					print("j=",j)
					raise

		return self

	def predict(self, X):

		if not hasattr(self, '_dim_models'):
			# Single Output Dim
			n_interactions = self.n_interactions
			if n_interactions is None:
				n_interactions = X.shape[1]
			X1 = self._poly.transform(X).iloc[:, X.shape[1]:]
			X2 = self._kbest.transform(X1)
			X_ = pandas.concat([X, X2], axis=1)
			return super().predict(X_)

		else:
			# multiple output dim
			return pandas.concat([
				pandas.Series(self._dim_models[j].predict(X), index=X.index, name=self._dim_names[j])
				for j in range(len(self._dim_models))
			], axis=1)


