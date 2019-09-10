

import pandas
import numpy
import warnings
import scipy.stats
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.base import clone
from sklearn.metrics import r2_score

from .frameable import FrameableMixin
from .model_selection import CrossValMixin

class LinearRegression(_LinearRegression, FrameableMixin, CrossValMixin):
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
			stats_on_fit=True,
	):
		super().__init__(
			fit_intercept=fit_intercept,
			normalize=normalize,
			copy_X=copy_X,
			n_jobs=n_jobs
		)
		self.stats_on_fit = stats_on_fit

	def fit(self, X, y, sample_weight=None):
		self._pre_fit(X, y)
		super().fit(X, y, sample_weight=sample_weight)

		if self.stats_on_fit:

			if isinstance(X, pandas.DataFrame):
				self._X_columns = list(X.columns)
			elif isinstance(X, pandas.Series):
				self._X_columns = [str(X.name),]
			else:
				self._X_columns = None

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

			r2 = r2_score(y, self.predict(X), sample_weight=sample_weight, multioutput='raw_values')

			if isinstance(self._Y_columns, str):
				self.r2 = pandas.Series(r2, index=[self._Y_columns,])
			elif isinstance(self._Y_columns, list):
				self.r2 = pandas.Series(r2, index=self._Y_columns)
			else:
				self.r2 = r2

		return self

	def predict(self, X):
		y_hat = super().predict(X)
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
			index=self._Y_columns,
			columns=self._X_columns,
		)
		if self.fit_intercept:
			beta['_Intercept_'] = self.intercept_
		beta = beta.stack()
		beta.name = 'Coefficient'

		if self.stats_on_fit:
			se = pandas.DataFrame(
				data=self.stderr_,
				index=self._Y_columns,
				columns=self._X_columns,
			)
			if self.fit_intercept:
				se['_Intercept_'] = self.stderr_intercept_
			se = se.stack()
			se.name = 'StdError'

			t = pandas.DataFrame(
				data=self.t_,
				index=self._Y_columns,
				columns=self._X_columns,
			)
			if self.fit_intercept:
				t['_Intercept_'] = self.t_intercept_
			t = t.stack()
			t.name = 't-Statistic'

			p = pandas.DataFrame(
				data=self.p_,
				index=self._Y_columns,
				columns=self._X_columns,
			)
			if self.fit_intercept:
				p['_Intercept_'] = self.p_intercept_
			p = p.stack()
			p.name = 'p'

			return pandas.concat([beta, se, t, p], axis=1)

		# no stats on fit, just return beta
		return beta



def LinearRegression_KBestPoly(
		k=None,
		degree=2,
		fit_intercept=True,
		normalize=False,
		copy_X=True,
		n_jobs=None,
		stats_on_fit=True,
		single_target=False,
):

	from sklearn.pipeline import make_pipeline, Pipeline
	from .feature_selection import SelectKBestPolynomialFeatures, SelectUniqueColumns
	from .multioutput import MultiOutputRegressor
	if single_target:
		regressor2 = lambda x: x
	else:
		regressor2 = lambda x: MultiOutputRegressor(x, n_jobs=n_jobs,)
	return regressor2(
		Pipeline([
			('KBestPoly', SelectKBestPolynomialFeatures(k=k, degree=degree)),
			('Uniques',   SelectUniqueColumns()),
			('LR',        LinearRegression(
							fit_intercept=fit_intercept,
							normalize=normalize,
							copy_X=copy_X,
							stats_on_fit=stats_on_fit,
			              )
			 ),
		])
	)


def LinearRegression_KRangeBestPoly(
		k_min=0,
		k_max=8,
		degree=2,
		fit_intercept=True,
		normalize=False,
		copy_X=True,
		n_jobs=None,
		stats_on_fit=True,
		cv=5,
):

	from sklearn.pipeline import make_pipeline, Pipeline
	from .feature_selection import SelectKBestPolynomialFeatures, SelectUniqueColumns
	from .multioutput import MultiOutputRegressor
	from sklearn.model_selection import GridSearchCV

	return MultiOutputRegressor(
		GridSearchCV(
			Pipeline([
				('KBestPoly', SelectKBestPolynomialFeatures(degree=degree)),
				('Uniques',   SelectUniqueColumns()),
				('LR',        LinearRegression(
								fit_intercept=fit_intercept,
								normalize=normalize,
								copy_X=copy_X,
								stats_on_fit=stats_on_fit,
							  )
				 ),
			]),
			cv=cv,
			param_grid={'KBestPoly__k': range(k_min, k_max+1)},
			iid=False,
		),
		n_jobs=n_jobs,
	)

