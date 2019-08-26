import numpy, pandas
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest as _SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures as _PolynomialFeatures


class SelectKBest(_SelectKBest):
	def transform(self, X):
		y = super().transform(X)
		if isinstance(X, pandas.DataFrame):
			return pandas.DataFrame(
				data=y,
				index=X.index,
				columns=X.columns[self.get_support()]
			)
		return y


class PolynomialFeatures(_PolynomialFeatures):
	def transform(self, X):
		try:
			y = super().transform(X)
		except:
			print(X.shape)
			print(self.n_input_features_)
			raise
		if isinstance(X, pandas.DataFrame):
			return pandas.DataFrame(
				data=y,
				index=X.index,
				columns=self.get_feature_names(X.columns),
			)
		return y



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


class SelectNAndKBest(
	TransformerMixin,
):
	"""
	Selects the first N features plus the K best other features.
	"""

	def __init__(self, n, k, func=None):
		self._n = n
		self._k = k
		self._func = mutual_info_regression if func is None else func

	def fit(self, X, y):
		if self._k > X.shape[1]-self._n:
			use_k = 'all'
		else:
			use_k = self._k

		if X.shape[1]-self._n <=0:
			self._feature_selector = None
		elif isinstance(X, pandas.DataFrame):
			self._feature_selector = SelectKBest(self._func, k=use_k).fit(X.iloc[:,self._n:], y)
		else:
			self._feature_selector = SelectKBest(self._func, k=use_k).fit(X[:,self._n:], y)
		return self

	def transform(self, X):
		if isinstance(X, pandas.DataFrame):
			X_outside = X.iloc[:,self._n:]
			if self._feature_selector is None:
				X2 = X_outside
			else:
				X2 = self._feature_selector.transform(X_outside)
				X2 = pandas.DataFrame(X2, index=X.index, columns=X_outside.columns[self._feature_selector.get_support()])
			return pandas.concat([X.iloc[:,:self._n], X2], axis=1)
		else:
			if self._feature_selector is None:
				X2 = X[:,self._n:]
			else:
				X2 = self._feature_selector.transform(X[:,self._n:])
			return numpy.concatenate([X[:,:self._n], X2], axis=1)

