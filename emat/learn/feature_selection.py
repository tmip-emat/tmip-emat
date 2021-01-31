
import numpy
import pandas
from sklearn.feature_selection import SelectKBest as _SelectKBest
from sklearn.utils import check_array, safe_mask

from sklearn.base import BaseEstimator, clone, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression

from .preprocessing import PolynomialFeatures
from .frameable import FrameableMixin

def drop_deficient_columns(df, remaining=None):
	"""
	Drop columns from a dataframe until it is full column rank.

	Parameters
	----------
	df : array-like, prefer pandas.DataFrame
	remaining : array-like, optional
		A mask that shows which columns have previously been not dropped.
	"""
	if remaining is not None:
		if isinstance(df, pandas.DataFrame):
			df_ = df.loc[:,remaining]
		else:
			df_ = df[:,remaining]
	else:
		remaining = numpy.ones(df.shape[1], bool)
		df_ = df
	w, v = numpy.linalg.eigh(numpy.dot(df_.T, df_))
	if w[0] < 1e-4:
		toss = numpy.where(numpy.round(v[:,0],7))[0][-1]
		toss = numpy.where(remaining)[0][toss]
		remaining[toss] = False
		slim, remaining = drop_deficient_columns(df, remaining)
		return slim, remaining
	else:
		return df_, remaining

class SelectorMixinNoZeroWarn(SelectorMixin):

	def transform(self, X):
		"""Reduce X to the selected features.

		Parameters
		----------
		X : array of shape [n_samples, n_features]
			The input samples.

		Returns
		-------
		X_r : array of shape [n_samples, n_selected_features]
			The input samples with only the selected features.
		"""
		X = check_array(X, dtype=None, accept_sparse='csr')
		mask = self.get_support()
		if not mask.any():
			return numpy.empty(0).reshape((X.shape[0], 0))
		if len(mask) != X.shape[1]:
			raise ValueError("X has a different shape than during fitting.")
		return X[:, safe_mask(X, mask)]


class SelectUniqueColumns(BaseEstimator, SelectorMixin):

	def __init__(self):
		super().__init__()

	def fit(self, X, y=None):
		"""
		Run filter on X to get unique columns.

		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
			The training input samples.

		y : array-like, shape = [n_samples]
			Not used.

		Returns
		-------
		self
		"""
		_, self.mask_ = drop_deficient_columns(X)
		return self

	def _get_support_mask(self):
		return self.mask_

	def transform(self, X):
		try:
			y = super().transform(X)
		except:
			print("shape.X",X.shape)
			print("self.mask_.shape",self.mask_.shape)
			raise
		if isinstance(X, pandas.DataFrame):
			return pandas.DataFrame(
				data=y,
				index=X.index,
				columns=X.columns[self.get_support()]
			)
		return y


class SelectKBest(_SelectKBest, SelectorMixinNoZeroWarn):
	"""
	Select features according to the k highest scores, preserving DataFrame labels.

	Parameters
	----------
	score_func : callable
		Function taking two arrays X and y, and returning a pair of arrays
		(scores, pvalues) or a single array with scores.
		Default is f_classif (see below "See also"). The default function only
		works with classification tasks.

	k : int or "all", optional, default=10
		Number of top features to select.
		The "all" option bypasses selection, for use in a parameter search.

	Attributes
	----------
	scores_ : array-like, shape=(n_features,)
		Scores of features.

	pvalues_ : array-like, shape=(n_features,)
		p-values of feature scores, None if `score_func` returned only scores.

	"""

	def transform(self, X):
		y = super().transform(X)
		if isinstance(X, pandas.DataFrame):
			return pandas.DataFrame(
				data=y,
				index=X.index,
				columns=X.columns[self.get_support()]
			)
		return y


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



def _get_duplicate_columns(df):
	'''
	Get a list of duplicate columns.

	Parameters
	----------
	df : Dataframe

	Returns
	-------
	list
		Columns whose contents are duplicates.
	'''
	dupes = set()
	for x in range(df.shape[1]):
		col = df.iloc[:, x]
		for y in range(x + 1, df.shape[1]):
			if col.equals(df.iloc[:, y]):
				dupes.add(df.columns.values[y])
	return list(dupes)

def _drop_duplicate_columns(df):
	if isinstance(df, pandas.DataFrame):
		return df.drop(columns=_get_duplicate_columns(df))
	else:
		df = pandas.DataFrame(data=df)
		df = df.drop(columns=_get_duplicate_columns(df))
		return df.values


class SelectKBestPolynomialFeatures(BaseEstimator):
	"""
	Select best polynomial features according to the k highest scores.

	Parameters
	----------
	score_func : callable
		Function taking two arrays X and y, and returning a pair of arrays
		(scores, pvalues) or a single array with scores.
		Default is f_classif (see below "See also"). The default function only
		works with classification tasks.

	k : int, optional
		Number of top features to select.  If not given, the number of selected
		polynomial features is set equal to the number of original features.

	degree : integer
		The maximum degree of the polynomial features. Default = 2.

	interaction_only : boolean, default = False
		If true, only interaction features are produced: features that are
		products of at most ``degree`` *distinct* input features (so not
		``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).

	exclude_degree_1 : boolean, default = True
		If true, degree 1 (simple linear) features are excluded from the analysis
		of best fitting columns.

	retain_degree_1 : boolean, default = True
		If true, degree 1 (simple linear) features are included in the transform
		result, regardless of quality of fit.  Only considered if `exclude_degree_1`
		is set to True.

	drop_duplicate_cols : boolean, default = True
		If true, extra columns of duplicate data are removed from the result.  This
		can be helpful if, for example, one of the k best polynomial fits is a power
		of a binary variable (in which case, the power is identical to the original).
	"""

	def __init__(
			self,
			score_func=None,
			k=None,
			degree=2,
			interaction_only=False,
			exclude_degree_1=True,
			retain_degree_1=True,
	):
		self.score_func = score_func
		self.k = k
		self.degree = degree
		self.interaction_only = interaction_only
		self.exclude_degree_1 = exclude_degree_1
		self.retain_degree_1 = retain_degree_1

	def fit(self, X, y, sample_weight=None):

		if len(y.shape) == 1 or y.shape[1] == 1:
			# Single Output Dimension, One Model
			n_interactions = self.k
			if n_interactions is None:
				n_interactions = X.shape[1]

			self._poly = PolynomialFeatures(
				self.degree,
				include_bias=False,
				interaction_only=self.interaction_only,
			).fit(X)
			X1 = self._poly.transform(X)
			if self.exclude_degree_1 and not self.interaction_only:
				if isinstance(X1, pandas.DataFrame):
					X1 = X1.iloc[:, X.shape[1]:]
				else:
					X1 = X1[:, X.shape[1]:]

			score_func = self.score_func
			if score_func is None:
				from sklearn.feature_selection import mutual_info_regression
				# fixed random state on mutual_info_regression for stability
				score_func = lambda *arg, **kwarg: mutual_info_regression(*arg, random_state=42, **kwarg)

			self._kbest = SelectKBest(score_func, k=n_interactions)
			self._kbest.fit(X1, y)

		else:

			raise ValueError("SelectKBestPolynomialFeatures only works with single target data")
			# # Multiple Output Dimensions, clone and iterate
			# self._dim_models = [clone(self) for j in range(y.shape[1])]
			# if isinstance(y, pandas.DataFrame):
			# 	self._dim_names = y.columns
			# 	for j in range(y.shape[1]):
			# 		self._dim_models[j].fit(X, y.iloc[:, j], sample_weight=sample_weight)
			# else:
			# 	self._dim_names = None
			# 	for j in range(y.shape[1]):
			# 		self._dim_models[j].fit(X, y[:, j], sample_weight=sample_weight)

		return self

	def transform(self, X):

		X1 = self._poly.transform(X)
		if self.exclude_degree_1 and not self.interaction_only:
			if isinstance(X1, pandas.DataFrame):
				X1 = X1.iloc[:, X.shape[1]:]
			else:
				X1 = X1[:, X.shape[1]:]

		y = self._kbest.transform(X1)

		if self.exclude_degree_1 and not self.interaction_only and self.retain_degree_1:
			if isinstance(y, pandas.DataFrame) and isinstance(y, pandas.DataFrame):
				y = pandas.concat([X, y], axis=1)
			else:
				import numpy
				y = numpy.hstack([numpy.asarray(X), numpy.asarray(y)])

		if isinstance(X, pandas.DataFrame):
			cols = list(X1.columns[self._kbest.get_support()])
			if self.exclude_degree_1 and not self.interaction_only and self.retain_degree_1:
				cols = list(X.columns) + cols

			y = pandas.DataFrame(
				data=y,
				index=X.index,
				columns=cols
			)

		return y
