
import pandas
from sklearn.feature_selection import SelectKBest as _SelectKBest

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection.base import SelectorMixin

from .preprocessing import PolynomialFeatures

class SelectKBest(_SelectKBest):
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
			drop_duplicate_cols=True,
	):
		self.score_func = score_func
		self.k = k
		self.degree = degree
		self.interaction_only = interaction_only
		self.exclude_degree_1 = exclude_degree_1
		self.retain_degree_1 = retain_degree_1
		self.drop_duplicate_cols = drop_duplicate_cols

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

		y = _drop_duplicate_columns(y)

		return y
