
import pandas
from sklearn.feature_selection import SelectKBest as _SelectKBest

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

