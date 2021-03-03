
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold as _VarianceThreshold

class VarianceThreshold(_VarianceThreshold):
	"""
	Feature selector that removes all low-variance features.

	Parameters
	----------
	threshold : float, optional
		Features with a training-set variance lower than this threshold will
		be removed. The default is to keep all features with nearly non-zero variance,
		i.e. remove the features that have the essentially the same value in all samples.

	Attributes
	----------
	variances_ : array, shape (n_features,)
		Variances of individual features.

	Examples
	--------
	The following dataset has integer features, two of which are the same
	in every sample. These are removed with the default setting for threshold::

		>>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
		>>> selector = VarianceThreshold()
		>>> selector.fit_transform(X)
		array([[2, 0],
			   [1, 4],
			   [1, 1]])
	"""

	def __init__(self, threshold=1e-20):
		super().__init__(threshold=threshold)

	def fit(self, X, y=None):
		"""Learn empirical variances from X.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			Sample vectors from which to compute variances.

		y : any
			Ignored. This parameter exists only for compatibility with
			sklearn.pipeline.Pipeline.

		Returns
		-------
		self
		"""
		super().fit(X,y)

		if isinstance(X, pd.DataFrame):
			self.invariant_values = X.iloc[0].loc[~self.get_support()]
		else:
			self.invariant_values = X[0,~self.get_support()]

		return self

	def transform(self, X):

		mask = self.get_support()

		if isinstance(self.invariant_values, pd.Series):
			try:
				pd.testing.assert_series_equal(
					self.invariant_values,
					X.iloc[0].loc[~mask].astype(self.invariant_values.dtype),
					check_names=False,
					check_series_type=False, # Allow when checking Series vs ExperimentalDesignSeries
				)
			except AssertionError:
				raise ValueError(
					"unexpected change in invariant inputs:\n"
					"invariant_values = \n{}\n\n"
					"X {}: \n{}\n\n"
					"X[Masked]: \n{}\n\n"
					"MASK: \n{}".format(
						self.invariant_values,
						X.iloc[0].dtypes, X.iloc[0],
						X.iloc[0].loc[~mask],
						mask
					)
				)
		else:
			if not np.all(self.invariant_values == (X[0,~mask])):
				raise ValueError("unexpected change in invariant inputs")

		result = super().transform(X)

		if isinstance(X, pd.DataFrame):
			result = pd.DataFrame(result, index=X.index, columns=X.columns[mask])

		return result