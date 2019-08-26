
import pandas
from sklearn.preprocessing import PolynomialFeatures as _PolynomialFeatures

class PolynomialFeatures(_PolynomialFeatures):
	"""
	Generate polynomial and interaction features, preserving DataFrame labels.

	Generate a new feature matrix consisting of all polynomial combinations
	of the features with degree less than or equal to the specified degree.
	For example, if an input sample is two dimensional and of the form
	[a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

	Parameters
	----------
	degree : integer
	    The degree of the polynomial features. Default = 2.

	interaction_only : boolean, default = False
	    If true, only interaction features are produced: features that are
	    products of at most ``degree`` *distinct* input features (so not
	    ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).

	include_bias : boolean
	    If True (default), then include a bias column, the feature in which
	    all polynomial powers are zero (i.e. a column of ones - acts as an
	    intercept term in a linear model).

	order : str in {'C', 'F'}, default 'C'
	    Order of output array in the dense case. 'F' order is faster to
	    compute, but may slow down subsequent estimators.

	Attributes
	----------
	powers_ : array, shape (n_output_features, n_input_features)
	    powers_[i, j] is the exponent of the jth input in the ith output.

	n_input_features_ : int
	    The total number of input features.

	n_output_features_ : int
	    The total number of polynomial output features. The number of output
	    features is computed by iterating over all suitably sized combinations
	    of input features.

	Notes
	-----
	Be aware that the number of features in the output array scales
	polynomially in the number of features of the input array, and
	exponentially in the degree. High degrees can cause overfitting.

	"""


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
