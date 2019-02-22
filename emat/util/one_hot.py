
import pandas
import numpy

def _to_categorical(y, num_classes=None, dtype='float32', compress=False):
	"""Converts a class array (integers) to binary class array.

	E.g. for use with categorical_crossentropy.

	Args:
		y (array-like): class array to be converted
			(integers from 0 to num_classes).
		num_classes (int, optional): total number of classes.  If not given, the
			number of classes is the maximum value found in `y` plus 1, or the
			actual number of unique values if `compress` is True.
		dtype (str, optional): The data type to be returned,
			(e.g. `float32`, `float64`, `int32`...)
		compress (bool, default False): Whether to compress the values in the input
			array, so that only classes with at least one observation appear
			in the result.  This is useful if the input array is not encoded with
			a sequential set of integer values.


	Returns:
		ndarray: A binary matrix representation of the input. The classes axis
			is placed last, such that the shape of this array is the same as that
			of `y` plus one dimension.

	"""
	y = numpy.array(y, dtype='int')
	if compress:
		y = numpy.unique(y, return_inverse=True)[1]
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = numpy.max(y) + 1
	n = y.shape[0]
	categorical = numpy.zeros((n, num_classes), dtype=dtype)
	categorical[numpy.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = numpy.reshape(categorical, output_shape)
	return categorical


def to_categorical(arr, index=None, dtype='float32'):

	if isinstance(arr, (pandas.DataFrame, pandas.Series)) and index is None:
		index = arr.index

	codes, positions = numpy.unique(arr, return_inverse=True)
	return pandas.DataFrame(
		data=_to_categorical(positions, dtype=dtype),
		columns=codes,
		index=index,
	)



def categorical_expansion(s, column=None, inplace=False, drop=False):
	"""
	Expand a pandas Series into a DataFrame containing a categorical dummy variables.

	This is sometimes called "one-hot" encoding.

	Parameters
	----------
	s : pandas.Series or pandas.DataFrame
		The input data
	column : str, optional
		If `s` is given as a DataFrame, expand this column
	inplace : bool, default False
		If true, add the result directly as new columns in `s`.
	drop : bool, default False
		If true, drop the existing column from `s`. Has no effect if
		`inplace` is not true.

	Returns
	-------
	pandas.DataFrame
		Only if inplace is false.
	"""
	if isinstance(s, pandas.DataFrame):
		input = s
		if column is not None and column not in s.columns:
			raise KeyError(f'key not found "{column}"')
		if len(s.columns) == 1:
			s = s.iloc[:, 0]
			column = s.columns[0]
		elif column is not None:
			s = s.loc[:, column]
	else:
		input = None

	onehot = to_categorical(s)
	onehot.columns = [f'{s.name}=={_}' for _ in onehot.columns]
	if inplace and input is not None:
		input[onehot.columns] = onehot
		if drop:
			input.drop([column], axis=1, inplace=True)
	else:
		return onehot


def categorical_expansion_all(df, inplace=False, drop=True):
	"""One-hot encode all categorical columns of a DataFrame.

	Args:
		df (pandas.DataFrame): The input dataframe
		inplace (bool, default False): Whether to make the expansion in-place.
		drop (bool, default False): If true, drop the existing categorical
			columns from `df`.

	Returns:
		pandas.DataFrame: Only if inplace is false.
	"""

	if inplace:
		d = df
	else:
		d = df.copy()

	d_dtypes = d.dtypes

	for col in d_dtypes[d_dtypes == 'category'].index:
		categorical_expansion(d, column=col, inplace=True, drop=drop)

	if not inplace:
		return d



from sklearn.preprocessing import OneHotEncoder

class OneHotCatEncoder(OneHotEncoder):

	def fit(self, X, y=None):
		if not isinstance(X, pandas.DataFrame):
			raise TypeError('X must be a DataFrame')

		Xc = X.select_dtypes('category')

		self.categorical_features_ = Xc.columns

		if len(self.categorical_features_):
			super().fit(Xc)

		return self

	def transform(self, X):

		if len(self.categorical_features_):

			Xc = X[self.categorical_features_]

			data = super().transform(Xc).todense()
			columns = [
				f'{k}=={col}'
				for j,k in zip(self.categories_,self.categorical_features_)
				for col in j
			]

			one_hotted = pandas.DataFrame(
				data=data,
				columns=columns,
				index=X.index,
			)

			return pandas.concat([
				X.drop(self.categorical_features_, axis='columns'),
				one_hotted
			], axis=1, sort=False)

		else:

			return X

