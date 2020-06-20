
import pandas
import numpy

def perturb_categorical(x, range_padding=0, add_variance=True):
	"""
	Add some random perturbation to categorical data for visualization.

	No perturbation is added to integer or float data.

	Parameters
	----------
	x : array-like
	range_padding : float, optional

	Returns
	-------
	x : array-like
		The new values, which are floats.
	x_ticktext : None or List[Str]
		The tick labels (i.e. category names) if any were found.
	x_tickvals : None or List[Int]
		The tick label positions (i.e. category indexes) if any
		categories were found.
	x_range : None or Tuple(Float,Float)
		The lower and upper bounds of the range of perturbed
		values, extended by range_padding.
	valid_scales : List
		Either ['linear'] or ['linear', 'log'], the latter only
		if the data is integer or float and is strictly positive.

	"""

	valid_scales = ['linear']
	x_ticktext = None
	x_tickvals = None
	x_range = None
	if not isinstance(x, pandas.Series):
		x = pandas.Series(x)
	try:
		is_bool = numpy.issubdtype(x.dtype, numpy.bool_)
	except TypeError:
		is_bool = False
	if is_bool:
		x = x.astype(pandas.CategoricalDtype(categories=[False, True], ordered=False))
	if isinstance(x.dtype, pandas.CategoricalDtype):
		x_categories = x.cat.categories
		codes = x.cat.codes
		x = codes.astype(float)
		if add_variance:
			s_ = x.size * 0.01
			s_ = s_ / (1 + s_)
			epsilon = 0.05 + 0.20 * s_
			x = x + numpy.random.uniform(-epsilon, epsilon, size=x.shape)
		else:
			epsilon = 0
		x_ticktext = list(x_categories)
		x_tickvals = list(range(len(x_ticktext)))
		x_range = [-epsilon-range_padding, x_tickvals[-1]+epsilon+range_padding]
	else:
		if x.min() > 0:
			valid_scales.append('log')

	return x, x_ticktext, x_tickvals, x_range, valid_scales
