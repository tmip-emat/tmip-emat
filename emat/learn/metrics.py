
from sklearn.metrics import r2_score as _r2_score
import pandas


def r2_score(y_true, y_pred, sample_weight=None, multioutput="raw_values"):
	"""
	R^2 (coefficient of determination) regression score function.

	Best possible score is 1.0 and it can be negative (because the
	model can be arbitrarily worse). A constant model that always
	predicts the expected value of y, disregarding the input features,
	would get a R^2 score of 0.0.

	Parameters
	----------
	y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
		Ground truth (correct) target values.

	y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
		Estimated target values.

	sample_weight : array-like of shape = (n_samples), optional
		Sample weights.

	multioutput : string in ['raw_values', 'uniform_average', 'variance_weighted'] or None or array-like of shape (n_outputs)

		Defines aggregating of multiple output scores.
		Array-like value defines weights used to average scores.
		Default is "uniform_average".

		'raw_values' :
			Returns a full set of scores in case of multioutput input.

		'uniform_average' :
			Scores of all outputs are averaged with uniform weight.

		'variance_weighted' :
			Scores of all outputs are averaged, weighted by the variances
			of each individual output.

		.. versionchanged:: 0.19
			Default value of multioutput is 'uniform_average'.

	Returns
	-------
	z : float or ndarray of floats
		The R^2 score or ndarray of scores if 'multioutput' is
		'raw_values'.

	Notes
	-----
	This is not a symmetric function.

	Unlike most other scores, R^2 score may be negative (it need not actually
	be the square of a quantity R).

	This metric is not well-defined for single samples and will return a NaN
	value if n_samples is less than two.
	"""

	result = _r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
	if multioutput == 'raw_values':
		if isinstance(y_true, pandas.DataFrame):
			return pandas.Series(result, index=y_true.columns)
		if isinstance(y_pred, pandas.DataFrame):
			return pandas.Series(result, index=y_pred.columns)
	return result