
from sklearn.base import clone
import inspect

def clone_or_construct(estimator):
	"""
	Clone an estimator, or construct a default one from a class or function.

	Parameters
	----------
	estimator : sklearn estimator instance, class, or a function returning same.

	Returns
	-------
	estimator
	"""
	try:
		return clone(estimator)
	except TypeError:
		if inspect.isclass(estimator) or inspect.isfunction(estimator):
			return estimator()
		else:
			raise

