

from sklearn.multioutput import MultiOutputRegressor as _MultiOutputRegressor
from .cross_val import CrossValMixin

class MultiOutputRegressor(
	_MultiOutputRegressor,
	CrossValMixin
):
	pass






