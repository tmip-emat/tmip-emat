
import pandas
from sklearn.multioutput import MultiOutputRegressor as _MultiOutputRegressor
from .frameable import FrameableMixin
from .model_selection import CrossValMixin

class MultiOutputRegressor(_MultiOutputRegressor, FrameableMixin, CrossValMixin):

	def fit(self, X, y, sample_weight=None):
		self._pre_fit(X,y)
		if isinstance(y, pandas.Series):
			y = y.to_frame()
		elif y.ndim == 1:
			y = y.reshape(-1,1)
		return super().fit(X, y, sample_weight=sample_weight)

	def predict(self, X):
		y = super().predict(X)
		y = self._post_predict(X,y)
		return y

