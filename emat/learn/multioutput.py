
import pandas
from sklearn.multioutput import MultiOutputRegressor as _MultiOutputRegressor
from .frameable import FrameableMixin
from .model_selection import CrossValMixin

class MultiOutputRegressor(_MultiOutputRegressor, FrameableMixin, CrossValMixin):

	def fit(self, X, y, sample_weight=None):
		self._pre_fit(X,y)
		return super().fit(X, y, sample_weight=sample_weight)

	def predict(self, X):
		y = super().predict(X)
		y = self._post_predict(X,y)
		return y

