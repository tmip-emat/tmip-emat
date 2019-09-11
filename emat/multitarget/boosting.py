


from sklearn.base import RegressorMixin, BaseEstimator, clone
from .frameable import FrameableMixin

class BoostedRegressor(BaseEstimator, RegressorMixin, FrameableMixin):

	def __init__(
			self,
			estimators,
	):
		self.estimators = estimators

	def fit(self, X, Y, sample_weight=None):

		if sample_weight is not None:
			raise NotImplementedError
		self._pre_fit(X, Y)
		self.estimators_ = []
		Y_ = Y
		for n,e in enumerate(self.estimators):
			e_ = clone(e)
			e_.fit(X, Y_)
			self.estimators_.append(e_)
			if n+1 < len(self.estimators):
				Y_ = Y_ - e_.predict(X)

	def predict(self, X):
		Yhat = self.estimators_[0].predict(X)
		for e_ in self.estimators_[1:]:
			Yhat += e_.predict(X)
		Yhat = self._post_predict(X, Yhat)
		return Yhat

