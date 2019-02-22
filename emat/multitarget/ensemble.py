
from sklearn.base import RegressorMixin, BaseEstimator

from .chained import ChainedTargetRegression
from .cross_val import CrossValMixin
from .detrend import DetrendMixin

class EnsembleRegressorChains(
		BaseEstimator,
		RegressorMixin,
		CrossValMixin,
):

	def __init__(self, keep_other_features=3, step2_cv_folds=5, replication=10):
		self.replication = replication
		self.keep_other_features = keep_other_features
		self.step2_cv_folds = step2_cv_folds
		self.ensemble = [
			ChainedTargetRegression(
				keep_other_features=keep_other_features,
				step2_cv_folds=step2_cv_folds,
				randomize_chain=n,
			)
			for n in range(self.replication)
		]

	def fit(self, X, Y):
		for c in self.ensemble:
			c.fit(X,Y)
		return self

	def predict(self, X):
		result = self.ensemble[0].predict(X)
		for c in self.ensemble[1:]:
			result += c.predict(X)
		result /= len(self.ensemble)
		return result



class DetrendedEnsembleRegressorChains(
	EnsembleRegressorChains,
	DetrendMixin
):

	def fit(self, X, Y):
		return super().fit(X, self.detrend_fit(X,Y))

	def predict(self, X, return_std=False, return_cov=False):
		return self.detrend_predict(X) + super().predict(X)


