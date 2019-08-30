

from sklearn.multioutput import RegressorChain as _RegressorChain
from .ensemble import VotingRegressor
from .frameable import FrameableMixin

class RegressorChain(
	_RegressorChain,
	FrameableMixin,
):
	def fit(self, X, Y):
		self._pre_fit(X, Y)
		return super().fit(X,Y)

	def predict(self, X):
		Y = super().predict(X)
		Y = self._post_predict(X, Y)
		return Y


def EnsembleRegressorChain(
		base_estimator,
		n_chains,
):

	ensemble = []
	for n in range(n_chains):
		e = RegressorChain(
			base_estimator=base_estimator,
			order='random',
			random_state=n,
		)
		ensemble.append((f'chain_{n}',e))
	return VotingRegressor(
		ensemble
	)