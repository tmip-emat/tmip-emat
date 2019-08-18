
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils import Bunch

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


from .frameable import FrameableMixin
from .boosting import BoostedRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.base import clone
from .linear import LinearRegression
from .anisotropic import AnisotropicGaussianProcessRegressor as AGPR
from sklearn.multioutput import RegressorChain
from sklearn.utils.validation import check_is_fitted


class EnsembleRegressorChain2(
	VotingRegressor,
	FrameableMixin,
):

	def __init__(
			self,
			base_estimator=None,
			n_chains=5,
			weights=None,
			n_jobs=None,
			random_state=None,
			cv=None,
	):
		self.base_estimator = base_estimator
		self.n_chains = n_chains
		self.random_state = random_state
		self.cv = cv
		super().__init__(None, weights=weights, n_jobs=n_jobs)

	def fit(self, X, y, sample_weight=None):

		if sample_weight is not None:
			raise NotImplementedError()

		if self.base_estimator is None:
			base_estimator = BoostedRegressor([
				LinearRegression(frame_out=True),
				AGPR(n_restarts_optimizer=250),
			])
		else:
			base_estimator = self.base_estimator
		random_state = None
		if isinstance(self.random_state, int):
			random_state = self.random_state
		elif self.random_state is not None:
			random_state = self.random_state.randint(2**30)

		self.estimators = []
		rc = RegressorChain(
			base_estimator,
			order='random',
			cv=self.cv,
		)
		for n in range(self.n_chains):
			n_rc = clone(rc)
			if random_state is not None:
				n_rc.random_state = random_state + n
			i = (f'chain{n}', n_rc)
			self.estimators.append(i)

		self._pre_fit(X, y)
		#return super().fit(X, y, sample_weight=sample_weight)
		"""
		common fit operations.
		"""
		if self.estimators is None or len(self.estimators) == 0:
			raise AttributeError('Invalid `estimators` attribute, `estimators`'
								 ' should be a list of (string, estimator)'
								 ' tuples')

		if (self.weights is not None and
				len(self.weights) != len(self.estimators)):
			raise ValueError('Number of `estimators` and weights must be equal'
							 '; got %d weights, %d estimators'
							 % (len(self.weights), len(self.estimators)))

		names, clfs = zip(*self.estimators)
		self._validate_names(names)

		n_isnone = np.sum(
			[clf in (None, 'drop') for _, clf in self.estimators]
		)
		if n_isnone == len(self.estimators):
			raise ValueError(
				'All estimators are None or "drop". At least one is required!'
			)

		# self.estimators_ = Parallel(n_jobs=self.n_jobs)(
		# 	delayed(_parallel_fit_estimator)(clone(clf), X, y,
		# 									 sample_weight=sample_weight)
		# 	for clf in clfs if clf not in (None, 'drop')
		# )

		self.estimators_ = []
		for (clfname, clf) in self.estimators:
			e = clone(clf).fit(X,y)
			self.estimators_.append(e)

		self.named_estimators_ = Bunch()
		for k, e in zip(self.estimators, self.estimators_):
			self.named_estimators_[k[0]] = e
		return self

	def predict(self, X):
		"""
		Predict regression target for X.

		The predicted regression target of an input sample is computed as the
		mean predicted regression targets of the estimators in the ensemble.

		Parameters
		----------
		X : {array-like, sparse matrix} of shape (n_samples, n_features)
			The input samples.

		Returns
		-------
		y : array of shape (n_samples, n_targets)
			The predicted values.
		"""
		check_is_fitted(self, "estimators_")
		y_hat = np.average(
			self._predict(X).T,
			axis=0,
			weights=self._weights_not_none,
		)
		try:
			return self._post_predict(X, y_hat)
		except:
			print("y_hat")
			print(y_hat)
			raise