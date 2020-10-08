
from sklearn.multioutput import check_is_fitted, check_array, Parallel, delayed, parallel_helper, np, MultiOutputEstimator

def predict_std(self, X, return_std=True):
	"""Predict multi-output variable using a model
	 trained for each target variable.

	Parameters
	----------
	X : (sparse) array-like, shape (n_samples, n_features)
		Data.
	return_std : bool
		Whether to return the standard deviation of the estimates.

	Returns
	-------
	y : (sparse) array-like, shape (n_samples, n_outputs)
		Multi-output targets predicted across multiple predictors.
		Note: Separate models are generated for each predictor.
	std : (sparse) array-like, shape (n_samples, n_outputs)
		Standard deviations of multi-output targets
	"""
	check_is_fitted(self, 'estimators_')
	if not hasattr(self.estimator, "predict"):
		raise ValueError("The base estimator should implement a predict method")

	X = check_array(X, accept_sparse=True)

	if not return_std:
		raise TypeError('only use predict_std to access return_std')

	from ..util import n_jobs_cap
	y = Parallel(n_jobs=n_jobs_cap(self.n_jobs))(
		delayed(parallel_helper)(e, 'predict', X, return_std)
		for e in self.estimators_)

	result = np.asarray(y)
	return result[:,0].T, result[:,1].T


MultiOutputEstimator.predict_std = predict_std

