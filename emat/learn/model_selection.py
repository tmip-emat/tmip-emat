
import pandas, numpy
from pandas.util import hash_pandas_object
from .warnings import ignore_warnings

from sklearn.metrics import r2_score, make_scorer
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, RepeatedStratifiedKFold

def multiscore(Y, Y_pred, sample_weight=None):
	"""
	Returns the coefficients of determination R^2 of the prediction.

	The coefficient R^2 is defined as (1 - u/v), where u is the residual
	sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
	sum of squares ((y_true - y_true.mean()) ** 2).sum().
	The best possible score is 1.0 and it can be negative (because the
	model can be arbitrarily worse). A constant model that always
	predicts the expected value of y, disregarding the input features,
	would get a R^2 score of 0.0.

	Notes
	-----
	R^2 is calculated by weighting all the targets equally using
	`multioutput='raw_values'`.  See documentation for
	sklearn.metrics.r2_score for more information.

	Parameters
	----------
	X : array-like, shape = (n_samples, n_features)
		Test samples. For some estimators this may be a
		precomputed kernel matrix instead, shape = (n_samples,
		n_samples_fitted], where n_samples_fitted is the number of
		samples used in the fitting for the estimator.

	Y : array-like, shape = (n_samples, n_outputs)
		True values for X.

	sample_weight : array-like, shape = [n_samples], optional
		Sample weights.

	Returns
	-------
	score : ndarray
		R^2 of self.predict(X) wrt. Y.
	"""
	return r2_score(Y, Y_pred, sample_weight=sample_weight,
					multioutput='raw_values')

def single_multiscore(n=0):
	return lambda *args, **kwargs: multiscore(*args, **kwargs)[n]


def check_cv(cv='warn', y=None, classifier=False, random_state=None, n_repeats=1, shuffle=False):
	"""Input checker utility for building a cross-validator

	Parameters
	----------
	cv : int, cross-validation generator or an iterable, optional
		Determines the cross-validation splitting strategy.
		Possible inputs for cv are:

		- None, to use the default 3-fold cross-validation,
		- integer, to specify the number of folds.
		- :term:`CV splitter`,
		- An iterable yielding (train, test) splits as arrays of indices.

		For integer/None inputs, if classifier is True and ``y`` is either
		binary or multiclass, :class:`StratifiedKFold` is used. In all other
		cases, :class:`KFold` is used.

		Refer :ref:`User Guide <cross_validation>` for the various
		cross-validation strategies that can be used here.

		.. versionchanged:: 0.20
			``cv`` default value will change from 3-fold to 5-fold in v0.22.

	y : array-like, optional
		The target variable for supervised learning problems.

	classifier : boolean, optional, default False
		Whether the task is a classification task, in which case
		stratified KFold will be used.

	n_repeats : int, default 1
		If greater than 1, a 'RepeatedKFold' or 'RepeatedStratifiedKFold'
		will be used.

	shuffle : bool, default False
		Whether to shuffle the observation indexes before cross-validation.
		This happens automatically when n_repeats is greater than 1.

	random_state : int or RandomState
		Used for shuffling.

	Returns
	-------
	checked_cv : a cross-validator instance.
		The return value is a cross-validator which generates the train/test
		splits via the ``split`` method.
	"""
	import numbers
	from sklearn.utils.multiclass import type_of_target

	if n_repeats is None:
		n_repeats = 1

	if isinstance(cv, numbers.Integral):
		if (classifier and (y is not None) and (type_of_target(y) in ('binary', 'multiclass'))):
			if n_repeats>1:
				return RepeatedStratifiedKFold(n_splits=cv, random_state=random_state, n_repeats=n_repeats)
			else:
				return StratifiedKFold(cv, random_state=random_state if shuffle else None, shuffle=shuffle)
		else:
			if n_repeats>1:
				return RepeatedKFold(n_splits=cv, random_state=random_state, n_repeats=n_repeats)
			else:
				return KFold(cv, random_state=random_state if shuffle else None, shuffle=shuffle)

	return cv  # New style cv objects are passed without any modification



class CrossValMixin:

	def _cross_validate(
			self,
			X,
			Y,
			cv=5,
			S=None,
			random_state=None,
			cache_metadata=None,
			use_cache=True,
			n_repeats=1,
			shuffle=False,
			n_jobs=-1,
	):
		"""
		Compute the cross validation scores for this model.

		Unlike other scikit-learn scores, this method returns
		a separate score value for each output when the estimator
		is for a multi-output process.

		If the estimator includes a `sample_stratification`
		attribute, it is used along with the
		ExogenouslyStratifiedKFold splitter.

		Args:
			X, Y : array-like
				The independent and dependent data to use for
				cross-validation.
			cv : int, default 5
				The number of folds to use in cross-validation.
			S : array-like
				The stratification data to use for stratified
				cross-validation.  This data must be categorical
				(or convertible into such), and should be a
				vector of length equal to the first dimension
				(i.e. number of observations) in the `X` and `Y`
				arrays.

		Returns:
			pandas.Series: The cross-validation scores, by output.

		"""
		if not hasattr(self, '_cross_validate_results'):
			self._cross_validate_results = {}

		try:
			if random_state is None or not use_cache:
				raise KeyError()
			hashkey = hash((
				hash_pandas_object(X).sum(),
				hash_pandas_object(Y).sum(),
				cv,
				hash_pandas_object(S).sum() if S is not None else None,
				random_state,
				cache_metadata,
				n_repeats,
			))
		except:
			p = None
			hashkey = None
		else:
			p = self._cross_validate_results.get(hashkey, None)

		if p is None:
			if S is not None:
				from .splits import ExogenouslyStratifiedKFold, RepeatedExogenouslyStratifiedKFold
				if n_repeats > 1:
					cv = RepeatedExogenouslyStratifiedKFold(
						exo_data=S,
						n_splits=cv,
						random_state=random_state,
						n_repeats=n_repeats,
					)
				else:
					cv = ExogenouslyStratifiedKFold(
						exo_data=S,
						n_splits=cv,
						random_state=random_state,
					)

			if isinstance(Y, pandas.DataFrame):
				self.Y_columns = Y.columns
			elif isinstance(Y, pandas.Series):
				self.Y_columns = [Y.name]
			else:
				self.Y_columns = [f"Untitled_{j}" for j in range(Y.shape[1])]
			with ignore_warnings(DataConversionWarning):
				ms = {
					j: make_scorer(single_multiscore(n))
					for n,j in enumerate(self.Y_columns)
				}
				from sklearn.base import is_classifier
				cv = check_cv(cv, Y, classifier=is_classifier(self),
							  random_state=random_state, n_repeats=n_repeats,
							  shuffle=shuffle)
				from ..util import n_jobs_cap
				n_jobs = n_jobs_cap(n_jobs)
				p = cross_validate(self, X, Y, cv=cv, scoring=ms, n_jobs=n_jobs)

		if hashkey is not None:
			self._cross_validate_results[hashkey] = p
		return p

	def cross_val_scores(
			self,
			X,
			Y,
			cv=5,
			S=None,
			random_state=None,
			n_repeats=1,
			cache_metadata=None,
			n_jobs=-1,
			shuffle=False,
	):
		"""
		Calculate the cross validation scores for this model.

		Unlike other scikit-learn scores, this method returns
		a separate score value for each output when the estimator
		is for a multi-output process.

		Args:
			X, Y : array-like
				The independent and dependent data to use for
				cross-validation.
			cv : int, default 5
				The number of folds to use in cross-validation.
			S : array-like
				The stratification data to use for stratified
				cross-validation.  This data must be categorical
				(or convertible into such), and should be a
				vector of length equal to the first dimension
				(i.e. number of observations) in the `X` and `Y`
				arrays.
			random_state : int or RandomState
				Used primarily to shuffle observations for
				making cross-validation splits.
			n_repeats : int, optional
				Repeat the cross validation exercise this many
				times, with different random seeds, and return
				the average result.

		Returns:
			pandas.Series: The cross-validation scores, by output.

		"""
		p = self._cross_validate(
			X, Y, cv=cv, S=S, random_state=random_state,
			cache_metadata=cache_metadata, n_repeats=n_repeats,
			n_jobs=n_jobs, shuffle=shuffle,
		)
		try:
			return pandas.Series({j:p[f"test_{j}"].mean() for j in self.Y_columns})
		except:
			print("p=",p)
			print(len(self.Y_columns))
			print("self.Y_columns=",self.Y_columns)
			raise


	def cross_val_predict(self, X, Y, cv=5):
		if isinstance(Y, pandas.DataFrame):
			self.Y_columns = Y.columns
			Yix = Y.index
		elif isinstance(Y, pandas.Series):
			self.Y_columns = [Y.name]
			Yix = Y.index
		else:
			self.Y_columns = ["Untitled"] * Y.shape[1]
			Yix = pandas.RangeIndex(Y.shape[0])
		with ignore_warnings(DataConversionWarning):
			p = cross_val_predict(self, X, Y, cv=cv)
		try:

			return pandas.DataFrame(p, columns=self.Y_columns, index=Yix)

		except:
			print(p.shape)
			print(len(self.Y_columns))
			print(Yix.shape)
			raise


def take_best(estimator):

	def _bring_other(orig,copy):
		for attr in ['_Y_columns',]:
			if hasattr(orig, attr):
				setattr(copy, attr, getattr(orig, attr))
		return copy

	if hasattr(estimator, 'take_best_estimators'):
		result = estimator.take_best_estimators()
		return _bring_other(estimator, result)

	if hasattr(estimator, 'best_estimator_'):
		result = estimator.best_estimator_
		return _bring_other(estimator, result)

	return estimator
