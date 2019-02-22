

import pandas, numpy
from . import ignore_warnings

from sklearn.metrics import r2_score
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_score, cross_val_predict


class CrossValMixin:

	def cross_val_scores(self, X, Y, cv=5):
		"""
		Calculate the cross validation scores for this model.

		Args:
			X, Y : array-like
				The independent and dependent data to use for
				cross-validation.
			cv : int, default 5
				The number of folds to use in cross-validation.

		Returns:
			pandas.Series: The cross-validation scores, by output.

		"""
		p = self.cross_val_predict(X, Y, cv=cv)
		return pandas.Series(
			r2_score(Y, p, sample_weight=None, multioutput='raw_values'),
			index=Y.columns
		)

	def cross_val_predict(self, X, Y, cv=3):
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