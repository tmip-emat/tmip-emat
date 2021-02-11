import pandas as pd

def cross_validation_styling(scores):
	"""
	Dataframe styling for cross validation scores.

	Replaces scores worse than -2 with "bad", and
	highlights poor scores with warning colors:
	- yellow if less than 0.75
	- orange if less that 0.50
	- red if less than 0.25, including 'bad'
	"""

	def _color(val):
		"""
		Takes a scalar and returns a string with
		the css property `'color: xxx'` for certain
		values, black otherwise.
		"""
		bgcolor = ''
		if val == 'bad':
			return 'background-color: red'
		if val < 0.75:
			bgcolor = 'yellow'
		if val < 0.50:
			bgcolor = 'orange'
		if val < 0.25:
			bgcolor = 'red'
		if bgcolor:
			return 'background-color: %s' % bgcolor
		return ''

	def _fmt(val):
		if val < -2:
			return "bad"
		return "{:.4f}".format(val)

	return pd.DataFrame(scores).style.applymap(_color).format(_fmt)


def feature_score_styling(scores, cmap='viridis'):
	"""
	Dataframe styling for feature scores.

	Per-row background gradient, with the maximum
	set to the row max, and the minimum set to zero.
	"""

	return scores.style.background_gradient(
		cmap=cmap,
		axis=1,
		text_color_threshold=0.5,
		vmin=0.0,
	)