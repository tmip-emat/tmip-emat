
from ..scope.measure import Measure

def nondominated_solutions(df, scope, robustness_functions, nearly=1e-5):
	"""
	Identify the set of non-dominated solutions among a set of candidate solutions.

	Parameters
	----------
	df : pandas.DataFrame
		Candidate solutions
	scope : emat.Scope
		The model scope
	robustness_functions : Collection[emat.Measure], optional
		Robustness functions
	nearly : float
		Threshold for removing nearly duplicate (or actually duplicate) solutions.

	Returns
	-------
	pandas.DataFrame
	"""

	keeps = set()
	flips = set()

	# flip all MINIMIZE outcomes (or unflip them if previously marked as flip)
	if robustness_functions is not None:
		for k in robustness_functions:
			if k.kind == Measure.MINIMIZE:
				flips.add(k.name)
			if k.kind == Measure.MAXIMIZE:
				keeps.add(k.name)
	if scope is not None:
		for k in scope.get_measures():
			if k.kind == Measure.MINIMIZE:
				flips.add(k.name)
			if k.kind == Measure.MAXIMIZE:
				keeps.add(k.name)

	keeps = [k for k in keeps if k in df.columns]
	flips = [k for k in flips if k in df.columns]

	solutions = df[keeps+flips].copy()
	solutions[flips] = -solutions[flips]

	solutions_ = solutions.values

	dominated_solutions = set()

	for i in range(len(solutions)):
		if i in dominated_solutions:
			continue
		for j in range(i+1, len(solutions)):
			if j in dominated_solutions:
				continue
			diff = (solutions_[i] - solutions_[j])
			dmax, dmin = diff.max(), diff.min()
			if dmax < 0:
				dominated_solutions.add(i)
			elif dmax == 0 and dmin < 0:
				dominated_solutions.add(i)
			elif dmin > 0:
				dominated_solutions.add(j)
			elif dmin == 0 and dmax > 0:
				dominated_solutions.add(j)
			elif nearly and dmax < nearly and dmin > -nearly:
				dominated_solutions.add(j)

	return df.loc[[k for k in range(len(solutions)) if k not in dominated_solutions]]


