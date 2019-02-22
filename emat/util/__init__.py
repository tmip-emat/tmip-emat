
from . import distributions

def close_excess_files():
	import psutil
	import os
	proc = psutil.Process()
	for i in proc.open_files():
		if 'matplotlib' in i.path and '.ttf' in i.path:
			os.close(i.fd)

def current_open_files():
	import psutil
	import os
	proc = psutil.Process()
	return proc.open_files()




def list_rv_frozen():
	"""
	Create a new rv_frozen distribution.
	"""
	print(list(distributions.__dict__.keys()))


def make_rv_frozen(name=None, discrete=False, min=None, max=None, **kwargs):
	"""
	Create a new rv_frozen distribution.
	"""
	if name == 'constant':
		return

	if name is None:
		name = 'randint' if discrete else 'uniform'

	if name == 'uniform' and discrete:
		name = 'randint'

	if hasattr(distributions, name):
		gen = getattr(distributions, name)
	else:
		raise NameError(f"no known distribution named '{name}'")

	# special processing of particular keywords
	if min is not None and max is not None:
		if name == 'uniform':
			kwargs['loc'] = min
			kwargs['scale'] = max - min
		elif name == 'triangle':
			kwargs['lower_bound'] = min
			kwargs['upper_bound'] = max
		elif name == 'pert':
			kwargs['lower_bound'] = min
			kwargs['upper_bound'] = max
		elif name == 'randint':
			kwargs['low'] = min
			kwargs['high'] = max+1

	frozen = gen(**kwargs)

	if discrete:
		if not isinstance(frozen.dist, distributions.rv_discrete):
			raise TypeError(f"distribution named '{name}' is not discrete")
	return frozen
