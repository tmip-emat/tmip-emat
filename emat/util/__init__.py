
from . import distributions
import numpy as np
from scipy.stats._distn_infrastructure import rv_frozen

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


from ..exceptions import DistributionTypeError, DistributionFreezeError

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
		elif name == 'loguniform':
			kwargs['a'] = min
			kwargs['b'] = max

	try:
		frozen = gen(**kwargs)
	except Exception as err:
		raise DistributionFreezeError(f"cannot freeze {name} using:\n{kwargs}") from err

	if discrete:
		if not isinstance(frozen.dist, distributions.rv_discrete):
			raise DistributionTypeError(f"distribution named '{name}' is not discrete")
	return frozen


def rv_frozen_as_dict(frozen, min=None, max=None):
	if not isinstance(frozen, rv_frozen):
		return frozen
	if frozen.dist.name == 'Distribution':
		x = {'name': frozen.dist.__class__.__name__.replace("_gen", "")}
	else:
		x = {'name': frozen.dist.name}
	if frozen.args:
		x['args'] = frozen.args
	if frozen.kwds:
		x.update(frozen.kwds)

	if x.get('name') == 'uniform':
		if min is not None and x.get('loc') == min:
			if max is not None and x.get('scale') == max - min:
				return 'uniform'
		if min is not None and max is not None and x.get('args') == (min, max - min):
			return 'uniform'

	if x.get('name') == 'triang':
		if min is not None and x.get('loc') == min:
			if max is not None and x.get('scale') == max - min:
				peak = x.get('c') * x.get('scale') + x.get('loc')
				return { 'name':'triangle', 'peak':peak }

	if x.get('name') == 'randint':
		if min is not None and x.get('low') == min:
			if max is not None and x.get('high') == max + 1:
				return 'uniform'
		if min is not None and max is not None and x.get('args') == (min, max + 1):
			return 'uniform'

	if x.get('name') == 'beta':
		if min is not None and x.get('loc') == min:
			if max is not None and x.get('scale') == max - min:
				rel_peak = (x.get('a') - 1) / (x.get('a') + x.get('b') - 2)
				peak = x.get('loc') + rel_peak * x.get('scale')
				mean = (1 / (1 + (x.get('b') / x.get('a')))) * x.get('scale') + x.get('loc')
				if np.absolute(peak - mean) < 1e-8:
					gamma = (x.get('a') - 1) * 2
				else:
					gamma = (x.get('loc') * 2 + x.get('scale')) / mean - 2
					gamma /= (1 - peak / mean)
				if np.absolute(gamma - 4) < 1e-5:
					return {'name': 'pert', 'peak': peak}
				return {'name': 'pert', 'peak': peak, 'gamma': gamma}

	return x


def n_jobs_cap(n_jobs):
	"""
	Cap the number of jobs for sklearn tasks on Windows.

	https://github.com/scikit-learn/scikit-learn/issues/13354

	Args:
		n_jobs: int

	Returns:
		n_jobs
	"""
	if n_jobs is None or n_jobs < 0 or n_jobs > 60:
		# Bug in windows if more than 60 jobs
		# https://github.com/scikit-learn/scikit-learn/issues/13354
		import platform
		if platform.system() == 'Windows':
			if n_jobs is None or n_jobs < 0:
				import multiprocessing
				n_jobs = max(multiprocessing.cpu_count() - 2, 1)
			n_jobs = min(n_jobs, 60)
	return n_jobs


def time_from_uuid(u):
	import datetime
	return datetime.datetime.fromtimestamp((u.time - 0x01b21dd213814000)*100/1e9)
