

from scipy.stats import *
from scipy._lib._util import _lazyselect

def _peaked_distribution_args(
		lower_bound,
		upper_bound=None,
		rel_peak=None,
		peak=None,
		width=None,
):
	if peak is None and rel_peak is None:
		rel_peak = 0.5

	if peak is not None and rel_peak is not None:
		raise ValueError("cannot give both peak and rel_peak")

	if width is None and upper_bound is None:
		raise ValueError("must give upper_bound or width")

	if width is None:
		width = upper_bound - lower_bound

	if rel_peak is None:
		rel_peak = (peak-lower_bound) / width

	if width < 0:
		raise ValueError("cannot have negative width")

	if upper_bound is None:
		upper_bound = lower_bound + width

	if peak is None:
		peak = (rel_peak * width) + lower_bound

	if not lower_bound <= peak <= upper_bound:
		raise ValueError('peak must be between lower and upper bounds')

	return lower_bound, upper_bound, rel_peak, peak, width


def triangle(
		lower_bound,
		upper_bound=None,
		*,
		rel_peak=None,
		peak=None,
		width=None,
):
	"""
	Generate a frozen scipy.stats.triang distribution.

	Args:
		lower_bound (numeric):
			The lower bound of the distribution.
		upper_bound (numeric, optional):
			The upper bound of the distribution.  Can be inferred
			from `width` if not given.
		rel_peak (numeric, optional):
			The relative position of the peak of the triangle.  Must
			be in the range (0,1).  If neither `peak` nor `rel_peak`
			is given, a default value of 0.5 is used.
		peak (numeric, optional):
			The location of the peak of the triangle, given as a particular
			value, which must be between the lower and upper bounds inclusive.
		width (numeric, optional):
			The distance between the lower and upper bounds.
			Can be inferred from those values if not given.

	Returns:
		scipy.stats.rv_frozen
	"""
	lower_bound, upper_bound, rel_peak, peak, width = _peaked_distribution_args(
		lower_bound, upper_bound, rel_peak, peak, width
	)

	return triang(c=rel_peak, loc=lower_bound, scale=width)



def pert(
		lower_bound,
		upper_bound=None,
		*,
		rel_peak=None,
		peak=None,
		width=None,
		gamma=4.0,
):
	"""
	Generate a frozen scipy.stats.beta PERT distribution.

	For details on the PERT distribution see
	`https://en.wikipedia.org/wiki/PERT_distribution`_

	Args:
		lower_bound (numeric):
			The lower bound of the distribution.
		upper_bound (numeric, optional):
			The upper bound of the distribution.  Can be inferred
			from `width` if not given.
		rel_peak (numeric, optional):
			The relative position of the peak of the triangle.  Must
			be in the range (0,1).  If neither `peak` nor `rel_peak`
			is given, a default value of 0.5 is used.
		peak (numeric, optional):
			The location of the peak of the triangle, given as a particular
			value, which must be between the lower and upper bounds inclusive.
		width (numeric, optional):
			The distance between the lower and upper bounds.
			Can be inferred from those values if not given.

	Returns:
		scipy.stats.rv_frozen
	"""

	lower_bound, upper_bound, rel_peak, peak, width = _peaked_distribution_args(
		lower_bound, upper_bound, rel_peak, peak, width
	)

	a, b, c = [float(x) for x in [lower_bound, peak, upper_bound]]
	if gamma < 0:
		raise ValueError('gamma must be non-negative')
	mu = (a + gamma * b + c) / (gamma + 2)
	if mu == b:
		a1 = a2 = 3.0
	else:
		a1 = ((mu - a) * (2 * b - a - c)) / ((b - mu) * (c - a))
		a2 = a1 * (c - mu) / (mu - a)

	if not (a1 > 0 and a2 > 0):
		raise ValueError('Beta "alpha" and "beta" parameters must be greater than zero')

	return beta(a=a1, b=a2, loc=a, scale=c-a)

def binary(p=0.5):
	return bernoulli(p=p)

def constant(
		lower_bound,
		upper_bound=None,
		*,
		value=None,
		default=None,
):
	v = value
	if v is None:
		v = default
	if v is None:
		v = lower_bound
	if v is None:
		v = upper_bound
	return uniform(v,0)

class truncated:

	def __init__(self, frozen_dist, lower_bound, upper_bound):
		self.frozen_dist = frozen_dist
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound

		self.mass_below_lower_bound = self.frozen_dist.cdf(lower_bound)
		total_truncated_mass = (1-self.frozen_dist.cdf(upper_bound)
								+self.mass_below_lower_bound)
		self.untruncated_mass = (1-total_truncated_mass)

	def rvs(self, *args, **kwargs):
		u = uniform(0,1).rvs(*args, **kwargs)
		return self.ppf(u)

	def ppf(self, x):
		return self.frozen_dist.ppf(
			(x * self.untruncated_mass) + self.mass_below_lower_bound
		)

	def cdf(self, x):
		r = _lazyselect([x < self.lower_bound,
						 self.lower_bound <= x <= self.upper_bound,
						 x > self.upper_bound],
						[lambda x: 0,
						 lambda x: ((self.frozen_dist.cdf(x)-self.mass_below_lower_bound)
									*self.untruncated_mass),
						 lambda x: 1],
						(x, ))
		return r

	def sf(self, x):
		return 1-self.cdf(x)

	def pdf(self, x):
		r = _lazyselect([x < self.lower_bound,
						 (self.lower_bound <= x) & (x <= self.upper_bound),
						 x > self.upper_bound],
						[lambda x: 0,
						 lambda x: self.frozen_dist.pdf(x)/self.untruncated_mass,
						 lambda x: 0],
						(x, ))
		return r

	def logpdf(self, x):
		r = _lazyselect([x < self.lower_bound,
						 (self.lower_bound <= x) & (x <= self.upper_bound),
						 x > self.upper_bound],
						[lambda x: 0,
						 lambda x: self.frozen_dist.logpdf(x)-np.log(self.untruncated_mass),
						 lambda x: 0],
						(x, ))
		return r

	def stats(self):
		raise NotImplementedError("not implemented for truncated")

	def entropy(self):
		raise NotImplementedError("not implemented for truncated")



