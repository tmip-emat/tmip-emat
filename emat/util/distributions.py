

from scipy.stats import *
from scipy._lib._util import _lazyselect
from scipy.stats._distn_infrastructure import rv_frozen


def get_bounds(rv):
	"""
	Get the lower and upper bounds for a distribution.

	Args:
		rv: A frozen distribution that has a ppf method,
			or an object with a `dist` attribute that is as such.

	Returns:
		tuple: (lower_bound, upper_bound)
	"""

	if not isinstance(rv, rv_frozen):
		if hasattr(rv, 'dist') and isinstance(rv.dist, rv_frozen):
			rv = rv.dist

	ppf_zero = 0

	if isinstance(rv.dist, rv_discrete):
		# ppf at actual zero for rv_discrete gives lower bound - 1
		# due to a quirk in the scipy.stats implementation
		# so we use the smallest positive float instead
		ppf_zero = 5e-324

	return (rv.ppf(ppf_zero), rv.ppf(1.0))


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

	This function provides the same actual distribution as ``triang``,
	but offers multiple intuitive ways to identify the peak and upper bound of the
	distribution, while ``triang`` is less flexible and intuitive (it must be defined
	using arguments labeled as 'c', 'loc', and 'scale').

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
	`wikipedia <https://en.wikipedia.org/wiki/PERT_distribution>`_.

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
		a1 = a2 = (1 + gamma/2)
	else:
		a1 = ((mu - a) * (2 * b - a - c)) / ((b - mu) * (c - a))
		a2 = a1 * (c - mu) / (mu - a)

	if not (a1 > 0 and a2 > 0):
		raise ValueError('Beta "alpha" and "beta" parameters must be greater than zero')

	return beta(a=a1, b=a2, loc=a, scale=c-a)

def binary(p=0.5, p_true=None, p_false=None):
	"""
	Generate a frozen scipy.stats.bernoulli distribution.

	The bernoulli distribution is for True/False outcomes,
	but allows for weighted (i.e. not 50%) probability of True.

	Args:
		p (numeric, default 0.5):
			The probability of a 'True' outcome.
		p_true (numeric, optional):
			Alias for `p`.
		p_false (numeric, optional):
			Alternative argument, will set `p` to the complement.  Ignored if
			`p_true` is also given.

	Returns:
		scipy.stats.rv_frozen
	"""
	if p_true is not None:
		p = p_true
	elif p_false is not None:
		p = 1-p_false
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



def is_discrete_dist(rv):
	"""Check if the (frozen) distribution is discrete."""

	if isinstance(rv, truncated):
		rv = rv.frozen_dist

	if not isinstance(rv, rv_frozen):
		if hasattr(rv, 'dist') and isinstance(rv.dist, rv_frozen):
			rv = rv.dist

	if isinstance(rv.dist, rv_discrete):
		return True

	return False


def get_distribution_bounds(rv):
	"""Get the bounds of a (frozen) distribution."""
	ppf_zero = 0
	if is_discrete_dist(rv):
		# ppf at actual zero for rv_discrete gives lower bound - 1
		# due to a quirk in the scipy.stats implementation
		# so we use the smallest positive float instead
		ppf_zero = 5e-324
	return rv.ppf(ppf_zero), rv.ppf(1.0)