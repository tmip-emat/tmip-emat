import numpy
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def lhs(n_factors, n_samples, genepool=10000, random_in_cell=True):
    """

    Parameters
    ----------
    n_factors : int
        The number of columns to sample
    n_samples : int
        The number of Latin hypercube samples (rows)
    genepool : int
        The nubmer of random permutation from which to find
                good (uncorrelated) columns
    random_in_cell : bool, default True
        If true, a uniform random point in each hypercube cell
                is chosen, otherwise the center point in each cell is chosen.

    Returns
    -------
    ndarray
    """
    candidates = numpy.empty([genepool, n_samples], dtype=numpy.float64)
    for i in range(genepool):
        candidates[i, :] = numpy.random.permutation(n_samples)
    corr = numpy.fabs(numpy.corrcoef(candidates))
    keepers = [0]
    keeper_gross_corr = 0
    for j in range(n_factors - 1):
        keeper_gross_corr += corr[keepers[-1], :]
        k = numpy.argmin(keeper_gross_corr)
        keepers.append(k)

    lhs = candidates[keepers, :].copy()
    if random_in_cell:
        lhs += numpy.random.rand(*(lhs.shape))
    else:
        lhs += 0.5
    lhs /= n_samples
    return lhs


def _avg_off_diag(a):
    upper = numpy.triu_indices(a.shape[0], 1)
    lower = numpy.tril_indices(a.shape[0], -1)
    return (a[upper].mean() + a[lower].mean()) / 2


def _stduniform_to_stdnormal(x):
    return norm(0, 1).ppf(x)


def _stdnormal_to_stduniform(x):
    return norm(0, 1).cdf(x)


def _make_correlated(x, corr, dim=0):
    d = x.shape[dim]
    s = numpy.full([d, d], fill_value=corr) + numpy.eye(d) * (1 - corr)
    chol = numpy.linalg.cholesky(s)
    return numpy.dot(chol, x)


def _induce_correlation(x, approx_corr, return_actual_corr=False):
    x = _stduniform_to_stdnormal(x)
    x = _make_correlated(x, approx_corr)
    if return_actual_corr:
        result = _stdnormal_to_stduniform(x)
        return _avg_off_diag(numpy.corrcoef(result))
    return _stdnormal_to_stduniform(x)


def induce_correlation(h, corr, rows=None, inplace=False):
    h_full = h
    if rows:
        h = h[rows, :]

    _target_corr = lambda m: (_induce_correlation(h, m, return_actual_corr=True) - corr) ** 2
    result = minimize_scalar(_target_corr, bounds=(0, 1), method='Bounded')
    h_result = _induce_correlation(h, result.x)

    if inplace:
        if rows:
            h_full[rows, :] = h_result[:]
        else:
            h[:] = h_result[:]
    else:
        return h_result


def lhs_corr(n_factors, n_samples, genepool=10, sigma=None,
             random_in_cell=True):
    """
    Correlated LHS.  Works only in theory, or for low correllation / small sample sizes

    Parameters
    ----------
    n_factors : int
        The number of columns to sample
    n_samples : int
        The number of Latin hypercube samples (rows)
    genepool : int
        The nubmer of random permutation from which to find good (uncorrelated) columns
    sigma : array
        The desired correlation matrix
    random_in_cell : bool, default True
        If true, a uniform random point in each hypercube cell is chosen, otherwise the
        center point in each cell is chosen.

    Returns
    -------
    ndarray
    """
    keepers = []
    candidates = numpy.empty([genepool, n_samples], dtype=numpy.float64)
    keeper_corr = numpy.zeros([n_factors, genepool])
    corr = None

    def _reroll():
        nonlocal candidates, keepers, corr
        for i in range(genepool):
            if i not in keepers:
                candidates[i, :] = numpy.random.permutation(n_samples)
        corr = numpy.corrcoef(candidates)
        o_corr = corr[corr < 0.999]
        # print("max_o_corr",numpy.max(o_corr))
        for j in range(len(keepers)):
            keeper_corr[j, :] = corr[keepers[j], :]

    _reroll()

    if sigma is None:
        sigma = numpy.eye(n_factors)

    # initially keep the first column
    keepers.append(0)
    for j in range(n_factors - 1):
        # load in the previously determined correlation row
        keeper_corr[j, :] = corr[keepers[-1], :]
        # ident the relevant part of sigma for comparing
        sig_part = sigma[j + 1, :j + 1]
        print("sig_part", sig_part)
        for repeat in range(50000):
            # find the closest match in the gene pool for target correlation
            kc = keeper_corr[:j + 1, :]
            kc = kc[kc < 0.999]
            if repeat % 1000 == 0:
                print(repeat, "max_corr", numpy.max(kc))
            sq_diff_from_target = ((keeper_corr[:j + 1, :] - sig_part[:, None])
                                   **2).sum(0)
            k = numpy.argmin(sq_diff_from_target)
            if sq_diff_from_target[k] < 0.001:
                break
            else:
                _reroll()
        keepers.append(k)
        print(keepers)

    lhs = candidates[keepers, :].copy()
    if random_in_cell:
        lhs += numpy.random.rand(*(lhs.shape))
    else:
        lhs += 0.5
    lhs /= n_samples
    return lhs
