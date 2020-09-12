import os
import numpy
import pandas
import warnings
import operator
from scipy import stats
from typing import Mapping

from ..workbench.em_framework.samplers import (
    AbstractSampler,
    LHSSampler,
    UniformLHSSampler,
    MonteCarloSampler,
    DefaultDesigns,
)

from ..exceptions import AsymmetricCorrelationError

def induce_correlation(std_uniform_sample, correlation_matrix, inplace=False):
    """
    Induce correlation in an independent standard uniform sample.

    The correlation is induced on a standard normal transformation
    of the input sample, which is then inverted, so that the final
    correlation of the uniform sample outputs may have a slightly
    different correlation than the defined `correlation_matrix`.

    Args:
        std_uniform_sample (array-like, shape [M,N]): An initial sample to modify.
            This sample should have M rows, one for each sampled observation
            and N columns, one for each variable.  Each variable should be
            samples from an independent standard uniform random variable
            in the 0-1 range.
        correlation_matrix (array-like, shape [N,N]):
            The correlation matrix that will be induced.  This must be a
            symmetric positive definite matrix with 1's on the diagonal.
        inplace (bool, default False): Whether to modify the input
            sample in-place.

    Returns:
        array-like, shape [M,N]: The correlated sample.
    """
    from scipy.stats import norm
    std_normal_sample = norm.ppf(std_uniform_sample)

    try:
        chol = numpy.linalg.cholesky(correlation_matrix)
    except numpy.linalg.LinAlgError as err:
        raise numpy.linalg.LinAlgError("failed correlation_matrix is\n"+str(correlation_matrix)) from err

    cor_normal_sample = chol.dot(std_normal_sample.T).T

    if inplace:
        std_uniform_sample[:, :] = norm.cdf(cor_normal_sample)
    else:
        cor_uniform_sample = norm.cdf(cor_normal_sample)
        return cor_uniform_sample


class CorrelatedSampler(AbstractSampler):

    def sample_std_uniform(self, size):
        raise NotImplementedError

    def generate_std_uniform_samples(self, parameters, size):
        '''
        The main method of :class: `~sampler.Sampler` and its
        children. This will call the sample method for each of the
        parameters and return the resulting designs.

        Args:
            parameters (Collection): a collection of emat.Parameter instances.
            size (int): the number of samples to generate.

        Returns:
            pandas.DataFrame

        '''
        return pandas.DataFrame({
            param.name: self.sample_std_uniform(size)
            for param in parameters
        })

    def get_correlation_matrix(
            self,
            parameters,
            validate=True,
            presorted=False,
            none_if_none=False,
    ):
        """
        Extract a correlation matrix from parameters.

        Args:
            parameters (Collection): Parameters for which to generate the
                correlation matrix for experimental designs
            validate (bool, default True): Check that the given
                correlation matrix is positive definite (a numerical
                requirement for any correlation matrix) and raise
                an error if it is not.
            presorted (bool, default False): If parameters are already
                sorted by name, set this to True to skip re-sorting.
            none_if_none (bool, default False): If there is no active
                correlation (i.e., the correlation matrix is an identity
                matrix) return None instead of the matrix.

        Returns:
            pandas.DataFrame or None
        """
        if not presorted:
            parameters = sorted(parameters, key=operator.attrgetter('name'))
        parameter_names = [i.name for i in parameters]

        any_corr = False

        # Define correlation matrix
        correlation = pandas.DataFrame(
            data=numpy.eye(len(parameter_names)),
            index=parameter_names,
            columns=parameter_names,
        )
        for p in parameters:
            corr = dict(getattr(p, 'corr', {}))
            for other_name, other_corr in corr.items():
                if correlation.loc[p.name, other_name] != 0:
                    # When correlation is already set, confirm it is identical
                    # or raise an exception
                    if correlation.loc[p.name, other_name] != other_corr:
                        raise AsymmetricCorrelationError(f"{p.name}, {other_name}")
                else:
                    any_corr = True
                    correlation.loc[p.name, other_name] = other_corr
                    correlation.loc[other_name, p.name] = other_corr

        if any_corr and validate:
            eigenval, eigenvec = numpy.linalg.eigh(correlation)
            if numpy.min(eigenval) < 0:
                raise numpy.linalg.LinAlgError("correlation matrix is not positive semi-definite")
            elif numpy.min(eigenval) <= 0.001:
                import warnings
                warnings.warn("correlation matrix is nearly singular, expect numerical problems")

        if not any_corr and none_if_none:
            return None

        return correlation

    def generate_designs(self, parameters, nr_samples):
        """
        External interface to sampler.

        Returns the computational experiments
        over the specified parameters, for the given number of samples for each
        parameter.

        Args:
            parameters (Collection): Parameters for which to generate the
                experimental designs
            nr_samples (int): the number of samples to draw for each parameter

        Returns:
            DefaultDesigns
                a generator object that yields the designs resulting from
                combining the parameters
        """
        parameters = sorted(parameters, key=operator.attrgetter('name'))

        # Define correlation matrix
        correlation = self.get_correlation_matrix(parameters, presorted=True)

        if correlation is None:
            sampled_parameters = self.generate_samples(parameters, nr_samples)
        else:
            sampled_parameters = self.generate_std_uniform_samples(parameters, nr_samples)

            # Induce correlation
            induce_correlation(sampled_parameters.values, correlation.values, inplace=True)

            # Apply distribution shapes
            for p in parameters:
                sampled_parameters[p.name] = p.dist.ppf(sampled_parameters[p.name])

        # Construct designs per usual workbench approach
        designs = zip(*[sampled_parameters[u.name] for u in parameters])
        designs = DefaultDesigns(designs, parameters, nr_samples)

        return designs


class CorrelatedLHSSampler(CorrelatedSampler, LHSSampler):
    """
    generates a Latin Hypercube sample for each of the parameters
    """

    def sample_std_uniform(self, size):
        '''
        Generate a standard uniform Latin Hypercube Sample.

        Args:
            size (int): the number of samples to generate

        Returns:
            numpy.ndarray

        '''

        perc = numpy.linspace(0, (size - 1) / size, size)
        numpy.random.shuffle(perc)
        smp = stats.uniform(perc, 1. / size).rvs()
        return smp



class CorrelatedMonteCarloSampler(CorrelatedSampler, MonteCarloSampler):
    """
    Generator for correlated Monte Carlo samples for each of the parameters
    """

    def sample_std_uniform(self, size):
        '''
        Generate a standard uniform monte carlo sample.

        Args:
            size (int): the number of samples to generate

        Returns:
            numpy.ndarray

        '''

        smp = stats.uniform().rvs(size)
        return smp


class TrimmedUniformLHSSampler(LHSSampler):

    def __init__(self, trim_value=0.01):
        super().__init__()
        self.trim_level = trim_value / 2

    def generate_samples(self, parameters, size):
        '''

        Parameters
        ----------
        parameters : collection
        size : int

        Returns
        -------
        dict
            dict with the paramertainty.name as key, and the sample as value

        '''

        samples = {}
        for param in parameters:
            lower_bound = param.dist.ppf(self.trim_level)
            upper_bound = param.dist.ppf(1.0-self.trim_level)
            if isinstance(param.dist.dist, stats.rv_continuous):
                dist = stats.uniform(lower_bound, upper_bound - lower_bound)
            else:
                dist = stats.randint(lower_bound, upper_bound + 1)
            samples[param.name] = self.sample(dist, size)
        return samples

