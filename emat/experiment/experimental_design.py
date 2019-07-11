# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd

from ..scope.scope import Scope
from ..database.database import Database

from ..util.loggers import get_module_logger
_logger = get_module_logger(__name__)

from .samplers import (
    LHSSampler,
    AbstractSampler,
    UniformLHSSampler,
    MonteCarloSampler,
    CorrelatedLHSSampler,
    CorrelatedMonteCarloSampler,
    TrimmedUniformLHSSampler,
)

samplers = {
    'lhs': CorrelatedLHSSampler,
    'ulhs': UniformLHSSampler,
    'mc': CorrelatedMonteCarloSampler,
    'ulhs99': lambda: TrimmedUniformLHSSampler(0.01),
    'ulhs98': lambda: TrimmedUniformLHSSampler(0.02),
    'ulhs95': lambda: TrimmedUniformLHSSampler(0.05),
}

def design_experiments(
        scope: Scope,
        n_samples_per_factor: int = 10,
        n_samples: int = None,
        random_seed: int = 1234,
        db: Database = None,
        design_name: str = None,
        sampler = 'lhs',
        sample_from = 'all',
        jointly = True,
):
    """
    Create a design of experiments based on a Scope.

    Args:
        scope (Scope): The exploratory scope to use for the design.
        n_samples_per_factor (int, default 10): The number of samples in the
            design per random factor.
        n_samples (int or tuple, optional): The total number of samples in the
            design.  If `jointly` is False, this is the number of samples in each
            of the uncertainties and the levers, the total number of samples will
            be the square of this value.  Give a 2-tuple to set values for
            uncertainties and levers respectively, to set them independently.
            If this argument is given, it overrides `n_samples_per_factor`.
        random_seed (int or None, default 1234): A random seed for reproducibility.
        db (Database, optional): If provided, this design will be stored in the
            database indicated.
        design_name (str, optional): A name for this design, to identify it in the
            database. If not given, a unique name will be generated based on the
            selected sampler.  Has no effect if no `db` is given.
        sampler (str or AbstractSampler, default 'lhs'): The sampler to use for this
            design.  Available pre-defined samplers include:
                - 'lhs': Latin Hypercube sampling
                - 'ulhs': Uniform Latin Hypercube sampling, which ignores defined
                    distribution shapes from the scope and samples everything
                    as if it was from a uniform distribution
                - 'mc': Monte carlo sampling
                - 'uni': Univariate sensitivity testing, whereby experiments are
                    generated setting each parameter individually to minimum and
                    maximum values (for numeric dtypes) or all possible values
                    (for boolean and categorical dtypes).  Note that designs for
                    univariate sensitivity testing are deterministic and the number
                    of samples given is ignored.
        sample_from ('all', 'uncertainties', or 'levers'): Which scope components
            from which to sample.  Components not sampled are set at their default
            values in the design.
        jointly (bool, default True): Whether to sample jointly all uncertainties
            and levers in a single design, or, if False, to generate separate samples
            for levers and uncertainties, and then combine the two in a full-factorial
            manner.  This argument has no effect unless `sample_from` is 'all'.
            Note that jointly may produce a very large design;

    Returns:
        pandas.DataFrame: The resulting design.
    """
    if db is False:
        db = None

    if sampler == 'uni':
        return design_sensitivity_tests(scope, db, design_name or 'uni')

    if not isinstance(sampler, AbstractSampler):
        if sampler not in samplers:
            raise ValueError(f"unknown sampler {sampler}")
        else:
            sample_generator = samplers[sampler]()
    else:
        sample_generator = sampler

    np.random.seed(random_seed)

    if db is not None and design_name is not None:
        if design_name in db.read_design_names(scope.name):
            raise ValueError(f'the design "{design_name}" already exists for scope "{scope.name}"')

    # If using the default name, append the design_name with a number
    # until a new unused name is found.
    if db is not None and design_name is None:
        if isinstance(sampler, str):
            proposed_design_name = sampler
        elif hasattr(sampler, 'name'):
            proposed_design_name = str(sampler.name)
        else:
            proposed_design_name = str(sampler)
        existing_design_names = set(db.read_design_names(scope.name))
        if proposed_design_name not in existing_design_names:
            design_name = proposed_design_name
        else:
            n = 2
            while f'{proposed_design_name}_{n}' in existing_design_names:
                n += 1
            design_name = f'{proposed_design_name}_{n}'


    if sample_from == 'all' and not jointly:

        if n_samples is None:
            n_samples_u = n_samples_per_factor * len(scope.get_uncertainties())
            n_samples_l = n_samples_per_factor * len(scope.get_levers())
        elif isinstance(n_samples, tuple):
            n_samples_u, n_samples_l = n_samples
        else:
            n_samples_u = n_samples_l = n_samples

        samples_u = sample_generator.generate_designs(scope.get_uncertainties(), n_samples_u)
        samples_u.kind = dict
        design_u = pd.DataFrame.from_records([_ for _ in samples_u])

        samples_l = sample_generator.generate_designs(scope.get_levers(), n_samples_l)
        samples_l.kind = dict
        design_l = pd.DataFrame.from_records([_ for _ in samples_l])

        design_u["____"] = 0
        design_l["____"] = 0
        design = pd.merge(design_u, design_l, on='____')
        design.drop('____', 1, inplace=True)

    else:
        parms = []
        if sample_from in ('all', 'uncertainties'):
            parms += [i for i in scope.get_uncertainties()]
        if sample_from in ('all', 'levers'):
            parms += [i for i in scope.get_levers()]

        if n_samples is None:
            n_samples = n_samples_per_factor * len(parms)
        samples = sample_generator.generate_designs(parms, n_samples)
        samples.kind = dict
        design = pd.DataFrame.from_records([_ for _ in samples])

    if sample_from in ('all', 'constants'):
        for i in scope.get_constants():
            design[i.name] = i.default

    if db is not None and sample_from is 'all':
        experiment_ids = db.write_experiment_parameters(scope.name, design_name, design)
        design.index = experiment_ids
        design.index.name = 'experiment'

    return design



def design_sensitivity_tests(
        s: Scope,
        db: Database = None,
        design_name: str = 'uni',
):
    """
    Create a design of univariate sensitivity tests based on a Scope.

    If any of the parameters or levers have an infinite lower or upper bound,
    the sensitivity test will be made at the 1st or 99th percentile value
    respectively, instead of at the infinite value.

    Args:
        scope (Scope): The exploratory scope to use for the design.
        db (Database, optional): If provided, this design will be stored in the
            database indicated.
        design_name (str, optional): A name for this design, to identify it in the
            database. If not given, a unique name will be generated based on the
            selected sampler.  Has no effect if no `db` is given.

    Returns:
        pandas.DataFrame: The resulting design.
    """
    n_rows = sum(
        (
            len(p.values)
            if (hasattr(p,'values') and p.values is not None)
            else 2
        ) for p in s.get_parameters()
    )
    design = pd.DataFrame.from_dict(
        {p.name: np.full(n_rows, p.default) for p in s.get_parameters()}
    )
    n = 0
    for p in s.get_parameters():
        if not hasattr(p,'values') or p.values is None:
            design.loc[n, p.name] = p.min
            if not np.isfinite(design.loc[n, p.name]):
                try:
                    design.loc[n, p.name] = p.rv_gen.ppf(0.01)
                except:
                    pass
            n += 1
            design.loc[n, p.name] = p.max
            if not np.isfinite(design.loc[n, p.name]):
                try:
                    design.loc[n, p.name] = p.rv_gen.ppf(0.99)
                except:
                    pass
            n += 1
        else:
            for v in p.values:
                design.loc[n, p.name] = v
                n += 1
    design.drop_duplicates(inplace=True)
    design.reset_index(inplace=True, drop=True)
    
    if db is not None:
        db.write_experiment_parameters(s.name, design_name, design)

    return design


def minimum_weighted_distance(fixed_points, other_points, weights):
    """
    Compute minimum weighted distance from one array of points to another.

    Args:
        fixed_points (array-like):
            The fixed reference points.  Each column is a dimension, and
            each row is a point.
        other_points (array-like):
            The candidate measurement points.  Each column is a dimension,
            and each row is a point.  The columns must exactly
            match the columns in `fixed_points`, while the number of rows
            and the content thereof can be (and probably should be)
            entirely different.
        weights (vector):
            A set of weights by dimension.
            The values in this vector should correspond to the columns
            in `fixed_points` and `other_points`.

    Returns:
        numpy.ndarray:
            The values correspond to the rows in `other_points`.
    """
    array1 = np.asarray(fixed_points, dtype=float)
    array2 = np.asarray(other_points, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(1, -1)
    result = np.zeros(array2.shape[0], dtype=float)

    for i in range(array2.shape[0]):
        row = array2[i, :].reshape(1, -1)
        sq_dist_by_axis = (array1 - row) ** 2
        result[i] = (sq_dist_by_axis * w).sum(1).min()
    return result

def count_within_buffer(fixed_points, other_points, weights, buffer_dist=1):
    """
    Count the number of fixed points in a buffer various selected points.

    Args:
        fixed_points (array-like):
            The fixed reference points.  Each column is a dimension, and
            each row is a point.  These are the points that will be counted.
        other_points (array-like):
            The candidate measurement points.  Each column is a dimension,
            and each row is a point.  The columns must exactly
            match the columns in `fixed_points`, while the number of rows
            and the content thereof can be (and probably should be)
            entirely different.  These are the center points of the
            various buffers.
        weights (vector):
            A set of weights by dimension.
            The values in this vector should correspond to the columns
            in `fixed_points` and `other_points`.
        buffer_dist (float, default 1):
            A buffer distance.

    Returns:
        pandas.DataFrame:
            The columns of the returned data correspond to the columns
            in the `weights` input, and the row correspond to the rows
            int the `df2` argument.
    """
    array1 = np.asarray(fixed_points, dtype=float)
    array2 = np.asarray(other_points, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(1, -1)
    result = np.zeros(array2.shape[0], dtype=int)

    for i in range(array2.shape[0]):
        row = array2[i, :].reshape(1, -1)
        distances = np.sqrt(((array1 - row) ** 2) * w)
        result[i] = (distances <= buffer_dist).sum()
    return result

def value_within_buffer(fixed_points, fixed_values, other_points, weights, buffer_dist=1):
    """
    Count the number of fixed points in a buffer various selected points.

    Args:
        fixed_points (array-like):
            The fixed reference points.  Each column is a dimension, and
            each row is a point.  These are the points that will be counted.
        fixed_values (vector):
            The values for each of the fixed points. Length is equal to
            the number of rows in fixed_points.
        other_points (array-like):
            The candidate measurement points.  Each column is a dimension,
            and each row is a point.  The columns must exactly
            match the columns in `fixed_points`, while the number of rows
            and the content thereof can be (and probably should be)
            entirely different.  These are the center points of the
            various buffers.
        weights (vector):
            A set of weights by dimension.
            The values in this vector should correspond to the columns
            in `fixed_points` and `other_points`.
        buffer_dist (float, default 1):
            A buffer distance.

    Returns:
        pandas.DataFrame:
            The columns of the returned data correspond to the columns
            in the `weights` input, and the row correspond to the rows
            int the `df2` argument.
    """
    array1 = np.asarray(fixed_points, dtype=float)
    array2 = np.asarray(other_points, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(1, -1)
    result = np.zeros(array2.shape[0], dtype=float)

    for i in range(array2.shape[0]):
        row = array2[i, :].reshape(1, -1)
        distances = np.sqrt(((array1 - row) ** 2) * w)
        result[i] = fixed_values[distances <= buffer_dist].sum()
    return result


def minimum_weighted_distances(df1, df2, weights):
    """
    Compute minimum weighted distance from one DataFrame to another.

    Args:
        df1 (pandas.DataFrame):
            The fixed reference points.  Each column is a dimension, and
            each row is a point.
        df2 (pandas.DataFrame):
            The candidate measurement points.  Each column is a dimension,
            and each row is a point.  The columns in this DataFrame must
            exactly match the columns in `df1`, while the number of rows
            and the content thereof can be (and probably should be)
            entirely different.
        weights (pandas.DataFrame):
            A set of weights as a DataFrame.  Each column is a complete
            set of weights to use in calculating the weighted distance.
            The rows in this DataFrame should correspond to the columns
            in `df1` and `df2`.

    Returns:
        pandas.DataFrame:
            The columns of the returned data correspond to the columns
            in the `weights` input, and the row correspond to the rows
            int the `df2` argument.
    """
    array1 = np.asarray(df1, dtype=float)
    array2 = np.asarray(df2, dtype=float)

    result = pd.DataFrame(0, columns=weights.columns, index=df2.index)
    for i, idx in enumerate(df2.index):
        row = array2[i, :].reshape(1, -1)
        sq_dist_by_axis = (array1 - row) ** 2
        for name, w in weights.iteritems():
            result.loc[idx, name] = (sq_dist_by_axis / w.values).sum(1).min()
    return result


def _pick_one_new_experiment(
        existing_experiments,
        proposed_experiments,
        possible_experiments,
        dimension_weights,
        future_experiments,
        future_experiments_std,
        buffer_weighting=1,
        debug=None,
):
    """
    Pick a single new experiment from a candidate population.

    Args:
        existing_experiments (pandas.DataFrame):
            A set of existing experiments.  These experiments have
            already been run through the core model and results are
            available.  The data of this DataFrame should all be
            of a format that is or can be cast to floating point
            (i.e., no strings or categorical data).
        proposed_experiments (pandas.DataFrame):
            The set of existing experiments, plus any other experiments
            that are already proposed to be completed.
        possible_experiments (pandas.DataFrame):
            A set of possible experiments.  These experiments have
            not been run through the core model and computed full results
            are not available.  The format of this DataFrame should be
            identical to `existing_experiments` in data types and columns.
        output_weights (Mapping):
            The keys of this mapping correspond to output measures from
            the core model and the values are relative importance weights.
        distance_scales (pandas.DataFrame):
            The distance scales to use.  The rows of this DataFrame should
            correspond to the columns in the `experiments` arguments, and
            the columns to keys in the `output_weights`.  Typically, this
            argument is the inverse of the result from the
            `get_length_scales` method of a `emat.MetaModel` that has been
            fit on the existing experiment results.

    Returns:
        int
            The row number from possible_experiments that is selected.
    """
    mwd = minimum_weighted_distance(
        proposed_experiments,
        possible_experiments,
        dimension_weights
    )

    if future_experiments is not None:
        cwb = value_within_buffer(
            future_experiments,
            future_experiments_std,
            possible_experiments,
            dimension_weights,
            buffer_dist=1,
        ).astype(float)
        cwb /= cwb.max()

        if debug:
            from matplotlib import pyplot as plt
            plt.clf()
            scat = plt.scatter(possible_experiments[debug[0]], possible_experiments[debug[1]], c=mwd)
            plt.colorbar(scat)
            plt.scatter(proposed_experiments[debug[0]], proposed_experiments[debug[1]], color='red')
            plt.scatter(existing_experiments[debug[0]], existing_experiments[debug[1]], color='pink', marker='x')
            plt.title("minimum weighted distance")
            plt.show()
            plt.clf()
            scat = plt.scatter(possible_experiments[debug[0]], possible_experiments[debug[1]], c=cwb)
            plt.colorbar(scat)
            plt.scatter(proposed_experiments[debug[0]], proposed_experiments[debug[1]], color='red')
            plt.scatter(existing_experiments[debug[0]], existing_experiments[debug[1]], color='pink', marker='x')
            plt.title("count within buffer")
            plt.show()

        mwd = mwd + (buffer_weighting * cwb)

    new_candidate_experiment = mwd.argmax()
    return new_candidate_experiment


def batch_pick_new_experiments(
        existing_experiments,
        possible_experiments,
        batch_size,
        dimension_weights,
        future_experiments,
        future_experiments_std,
        buffer_weighting = 1,
        debug = None,
):
    """
    Pick a batch of new experiments from a candidate population.

    Args:
        existing_experiments (pandas.DataFrame):
            A set of existing experiments.  These experiments have
            already been run through the core model and results are
            available.  The data of this DataFrame should all be
            of a format that is or can be cast to floating point
            (i.e., no strings or categorical data).
        possible_experiments (pandas.DataFrame):
            A set of possible experiments.  These experiments have
            not been run through the core model and computed full results
            are not available.  The format of this DataFrame should be
            identical to `existing_experiments` in data types and columns.
        batch_size (int):
            How many new experiments should be selected for this batch.
        output_weights (Mapping):
            The keys of this mapping correspond to output measures from
            the core model and the values are relative importance weights.
        distance_scales (pandas.DataFrame):
            The distance scales to use.  The rows of this DataFrame should
            correspond to the columns in the `experiments` arguments, and
            the columns to keys in the `output_weights`.  Typically, this
            argument is the inverse of the result from the
            `get_length_scales` method of a `emat.MetaModel` that has been
            fit on the existing experiment results.

    Returns:
        pandas.DataFrame:
            This contains `batch_size` rows selected from
            `possible_experiments`.
    """
    proposed_experiments = existing_experiments.copy()

    # Initial selection, greedy
    for i in range(batch_size):
        new_candidate_experiment = _pick_one_new_experiment(
            existing_experiments,
            proposed_experiments,
            possible_experiments,
            dimension_weights,
            future_experiments,
            future_experiments_std,
            buffer_weighting=buffer_weighting,
            debug=debug,
        )
        proposed_experiments = proposed_experiments.append(possible_experiments.iloc[new_candidate_experiment])
        _logger.info(f"Selecting {proposed_experiments.index[-1]}")

    new_experiments = proposed_experiments.iloc[-batch_size:]

    # Fedorov Exchanges
    n_exchanges = 1
    while n_exchanges > 0:
        n_exchanges = 0
        for i in range(batch_size):
            provisionally_dropping = new_experiments.index[i]
            proposed_experiments = pd.concat([
                existing_experiments,
                new_experiments.drop(new_experiments.index[i])
            ])
            # new_candidate_experiment = minimum_weighted_distance(
            #     proposed_experiments,
            #     possible_experiments,
            #     dimension_weights
            # ).argmax()
            new_candidate_experiment = _pick_one_new_experiment(
                existing_experiments,
                proposed_experiments,
                possible_experiments,
                dimension_weights,
                future_experiments,
                future_experiments_std,
                buffer_weighting=buffer_weighting,
                debug=debug,
            )
            provisional_replacement = possible_experiments.index[new_candidate_experiment]
            if provisional_replacement != provisionally_dropping:
                n_exchanges += 1
                new_index = new_experiments.index.tolist()
                new_index[i] = provisional_replacement
                new_experiments.index = new_index
                new_experiments.iloc[i] = possible_experiments.iloc[new_candidate_experiment]
                _logger.info(f"Replacing {provisionally_dropping} with {provisional_replacement}")
        _logger.info(f"{n_exchanges} Fedorov Exchanges completed.")
    return new_experiments



def heuristic_pick_experiment(
    metamodel,
    candidate_experiments,
    poorness_of_fit,
    candidate_density,
    plot=True,
):
    candidate_std = metamodel.compute_std(candidate_experiments)
    candidate_raw_value = (poorness_of_fit * candidate_std).sum(axis=1)
    candidate_wgt_value = candidate_raw_value * candidate_density
    proposed_experiment = candidate_wgt_value.idxmax()
    if plot:
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(1,1, figsize=(4,4))
        axs.scatter(
            candidate_experiments.iloc[:,0],
            candidate_experiments.iloc[:,1],
            c=candidate_wgt_value,
        )
        axs.scatter(
            candidate_experiments.iloc[:,0].loc[proposed_experiment],
            candidate_experiments.iloc[:,1].loc[proposed_experiment],
            color="red", marker='x',
        )
        plt.show()
        plt.close(fig)
    return proposed_experiment


def heuristic_batch_pick_experiment(
        batch_size,
        metamodel,
        candidate_experiments,
        scope,
        poorness_of_fit=None,
        plot=True,
):
    _logger.info(f"Computing Density")
    candidate_density = candidate_experiments.apply(lambda x: scope.get_density(x), axis=1)

    if poorness_of_fit is None:
        _logger.info(f"Computing Poorness of Fit")
        crossval = metamodel.function.cross_val_scores()
        poorness_of_fit = dict(1 - crossval)

    proposed_candidate_ids = set()
    proposed_candidates = None

    for i in range(batch_size):
        metamodel.function.regression.set_hypothetical_training_points(proposed_candidates)
        proposed_id = heuristic_pick_experiment(
            metamodel,
            candidate_experiments,
            poorness_of_fit,
            candidate_density,
            plot=plot,
        )
        proposed_candidate_ids.add(proposed_id)
        proposed_candidates = candidate_experiments.loc[proposed_candidate_ids]

    proposed_candidate_ids = list(proposed_candidate_ids)

    # Exchanges
    n_exchanges = 1
    while n_exchanges > 0:
        n_exchanges = 0
        for i in range(batch_size):
            provisionally_dropping = proposed_candidate_ids[i]
            metamodel.function.regression.set_hypothetical_training_points(
                candidate_experiments.loc[set(proposed_candidate_ids) - {provisionally_dropping}]
            )
            provisional_replacement = heuristic_pick_experiment(
                metamodel,
                candidate_experiments,
                poorness_of_fit,
                candidate_density,
                plot=plot,
            )
            if provisional_replacement not in proposed_candidate_ids:
                n_exchanges += 1
                proposed_candidate_ids[i] = provisional_replacement
                _logger.info(f"Replacing {provisionally_dropping} with {provisional_replacement}")
        _logger.info(f"{n_exchanges} Exchanges completed.")



    metamodel.function.regression.clear_hypothetical_training_points()
    return proposed_candidates


