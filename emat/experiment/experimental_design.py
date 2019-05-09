# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd

from ..scope.scope import Scope
from ..database.database import Database

from .samplers import (
    LHSSampler,
    AbstractSampler,
    UniformLHSSampler,
    MonteCarloSampler,
    CorrelatedLHSSampler,
)

samplers = {
    'lhs': CorrelatedLHSSampler,
    'ulhs': UniformLHSSampler,
    'mc': MonteCarloSampler,
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

