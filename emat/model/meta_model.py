# -*- coding: utf-8 -*-

import pandas
import numpy
import warnings
from typing import Mapping
from ..learn.base import clone_or_construct
from ..learn.boosting import LinearAndGaussian
from ..util.one_hot import OneHotCatEncoder
from ..util.variance_threshold import VarianceThreshold
from ..experiment.experimental_design import batch_pick_new_experiments, minimum_weighted_distance
from ..database.database import Database
from ..scope.scope import Scope
from ..exceptions import ReadOnlyDatabaseError

from ..util.loggers import get_module_logger
_logger = get_module_logger(__name__)


def create_metamodel(
        scope,
        experiments=None,
        metamodel_id=None,
        db=None,
        include_measures=None,
        exclude_measures=None,
        random_state=None,
        experiment_stratification=None,
        suppress_converge_warnings=False,
        regressor=None,
        name=None,
        design_name=None,
        find_best_metamodeltype=False,
):
    """
    Create a MetaModel from a set of input and output observations.

    Args:
        scope (emat.Scope): The scope for this model.
        experiments (pandas.DataFrame): This dataframe
            should contain all of the experimental inputs and outputs,
            including values for each uncertainty, level, constant, and
            performance measure.
        metamodel_id (int, optional): An identifier for this meta-model.
            If not given, a unique id number will be created randomly
            (if not `db` is given) or sequentially based on any existing
            metamodels already stored in the database.
        db (Database, optional): The database to use for loading and
            saving metamodels. If none is given here, the metamodel will
            not be stored in a database otherwise, the metamodel is
            automatically saved to the database after it is created.
        include_measures (Collection[str], optional): If provided, only
            output performance measures with names in this set will be included.
        exclude_measures (Collection[str], optional): If provided, only
            output performance measures with names not in this set will be included.
        random_state (int, optional): A random state to use in the metamodel
            regression fitting.
        experiment_stratification (pandas.Series, optional):
            A stratification of experiments, used in cross-validation.
        suppress_converge_warnings (bool, default False):
            Suppress convergence warnings during metamodel fitting.
        regressor (Estimator, optional): A scikit-learn estimator implementing a
            multi-target regression.  If not given, a detrended simple Gaussian
            process regression is used.
        name (str, optional): A descriptive name for this metamodel.
        design_name (str, optional): The name of the design of experiments
            from `db` to use to create the metamodel. Only used if `experiments`
            is not given explicitly.
        find_best_metamodeltype (int, default 0):
            Run a search to find the best metamodeltype for each
            performance measure, repeating each cross-validation
            step this many times.  For more stable results, choose
            3 or more, although larger numbers will be slow.  If
            domain knowledge about the normal expected range and
            behavior of each performance measure is available,
            it is better to give the metamodeltype explicitly in
            the Scope.

    Returns:
        PythonCoreModel:
            a callable object that, when called as if a
            function, accepts keyword arguments as inputs and
            returns a dictionary of (measure name: value) pairs.
    """
    _logger.info("creating metamodel from data")

    from .core_python import PythonCoreModel

    if experiments is None:
        if design_name is None or db is None:
            raise ValueError('must give `experiments` as a DataFrame or both `db` and `design_name`')
        experiments = db.read_experiment_all(scope.name, design_name, only_with_measures=True)

    experiments = scope.ensure_dtypes(experiments)
    _actually_excluded_measures = []

    meas = []
    for j in scope.get_measure_names():
        if include_measures is not None and j not in include_measures:
            _actually_excluded_measures.append(j)
            continue
        if exclude_measures is not None and j in exclude_measures:
            _actually_excluded_measures.append(j)
            continue
        if j not in experiments:
            _actually_excluded_measures.append(j)
            continue
        j_na_count = experiments[j].isna().sum()
        if j_na_count == len(experiments[j]):
            warnings.warn(f"measure '{j}' is all missing data, excluding it from the metamodel")
            _actually_excluded_measures.append(j)
            continue
        if j_na_count:
            warnings.warn(f"measure '{j}' has some missing data, excluding it from the metamodel")
            _actually_excluded_measures.append(j)
            continue # TODO: allow for development of meta-models with the non-missing parts
        meas.append(j)
    experiment_outputs = experiments[meas]

    params = []
    for j in scope.get_parameter_names():
        if j not in experiments:
            continue
        params.append(j)
    experiment_inputs = experiments[params]

    if metamodel_id is None:
        if db is not None:
            metamodel_id = db.get_new_metamodel_id(scope.name)
    if metamodel_id is None:
        metamodel_id = numpy.random.randint(1<<32, 1<<63, dtype='int64')

    if find_best_metamodeltype:
        output_transforms, metamodeltype_tabulation = select_best_metamodeltype(
            experiment_inputs, experiment_outputs, return_tabulation=True
        )
        output_transforms = dict(output_transforms)
    else:
        output_transforms = {
            i.name: i.metamodeltype
            for i in scope.get_measures()
            if i.name in meas
        }
        metamodeltype_tabulation = None

    # change log tranforms to log1p when the experimental minimum is
    # non-positive but not less than -1
    for i, i_transform in output_transforms.items():
        if i_transform == 'log' and i in experiment_outputs:
            i_min = experiment_outputs[i].min()
            if -1 < i_min <= 0:
                output_transforms[i] = 'log1p'

    disabled_outputs = [i for i in scope.get_measure_names()
                        if i not in experiment_outputs.columns]

    func = MetaModel(
        experiment_inputs,
        experiment_outputs,
        metamodel_types=output_transforms,
        disabled_outputs=disabled_outputs,
        random_state=random_state,
        sample_stratification=experiment_stratification,
        suppress_converge_warnings=suppress_converge_warnings,
        regressor=regressor,
    )

    if metamodeltype_tabulation is not None:
        func.metamodeltype_tabulation = metamodeltype_tabulation

    scope_ = scope.duplicate(
        strip_measure_transforms=True,
        include_measures=include_measures,
        exclude_measures=_actually_excluded_measures,
    )

    result = PythonCoreModel(
        func,
        configuration=None,
        scope=scope_,
        safe=True,
        db=db,
        name=name or f"MetaModel{metamodel_id}",
        metamodel_id=metamodel_id,
    )

    if db is not None:
        try:
            db.write_metamodel(result)
        except ReadOnlyDatabaseError:
            pass # read only database, don't store
        except Exception as err:
            _logger.exception("exception in storing metamodel in database")

    return result


class MetaModel:
    """
    A gaussian process regression-based meta-model.

    The MetaModel is a callable object that provides an EMA Workbench standard
    python interface, taking keyword arguments for parameters and returning a python
    dictionary of named outcomes.

    Args:
        input_sample (pandas.DataFrame): A set of experimental parameters, where
            each row in the dataframe is an experiment that has already been evaluated
            using the core model.  Each column will be a required keyword parameter
            when calling this meta-model.
        output_sample (pandas.DataFrame): A set of experimental performance measures, where
            each row in the dataframe is the results of the experiment evaluated using the
            core model.  Each column of this dataframe will be a named output value
            in the returned dictionary from calling this meta-model.
        metamodel_types (Mapping, optional): If given, the keys of this mapping should
            include a subset of the columns in `output_sample`, and the values indicate
            the metamodel type for each performance measure, given as `str`.  Available
            metamodel types include:

            + *log*: The natural log of the performance measure is taken before
              fitting the regression model.  This is appropriate only when the performance
              measure will always give a strictly positive outcome. If the performance
              measure can take on non-positive values, this may result in errors.

            + *log1p*: The natural log of 1 plus the performance measure is taken before
              fitting the regression model.  This is preferred to log-linear when the
              performance measure is only guaranteed to be non-negative, rather than
              strictly positive.

            + *logxp(X)*: The natural log of X plus the performance measure is taken before
              fitting the regression model.  This allows shifting the position of the
              regression intercept to a point other than 0.

            + *clip(LO,HI)*: A linear model is used, but results are truncated to the range
              (LO,HI). Set either value as None to have a one-sided truncation range.

            + *linear*: No transforms are made.  This is the default when a performance
              measure is not included in `metamodel_types`.

        disabled_outputs (Collection, optional): A collection of disabled outputs. All names
            included in this collection will be returned in the resulting outcomes dictionary
            when this meta-model is evaluated, but with a value of `None`.  It is valid to
            include names in `disabled_outputs` that are included in the columns
            of `output_sample`, although the principal use of this argument is to include
            names that are *not* included, as disabling outputs that are included will not
            prevent these values from being included in the computational process.

        random_state (int, optional): A random state, passed to the created regression
            (but only if that regressor includes a 'random_state' parameter).

        regressor (Estimator, optional): A scikit-learn estimator implementing a
            multi-target regression.  If not given, a detrended simple Gaussian
            process regression is used.
    """

    _metamodel_types = {
        'log': (numpy.log, numpy.exp),
        'log-linear': (numpy.log, numpy.exp),
        'ln': (numpy.log, numpy.exp),
        'log1p': (numpy.log1p, numpy.expm1),
        'log1p-linear': (numpy.log1p, numpy.expm1),
        'exp': (numpy.exp, numpy.log),
        'logit': (lambda x: numpy.log(x/(1-x)), lambda x: numpy.exp(x)/(numpy.exp(x)+1)),
        # x is applied immediate from arguments in metamodeltype string, y is the eventual data
        'logxp': (lambda x: (lambda y: numpy.log(y + x)), lambda x: (lambda y: numpy.exp(y) - x)),
        'logxp-linear': (lambda x: (lambda y: numpy.log(y + x)), lambda x: (lambda y: numpy.exp(y) - x)),
        'clip': (lambda x: (lambda y: y), lambda x: (lambda y: numpy.clip(y, *x))),
    }

    def __init__(
            self,
            input_sample,
            output_sample,
            metamodel_types=None,
            disabled_outputs=None,
            random_state=None,
            sample_stratification=None,
            suppress_converge_warnings=False,
            regressor=None,
            use_best_cv=True,
    ):

        if not isinstance(input_sample, pandas.DataFrame):
            raise TypeError('input_sample must be DataFrame')

        if not isinstance(output_sample, pandas.DataFrame):
            raise TypeError('output_sample must be DataFrame')

        self.raw_input_columns = input_sample.columns

        self.disabled_outputs = disabled_outputs

        # One-hot encode here and save the mapping
        self.cat_encoder = OneHotCatEncoder().fit(input_sample)
        input_sample = self.cat_encoder.transform(input_sample)
        input_sample = input_sample.astype(numpy.float64)

        self.var_thresh = VarianceThreshold().fit(input_sample)
        input_sample = self.var_thresh.transform(input_sample)

        self.input_sample = input_sample
        self.output_sample = output_sample.copy(deep=(metamodel_types is not None)).astype(float)
        self.sample_stratification = sample_stratification

        self.output_transforms = {}
        if metamodel_types is not None:
            self.metamodel_types = metamodel_types
            for k,t in metamodel_types.items():
                if t is None:
                    continue
                if "(" in t:
                    t, t_args = t.split("(", 1)
                    import ast
                    t_args = ast.literal_eval(t_args.strip("()"))
                    if isinstance(t, str):
                        t = t.lower()
                    if t == 'linear':
                        continue
                    if t not in self._metamodel_types:
                        raise ValueError(f'unknown metamodeltype "{t}" for output "{k}"')
                    self.output_transforms[k] = (
                        self._metamodel_types[t][0](t_args),
                        self._metamodel_types[t][1](t_args)
                    )
                else:
                    if isinstance(t, str):
                        t = t.lower()
                    if t == 'linear':
                        continue
                    if t not in self._metamodel_types:
                        raise ValueError(f'unknown metamodeltype "{t}" for output "{k}"')
                    self.output_transforms[k] = self._metamodel_types[t]

        for k, (v_func,_) in self.output_transforms.items():
            self.output_sample[k] = v_func(self.output_sample[k])

        if regressor is None:
            regressor = LinearAndGaussian()

        self.regression = clone_or_construct(regressor)

        if random_state is not None and 'random_state' in self.regression.get_params():
            self.regression.set_params(random_state=random_state)

        if suppress_converge_warnings:
            from sklearn.exceptions import ConvergenceWarning
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.regression.fit(self.input_sample, self.output_sample)
        else:
            self.regression.fit(self.input_sample, self.output_sample)

        if use_best_cv:
            from ..learn.model_selection import take_best
            self.regression = take_best(self.regression)

    def preprocess_raw_input(self, df, to_type=None):
        """
        Preprocess raw data input.

        This convenience method provides batch-processing of a
        raw data input DataFrame into the format used for regression.

        Args:
            df (pandas.DataFrame):
                The raw input data to process, which can include input
                values for multiple experiments.
            to_type (dtype, optional):
                If given, the entire resulting DataFrame is cast to
                this data type.

        Returns:
            pandas.DataFrame
        """
        result = self.cat_encoder.transform(df[self.raw_input_columns])
        if to_type is not None:
            result = result.astype(to_type)
        result = self.var_thresh.transform(result)
        return result

    def __call__(self, *args, **kwargs):
        """
        Evaluate the meta-model.

        Args:
            **kwargs:
                All defined (meta)model parameters are passed as keyword
                arguments, including both uncertainties and levers.

        Returns:
            dict:
                A single dictionary containing all performance measure outcomes.
        """
        if len(args) == 1:
            if isinstance(args[0], pandas.DataFrame):
                return args[0].apply(
                    lambda x: pandas.Series(self.__call__(**x)),
                    axis=1,
                )
            else:
                raise TypeError(f'mm(...) optionally takes a DataFrame as a '
                                f'positional argument, not {type(args[0])}')
        elif len(args) > 1:
            raise TypeError(f'mm(...) takes at most one '
                            f'positional argument, not {len(args)}')

        input_row = pandas.DataFrame.from_dict(kwargs, orient='index').T[self.raw_input_columns]
        input_row = self.preprocess_raw_input(input_row, to_type=numpy.float)

        output_row = self.regression.predict(input_row)
        result = dict(output_row.iloc[0])

        # undo the output transforms
        for k, (_,v_func) in self.output_transforms.items():
            result[k] = v_func(result[k])

        for i in self.disabled_outputs:
            result[i] = None

        return result

    def compute_std(self, *args, **kwargs):
        """
        Evaluate standard deviations of estimates generated by the meta-model.

        Args:
            df (pandas.DataFrame, optional)
            **kwargs:
                All defined (meta)model parameters are passed as keyword
                arguments, including both uncertainties and levers.

        Returns:
            dict:
                A single dictionary containing the standard deviaition of the
                estimate of all performance measure outcomes.
        """
        if len(args) == 1:
            if isinstance(args[0], pandas.DataFrame):
                return args[0].apply(
                    lambda x: pandas.Series(self.compute_std(**x)),
                    axis=1,
                )
            else:
                raise TypeError(f'compute_std() optionally takes a DataFrame as a '
                                f'positional argument, not {type(args[0])}')
        elif len(args) > 1:
            raise TypeError(f'compute_std() takes at most one '
                            f'positional argument, not {len(args)}')

        input_row = pandas.DataFrame.from_dict(kwargs, orient='index').T[self.raw_input_columns]
        input_row = self.preprocess_raw_input(input_row, to_type=numpy.float)

        output_row, output_std = self.regression.predict(input_row, return_std=True)

        result = dict(output_std.iloc[0])

        # DO NOT undo the output transforms
        # for k, (_,v_func) in self.output_transforms.items():
        #     result[k] = v_func(result[k])

        for i in self.disabled_outputs:
            result[i] = None

        return result

    def predict(self, *args, trend_only=False, residual_only=False, **kwargs):
        """
        Generate predictions using the meta-model.

        Args:
            df (pandas.DataFrame, optional)
            trend_only, residual_only (bool)
            **kwargs:
                All defined (meta)model parameters are passed as keyword
                arguments, including both uncertainties and levers.

        Returns:
            dict:
                A single dictionary containing the standard deviaition of the
                estimate of all performance measure outcomes.
        """
        if len(args) == 1:
            if isinstance(args[0], pandas.DataFrame):

                input_row = args[0][self.raw_input_columns]
                input_row = self.preprocess_raw_input(input_row, to_type=numpy.float64)

                if trend_only:
                    result = self.regression.lr.predict(input_row)
                elif residual_only:
                    result = self.regression.gpr.predict(input_row)
                else:
                    result = self.regression.predict(input_row)

                # undo the output transforms
                for k, (_, v_func) in self.output_transforms.items():
                    result[k] = v_func(result[k])

                if self.disabled_outputs:
                    drop_cols = [i for i in self.disabled_outputs if i in result.columns]
                    result = result.drop(drop_cols, axis=1)

                return result

            else:
                raise TypeError(f'predict() optionally takes a DataFrame as a '
                                f'positional argument, not {type(args[0])}')
        elif len(args) > 1:
            raise TypeError(f'predict() takes at most one '
                            f'positional argument, not {len(args)}')

        input_row = pandas.DataFrame.from_dict(kwargs, orient='index').T[self.raw_input_columns]
        input_row = self.preprocess_raw_input(input_row, to_type=numpy.float64)

        if trend_only:
            output_row = self.regression.detrend_predict(input_row)
        elif residual_only:
            output_row = self.regression.residual_predict(input_row)
        else:
            output_row = self.regression.predict(input_row)

        result = dict(output_row.iloc[0])

        # undo the output transforms
        for k, (_,v_func) in self.output_transforms.items():
            result[k] = v_func(result[k])

        for i in self.disabled_outputs:
            result[i] = None

        return result


    def cross_val_scores(
            self,
            cv=5,
            gpr_only=False,
            use_cache=True,
            return_type='styled',
            shortnames=None,
            suppress_converge_warnings=False,
            **kwargs,
    ):
        """
        Calculate the cross validation scores for this meta-model.

        Args:
            cv (int, default 5): The number of folds to use in
                cross-validation.
            gpr_only (bool, default False): Whether to limit the
                cross-validation analysis to only the GPR step (i.e.,
                to measure the improvement in meta-model fit from
                using the GPR-based meta-model, over and above
                using the linear regression meta-model alone.)
            use_cache (bool, default True): Use cached cross
                validation results if available.  All other arguments
                are ignored if a cached results if available.
            return_type ({'styled', 'raw'}):  How to return the
                results.
            shortnames (Scope or callable):
                If given, use this function to convert the measure
                names into more readable `shortname` values from the
                scope, or by using a function that maps measures
                names to something else.

        Returns:
            pandas.Series: The cross-validation scores, by output.

        """
        result = None
        if use_cache:
            cached_value = getattr(self, '_cv_cache', None)
            if cached_value is not None:
                result = cached_value

        if result is None:
            if self.sample_stratification is not None:
                from ..learn.splits import ExogenouslyStratifiedKFold
                cv = ExogenouslyStratifiedKFold(exo_data=self.sample_stratification, n_splits=cv)

            if gpr_only:
                raise NotImplementedError
                # residuals = self.regression.residual_predict(self.input_sample)
                # regression = multitarget.MultipleTargetRegression()
                # return regression.cross_val_scores(self.input_sample, residuals, cv=cv)

            if suppress_converge_warnings:
                from sklearn.exceptions import ConvergenceWarning
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    result = self.regression.cross_val_scores(self.input_sample, self.output_sample, cv=cv, **kwargs)
            else:
                result = self.regression.cross_val_scores(self.input_sample, self.output_sample, cv=cv, **kwargs)
            result.name = "Cross Validation Score"

        if use_cache:
            self._cv_cache = result

        if shortnames is not None:
            if isinstance(shortnames, Scope):
                result.index = result.index.map(shortnames.shortname)
            else:
                result.index = result.index.map(shortnames)

        if return_type == 'styled':
            from ..util.styling import cross_validation_styling
            return cross_validation_styling(result)

        return result

    def cross_val_predicts(self, cv=5):
        """
        Generate cross validated predictions using this meta-model.

        Args:
            cv (int, default 5): The number of folds to use in
                cross-validation. Set to zero for leave-one-out
                (i.e., the maximum number of folds), which may be
                quite slow.

        Returns:
            pandas.DataFrame: The cross-validated predictions.

        """
        if cv==0:
            cv = len(self.input_sample)
        return self.regression.cross_val_predict(self.input_sample, self.output_sample, cv=cv)


    def __repr__(self):
        in_dim = len(self.raw_input_columns)
        out_dim = len(self.output_sample.columns)
        if self.disabled_outputs:
            out_dims = f"{out_dim} active and {out_dim + len(self.disabled_outputs)} total outputs"
        else:
            out_dims = f"{out_dim} outputs"
        return f"<emat.MetaModel {in_dim} inputs -> {out_dims}>"

    def get_length_scales(self):
        """
        Get the length scales from the GPR kernels of this metamodel.

        This MetaModel must already be `fit` to use this method, although
        the fit process is generally completed when the MetaModel is
        instantiated.

        Returns:
            pandas.DataFrame:
                The columns correspond to the columns of pre-processed
                input (not raw input) and the rows correspond to the
                outputs.
        """
        return pandas.DataFrame(
            [
                est.kernel_.length_scale
                for est in self.regression.step1.estimators_
            ],
            index=self.regression.Y_columns,
            columns=self.input_sample.columns,
        ).T

    def mix_length_scales(self, balance=None, inv=True):
        """
        Mix the length scales from the GPR kernels of this metamodel.

        This MetaModel must already be `fit` to use this method, although
        the fit process is generally completed when the MetaModel is
        instantiated.

        Args:
            balance (Mapping or Collection, optional):
                When given as a mapping, the keys are the output measures
                that are included in the mix, and the values are the
                relative weights to use for mixing.
                When given as a collection, the items are the output
                measures that are included in the mix, all with equal
                weight.
            inv (bool, default True):
                Take the inverse of the length scales before mixing.

        Returns:
            ndarray:
                The columns correspond to the columns of pre-processed
                input (not raw input) and the rows correspond to the
                outputs.
        """
        s = self.get_length_scales()
        if inv:
            s = s.rtruediv(1, fill_value=1)  # s = 1/s
        if balance is None:
            w = numpy.full(len(s.columns), 1.0/len(s.columns))
        elif isinstance(balance, Mapping):
            w = numpy.zeros(len(s.columns))
            for i,col in enumerate(s.columns):
                w[i] = balance.get(col, 0)
        else:
            w = numpy.zeros(len(s.columns))
            balance = set(balance)
            each_w = 1/len(balance)
            for i,col in enumerate(s.columns):
                w[i] = each_w if col in balance else 0
        return numpy.dot(s, w)

    def pick_new_experiments(
            self,
            possible_experiments,
            batch_size,
            output_focus=None,
            scope: Scope=None,
            db: Database=None,
            design_name: str=None,
            debug=None,
            future_experiments=None,
            future_experiments_std=None,
    ):
        """
        Select a set of new experiments to perform from a pool of candidates.

        This method implements the "maximin" approach described by Johnson et al (1990),
        as proposed for batch-sequential augmentation of designs by Loeppky et al (2010).
        New experiments are selected from a pool of possible new experiments by
        maximizing the minimum distance between the set of selected experiments,
        with distances between experiments scaled by the correlation parameters
        from a GP regression fitted to the initial experimental results. Note that
        the "binning" aspect of Loeppky is not presently implemented here,
        instead favoring the analyst's capability to manually focus the new experiments
        by manipulating the input `possible_experiments`.

        We also extend Loeppky et al by allowing for multiple output models, mixing the
        results from a selected set of outputs, to potentially focus the information
        from the new experiments on a subset of output measures.

        Args:
            possible_experiments:
                A pool of possible experiments.  All selected experiments will
                be selected from this pool, so the pool should be sufficiently
                large and diverse to provide requried support for this process.
            batch_size (int):
                How many experiments to select from `possible_experiments`.
            output_focus (Mapping or Collection, optional):
                 A subset of output measures that will be the focus of these new
                 experiments. The length scales of these measures will be mixed
                 when developing relative weights.
            scope (Scope, optional): The exploratory scope to use for writing the
                design to a database. Ignored unless `db` is also given.
            db (Database, optional): If provided, this design will be stored in the
                database indicated.  Ignored unless `scope` is also given.
            design_name (str, optional): A name for this design, to identify it in the
                database. If not given, a unique name will be generated.  Has no effect
                if no `db` or `scope` is given.
            debug (Tuple[str,str], optional): The names of x and y axis to plot for
                debugging.

        Returns:
            pandas.DataFrame:
                A subset of rows from `possible_experiments`

        References:
            - Johnson, M.E., Moore, L.M., and Ylvisaker, D., 1990. "Minimax and maximin
              distance designs." Journal of Statistical Planning and Inference 26, 131–148.
            - Loeppky, J., Moore, L., and Williams, B.J., 2010. "Batch sequential designs
              for computer experiments." Journal of Statistical Planning and Inference 140,
              1452–1464.

        """

        dimension_weights = self.mix_length_scales(output_focus, inv=True)
        if debug:
            _logger.info(f"output_focus = {output_focus}")
            _logger.info(f"length_scales =\n{self.get_length_scales()}")
            _logger.info(f"dimension_weights = {dimension_weights}")

        possible_experiments_processed = self.preprocess_raw_input(possible_experiments, float)

        picks = batch_pick_new_experiments(
                self.input_sample,
                possible_experiments_processed,
                batch_size,
                dimension_weights,
                future_experiments,
                future_experiments_std,
                debug=debug,
        )

        design = possible_experiments.loc[picks.index]

        # If using the default design_name, append the design_name with a number
        # until a new unused name is found.
        if db is not None and scope is not None and design_name is None:
            proposed_design_name = 'augment'
            existing_design_names = set(db.read_design_names(scope.name))
            if proposed_design_name not in existing_design_names:
                design_name = proposed_design_name
            else:
                n = 2
                while f'{proposed_design_name}_{n}' in existing_design_names:
                    n += 1
                design_name = f'{proposed_design_name}_{n}'

        if db is not None and scope is not None:
            experiment_ids = db.write_experiment_parameters(scope.name, design_name, design)
            design.index = experiment_ids
            design.index.name = 'experiment'

        if debug:
            debug_x, debug_y = debug
            mwd = minimum_weighted_distance(
                self.input_sample,
                possible_experiments,
                dimension_weights
            )

            from matplotlib import pyplot as plt
            plt.clf()
            plt.scatter(possible_experiments[debug_x], possible_experiments[debug_y], c=mwd)
            plt.scatter(self.input_sample[debug_x], self.input_sample[debug_y], color='red')
            plt.scatter(design[debug_x], design[debug_y], color="red", marker='x')
            plt.show()

        return design

    def heuristic_pick_experiment(
            self,
            candidate_experiments,
            poorness_of_fit,
            candidate_density,
            plot=True,
    ):
        candidate_std = self.compute_std(candidate_experiments)
        candidate_raw_value = (poorness_of_fit * candidate_std).sum(axis=1)
        candidate_wgt_value = candidate_raw_value * candidate_density
        proposed_experiment = candidate_wgt_value.idxmax()
        if plot:
            from matplotlib import pyplot as plt
            fig, axs = plt.subplots(1, 1, figsize=(4, 4))
            axs.scatter(
                candidate_experiments.iloc[:, 0],
                candidate_experiments.iloc[:, 1],
                c=candidate_wgt_value,
            )
            axs.scatter(
                candidate_experiments.iloc[:, 0].loc[proposed_experiment],
                candidate_experiments.iloc[:, 1].loc[proposed_experiment],
                color="red", marker='x',
            )
            plt.show()
            plt.close(fig)
        return proposed_experiment

    def heuristic_batch_pick_experiment(
            self,
            batch_size,
            candidate_experiments,
            scope,
            poorness_of_fit=None,
            plot=True,
    ):
        _logger.info(f"computing density")
        candidate_density = candidate_experiments.apply(lambda x: scope.get_density(x), axis=1)

        if poorness_of_fit is None:
            _logger.info(f"computing poorness of fit")
            crossval = self.cross_val_scores()
            poorness_of_fit = dict(1 - crossval)

        proposed_candidate_ids = set()
        proposed_candidates = None

        _logger.info(f"populating initial batch")
        for i in range(batch_size):
            self.regression.set_hypothetical_training_points(proposed_candidates)
            proposed_id = self.heuristic_pick_experiment(
                candidate_experiments,
                poorness_of_fit,
                candidate_density,
                plot=plot,
            )
            proposed_candidate_ids.add(proposed_id)
            proposed_candidates = candidate_experiments.loc[proposed_candidate_ids]

        proposed_candidate_ids = list(proposed_candidate_ids)

        _logger.info(f"initial batch complete, checking for exchanges")
        # Exchanges
        n_exchanges = 1
        while n_exchanges > 0:
            n_exchanges = 0
            for i in range(batch_size):
                provisionally_dropping = proposed_candidate_ids[i]
                self.regression.set_hypothetical_training_points(
                    candidate_experiments.loc[set(proposed_candidate_ids) - {provisionally_dropping}]
                )
                provisional_replacement = self.heuristic_pick_experiment(
                    candidate_experiments,
                    poorness_of_fit,
                    candidate_density,
                    plot=plot,
                )
                if provisional_replacement not in proposed_candidate_ids:
                    n_exchanges += 1
                    proposed_candidate_ids[i] = provisional_replacement
                    _logger.info(f"replacing {provisionally_dropping} with {provisional_replacement}")
            _logger.info(f"{n_exchanges} exchanges completed.")

        self.regression.clear_hypothetical_training_points()
        return proposed_candidates


def select_best_metamodeltype(
        params,
        measures,
        random_state=0,
        suppress_converge_warnings=True,
        possible_types=None,
        n_repeats=3,
        regressor=None,
        return_tabulation=False,
):
    if possible_types is None:
        possible_types = {'linear', 'log', 'log1p', 'logit', 'exp'}

    if regressor is None:
        from ..learn.boosting import LinearAndGaussian
        regressor = LinearAndGaussian

    def _metamodel_scores(t, filter_cols):
        if t in possible_types and len(filter_cols):
            result = MetaModel(
                params,
                measures[filter_cols],
                {
                    i: t
                    for i in filter_cols
                },
                disabled_outputs=None,
                random_state=random_state,
                suppress_converge_warnings=suppress_converge_warnings,
                regressor=LinearAndGaussian,
            ).cross_val_scores(
                random_state=random_state,
                n_repeats=n_repeats,
            ).data
            result.columns = [t]
            return result
        else:
            return pandas.Series(-2.0, index=filter_cols, name=t)

    scores = [pandas.Series(-1.0, index=measures.columns, name='linear')]
    scores.append(_metamodel_scores('linear', measures.columns))

    check_log = measures.columns[(measures.min() > 0)]
    scores.append(_metamodel_scores('log', check_log))

    check_log1p = measures.columns[(measures.min() > -1)]
    scores.append(_metamodel_scores('log1p', check_log1p))

    check_logit = measures.columns[(measures.min() > 0) & (measures.max() < 1)]
    scores.append(_metamodel_scores('logit', check_logit))

    check_exp = measures.columns[(measures.max() < 10)]
    scores.append(_metamodel_scores('exp', check_exp))

    tabulation = pandas.concat(scores, axis=1)
    if return_tabulation:
        return tabulation.idxmax(axis=1), tabulation
    else:
        return tabulation.idxmax(axis=1)