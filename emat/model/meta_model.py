# -*- coding: utf-8 -*-

import pandas
import numpy
from .. import multitarget
from ..util.one_hot import OneHotCatEncoder


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

            + *linear*: No transforms are made.  This is the default when a performance
              measure is not included in `metamodel_types`.

        disabled_outputs (Collection, optional): A collection of disabled outputs. All names
            included in this collection will be returned in the resulting outcomes dictionary
            when this meta-model is evaluated, but with a value of `None`.  It is valid to
            include names in `disabled_outputs` that are included in the columns
            of `output_sample`, although the principal use of this argument is to include
            names that are *not* included, as disabling outputs that are included will not
            prevent these values from being included in the computational process.

        random_state (int, optional): A random state, passed to the created regression.
    """

    _metamodel_types = {
        'log': (numpy.log, numpy.exp),
        'log-linear': (numpy.log, numpy.exp),
        'ln': (numpy.log, numpy.exp),
        'log1p': (numpy.log1p, numpy.expm1),
        'log1p-linear': (numpy.log1p, numpy.expm1),

        'logxp': (lambda x: (lambda y: numpy.log(y + x)), lambda x: (lambda y: numpy.exp(y) - x)),
        'logxp-linear': (lambda x: (lambda y: numpy.log(y + x)), lambda x: (lambda y: numpy.exp(y) - x)),
    }

    def __init__(
            self,
            input_sample,
            output_sample,
            metamodel_types=None,
            disabled_outputs=None,
            random_state=None
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

        self.input_sample = input_sample
        self.output_sample = output_sample.copy(deep=(metamodel_types is not None))

        self.output_transforms = {}
        if metamodel_types is not None:
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

        self.regression = multitarget.DetrendedMultipleTargetRegression(random_state=random_state)
        self.regression.fit(self.input_sample, self.output_sample)

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
        return result

    def __call__(self, **kwargs):
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
        input_row = pandas.DataFrame.from_dict(kwargs, orient='index').T[self.raw_input_columns]
        input_row = self.cat_encoder.transform(input_row)

        output_row = self.regression.predict(input_row)
        result = dict(output_row.iloc[0])

        # undo the output transforms
        for k, (_,v_func) in self.output_transforms.items():
            result[k] = v_func(result[k])

        for i in self.disabled_outputs:
            result[i] = None

        return result

    def compute_std(self, **kwargs):
        """
        Evaluate standard deviations of estimates generated by the meta-model.

        Args:
            **kwargs:
                All defined (meta)model parameters are passed as keyword
                arguments, including both uncertainties and levers.

        Returns:
            dict:
                A single dictionary containing the standard deviaition of the
                estimate of all performance measure outcomes.
        """
        input_row = pandas.DataFrame.from_dict(kwargs, orient='index').T[self.raw_input_columns]
        input_row = self.cat_encoder.transform(input_row)

        output_row, output_std = self.regression.predict(input_row, return_std=True)

        result = dict(output_std.iloc[0])

        # DO NOT undo the output transforms
        # for k, (_,v_func) in self.output_transforms.items():
        #     result[k] = v_func(result[k])

        for i in self.disabled_outputs:
            result[i] = None

        return result


    def cross_val_scores(self, cv=5, gpr_only=False):
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

        Returns:
            pandas.Series: The cross-validation scores, by output.

        """
        if gpr_only:
            residuals = self.regression.residual_predict(self.input_sample)
            regression = multitarget.MultipleTargetRegression()
            return regression.cross_val_scores(self.input_sample, residuals, cv=cv)
        return self.regression.cross_val_scores(self.input_sample, self.output_sample, cv=cv)

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

