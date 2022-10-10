# -*- coding: utf-8 -*-

import numpy
import re
from ..workbench import ScalarOutcome
from .names import ShortnameMixin, TaggableMixin

class Measure(ScalarOutcome, ShortnameMixin, TaggableMixin):
    '''
    Measure represents an outcome measure of the model.

    Args:
        name (str): Name of the measure.
        kind (str or int, optional): one of {'info', 'minimize', 'maximize'},
            defaults to 'info' if not given. This represents the
            generally preferred direction of the measure.
        min (number, optional): An expected minimum value that
            might be observed under any policy and scenario.  This
            value is currently only used by the HyperVolume convergence
            metric.
        max (number, optional): An expected maximum value that
            might be observed under any policy and scenario.  This
            value is currently only used by the HyperVolume convergence
            metric.
        address (obj, optional): The address or instruction for how to
            extract this measure from the model. This is model-specific
            and can potentially be any Python object. For example, if the
            model is an Excel model, this can be a cell reference given
            as a `str`.
        dtype ({'real','int','bool','cat'}, default 'real'): The desired
            dtype to be enforced for this measure.
        function (callable, optional): A callable function that will be
            used to transform the raw measure as returned by a core model.
            This transformation will be applied to core model results by
            the evaluator before they are returned to the user or stored
            in a database. It is recommended that EMAT analysis work with the
            original untransformed raw values, and employ the `metamodeltype`
            functionality as required for non-linear responses.
        transform (str, optional): As an alternative to passing a callable
            object, use this argument to pass the name of a `numpy` function.
            This argument is ignored if `function` is given.
        variable_name (str, optional): The name of the raw measure as
            output by the underlying core model. If not given, this name is
            assumed to be the same as `name`.  If no `transform` is set,
            it is strongly recommended to not give this argument either.
            A principal use of this argument is to descriptively rename
            measures that have been transformed, for example if the raw
            output measure is 'Total VMT' and a log transform function is
            applied, the result can be more descriptively renamed
            as 'log(Total VMT)'.
        metamodeltype (str, optional): The transformation type to use for
            metamodel estimation.  This transformation is applied only
            internally within the metamodel, and all inputs and outputs
            passed to or from the metamodel will not appear in a transformed
            state, including measure values stored within the database.
            Available metamodel types include:

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

            + *linear*: No transforms are made.  This is the default.


    Attributes:
        name (str): Name of the measure.
        kind (int): {MINIMIZE, MAXIMIZE, INFO, TEMP}
        transform (str): The name of the transform function, if any.
        address (obj): The address or instruction for how to
            extract this measure from the model.
        metamodeltype (str): The transformation type to use for
            metamodel estimation.
    '''

    def __init__(
            self,
            name,
            kind=ScalarOutcome.INFO,
            min=None,
            max=None,
            address=None,
            dtype=None,
            function=None,
            transform=None,
            variable_name=None,
            metamodeltype=None,
            shortname=None,
            desc=None,
            formula=None,
            tags=None,
            parser=None,
            meta=None,
    ):

        if isinstance(kind, str):
            if kind.lower()=='minimize':
                kind = ScalarOutcome.MINIMIZE
            elif kind.lower()=='maximize':
                kind = ScalarOutcome.MAXIMIZE
            elif kind.lower() == 'info':
                kind = ScalarOutcome.INFO
            elif kind.lower() == 'temp':
                kind = ScalarOutcome.TEMP
            else:
                raise TypeError(f'invalid kind {kind}')

        if transform is None:
            func = function
            if function is not None:
                transform = re.sub(' at 0x[0-9a-fA-F]*', '', f'f:{function}')
        elif isinstance(transform, str) and hasattr(numpy, transform):
            func = getattr(numpy, transform)
        elif isinstance(transform, str) and transform.lower() in ('none',):
            func = None
        else:
            raise TypeError(f'invalid transform {transform}')

        if min is not None and max is not None:
            expected_range = (min, max)
        else:
            expected_range = None

        super().__init__(name, kind=kind, function=func,
                         expected_range=expected_range,
                         variable_name=variable_name)
        self.transform = transform if transform is not None else 'none'
        self.address = address
        self.dtype = dtype if dtype is not None else 'real'
        self.metamodeltype = metamodeltype if metamodeltype is not None else 'linear'
        self._shortname = shortname
        self.desc = desc
        """str: Human readable description of this performance measure, for reference only"""

        self.formula = formula
        """str: An eval-able expression to compute this performance measure from other measures"""

        self.parser = parser
        """dict: Instructions for how to parse this performance measure from raw output files"""

        if tags:
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                self.add_tag(tag)

        self.meta = meta

    def __repr__(self):
        return super().__repr__()

    def _hash_it(self, ha=None):
        from ..util.hasher import hash_it
        return hash_it(
            self.name,
            self.kind,
            self._expected_range,
            self.address,
            self.dtype,
            self.function is None,
            self.transform,
            tuple(self.variable_name),
            self.shape,
            self.shortname,
            self.metamodeltype,
            ha=ha,
        )

    def info(self, return_string=False):
        """Print some information about this measure

        Args:
            return_string (bool): Defaults False (print to stdout) but if given as True
                then this function returns the string instead of printing it.
        """

        if return_string:
            import io
            f = io.StringIO
        else:
            f = None

        print(f"{self.name}:")
        if self._shortname:
            print(f"  shortname: {self._shortname}", file=f)
        kind = {
            ScalarOutcome.MINIMIZE: 'minimize',
            ScalarOutcome.MAXIMIZE: 'maximize',
            ScalarOutcome.INFO: 'info',
            ScalarOutcome.TEMP: 'temp',
        }.get(self.kind)
        print(f"  kind: {kind}", file=f)
        if self.address:
            print(f"  address: {self.address}", file=f)
        if self.dtype != 'real':
            print(f"  dtype: {self.dtype}", file=f)
        if self.metamodeltype != 'linear':
            print(f"  metamodeltype: {self.metamodeltype}", file=f)
        try:
            expected_range = self.expected_range
        except ValueError:
            expected_range = None
        if expected_range is not None:
            print(f"  min: {expected_range[0]}", file=f)
            print(f"  max: {expected_range[1]}", file=f)

        if return_string:
            return f.getvalue()
