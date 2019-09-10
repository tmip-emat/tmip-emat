# -*- coding: utf-8 -*-

import numpy
from ema_workbench import ScalarOutcome
from .names import ShortnameMixin

class Measure(ScalarOutcome, ShortnameMixin):
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
        kind (int): {MINIMIZE, MAXIMIZE, INFO}
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
    ):

        if isinstance(kind, str):
            if kind.lower()=='minimize':
                kind = ScalarOutcome.MINIMIZE
            elif kind.lower()=='maximize':
                kind = ScalarOutcome.MAXIMIZE
            elif kind.lower() == 'info':
                kind = ScalarOutcome.INFO
            else:
                raise TypeError(f'invalid kind {kind}')

        if transform is None:
            func = function
            if function is not None:
                transform = f'f:{function}'
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

    def __repr__(self):
        return super().__repr__()


