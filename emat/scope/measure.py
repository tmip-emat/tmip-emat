# -*- coding: utf-8 -*-

import numpy
from ema_workbench import ScalarOutcome


class Measure(ScalarOutcome):
    '''
    Measure represents an outcome measure of the model.

    Args:
        name (str): Name of the measure.
        kind (str or int, optional): one of {'info', 'minimize', 'maximize'},
            defaults to 'info' if not given. This represents the
            generally preferred direction of the measure.
        transform (str, optional): what kind of post-processing
            transformation should be applied to the measure before
            subsequent analysis, if any. Currently only "ln" is
            implemented.
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

    Attributes:
        name (str): Name of the measure.
        kind (int): {MINIMIZE, MAXIMIZE, INFO}

    '''

    def __init__(
            self, name, kind=ScalarOutcome.INFO, transform=None,
            min=None, max=None, address=None, variable_name=None,
            function=None, dtype=None, metamodeltype=None,
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
        elif isinstance(transform, str) and transform.lower() in ('ln', 'log'):
            func = numpy.log
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
        self.dtype = dtype
        self.metamodeltype = metamodeltype if metamodeltype is not None else 'linear'

    def __repr__(self):
        return super().__repr__()

