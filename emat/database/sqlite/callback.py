
import pandas as pd
import numpy as np

from ...workbench import RealParameter, IntegerParameter, BooleanParameter, CategoricalParameter
from ...workbench.em_framework.callbacks import AbstractCallback
from ...workbench.util.ema_exceptions import EMAError

from ...util.loggers import get_module_logger
from ..._pkg_constants import *

_logger = get_module_logger(__name__)

class SQLiteCallback(AbstractCallback):
    """
    default callback system
    callback can be used in perform_experiments as a means for
    specifying the way in which the results should be handled. If no
    callback is specified, this default implementation is used. This
    one can be overwritten or replaced with a callback of your own
    design. For example if you prefer to store the result in a database
    or write them to a text file
    """
    i = 0
    cases = None
    results = {}

    shape_error_msg = "can only save up to 2d arrays, this array is {}d"
    constraint_error_msg = ('can only save 1d arrays for constraint, '
                            'this array is {}d')

    def __init__(self, uncs, levers, outcomes, nr_experiments,
                 reporting_interval=100, reporting_frequency=10,
                 scope_name=None, design_name=None, db=None,
                 using_metamodel=False, metamodel_id=12345,
                 ):
        '''

        Parameters
        ----------
        uncs : list
                a list of the parameters over which the experiments
                are being run.
        outcomes : list
                   a list of outcomes
        nr_experiments : int
                         the total number of experiments to be executed
        reporting_interval : int, optional
                             the interval between progress logs
        reporting_frequency: int, optional
                             the total number of progress logs

        '''
        super().__init__(uncs, levers, outcomes,
                         nr_experiments, reporting_interval,
                         reporting_frequency)
        self.i = 0
        self.cases = None
        self.results = {}

        self.outcomes = [outcome.name for outcome in outcomes]

        # determine data types of parameters
        columns = []
        dtypes = []
        self.parameters = []

        for parameter in uncs + levers:
            name = parameter.name
            self.parameters.append(name)
            dataType = 'float'

            if isinstance(parameter, CategoricalParameter):
                dataType = 'object'
            elif isinstance(parameter, BooleanParameter):
                dataType = 'bool'
            elif isinstance(parameter, IntegerParameter):
                dataType = 'int'
            columns.append(name)
            dtypes.append(dataType)

        for name in ['scenario', 'policy', 'model']:
            columns.append(name)
            dtypes.append('object')

        df = pd.DataFrame(index=np.arange(nr_experiments))

        for name, dtype in zip(columns, dtypes):
            df[name] = pd.Series(dtype=dtype)
        self.cases = df
        self.nr_experiments = nr_experiments

        self.scope_name = scope_name
        self.design_name = design_name
        self.db = db
        self.using_metamodel = using_metamodel
        self.metamodel_id = metamodel_id

    def _store_case(self, experiment):
        scenario = experiment.scenario
        policy = experiment.policy
        index = experiment.experiment_id

        self.cases.at[index, 'scenario'] = scenario.name
        self.cases.at[index, 'policy'] = policy.name
        self.cases.at[index, 'model'] = experiment.model_name

        for k, v in scenario.items():
            self.cases.at[index, k] = v

        for k, v in policy.items():
            self.cases.at[index, k] = v

        ex_ids = self.db.write_experiment_parameters(self.scope_name, self.design_name, self.cases.iloc[index:index+1, :-3])
        return ex_ids[0]

    def _store_outcomes(self, case_id, outcomes, ex_id):
        for outcome in self.outcomes:

            try:
                outcome_res = outcomes[outcome]
            except KeyError:
                message = "%s not specified as outcome in msi" % outcome
                _logger.debug(message)
            else:
                # outcome is found, store it
                try:
                    self.results[outcome][case_id, ] = outcome_res
                except KeyError:
                    # outcome is non-scalar
                    shape = np.asarray(outcome_res).shape

                    if len(shape) > 2:
                        message = self.shape_error_msg.format(len(shape))
                        raise EMAError(message)

                    shape = list(shape)
                    shape.insert(0, self.nr_experiments)

                    self.results[outcome] = np.empty(shape)
                    self.results[outcome][:] = np.NAN
                    self.results[outcome][case_id, ] = outcome_res

                _logger.debug("stored {} = {}".format(outcome, outcome_res))

                self.db.write_ex_m_1(self.scope_name,
                                     SOURCE_IS_CORE_MODEL if not self.using_metamodel else self.metamodel_id,
                                     ex_id,
                                     outcome,
                                     outcome_res,)

    def __call__(self, experiment, outcomes):
        '''
        Method responsible for storing results. This method calls
        :meth:`super` first, thus utilizing the logging provided there.

        Parameters
        ----------
        experiment: Experiment instance
        outcomes: dict
                the outcomes dict

        '''
        super().__call__(experiment, outcomes)

        # store the case
        ex_id = self._store_case(experiment)

        # store outcomes
        self._store_outcomes(experiment.experiment_id, outcomes, ex_id)

    def get_results(self):
        return self.cases, self.results



def SQLiteCallbackFactory(scope_name=None, design_name=None, db=None, using_metamodel=False):
    return lambda *a, **k: SQLiteCallback(*a,**k,
                                          scope_name=scope_name,
                                          design_name=design_name,
                                          db=db,
                                          using_metamodel=using_metamodel)

