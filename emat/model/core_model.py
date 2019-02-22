# -*- coding: utf-8 -*-
""" core_model.py - define coure model API"""
import abc
import yaml
import pandas as pd
import numpy as np
from typing import Union, Mapping
from ema_workbench.em_framework.model import AbstractModel as AbstractWorkbenchModel
from typing import Collection

from ..database.database import Database
from ..scope.scope import Scope
from .._pkg_constants import *

from ..util.loggers import get_module_logger
_logger = get_module_logger(__name__)

class AbstractCoreModel(abc.ABC, AbstractWorkbenchModel):
    """
    An interface for using a model with EMAT.

    Individual models should be instantiated using derived
    subclasses of this abstract base class, and not using
    this class directly.

    Args:
        configuration: The configuration for this
            core model. This can be passed as a dict, or as a str
            which gives the filename of a YAML file that will be
            loaded. If there is no configuration, giving None is
            also acceptable.
        scope (Scope or str): The exploration scope, as a Scope object or as
            a str which gives the filename of a YAML file that will be
            loaded.
        safe: Load the configuration YAML file in 'safe' mode.
            This can be disabled if the configuration requires
            custom Python types or is otherwise not compatible with
            safe mode. Loading configuration files with safe mode
            off is not secure and should not be done with files from
            untrusted sources.
        db: An optional Database to store experiments and results.
        name: A name for this model, given as an alphanumeric string.
            The name is required by ema_workbench operations.
            If not given, "EMAT" is used.
        metamodel_id: An identifier for this model, if it is a meta-model.
            Defaults to 0 (i.e., not a meta-model).
    """

    def __init__(self,
                 configuration:Union[str,Mapping,None],
                 scope:Union[Scope,str],
                 safe:bool=True,
                 db:Database=None,
                 name:str='EMAT',
                 metamodel_id:int=0,
                 ):
        if isinstance(configuration, str):
            with open(configuration, 'r') as stream:
                if safe:
                    configuration = yaml.safe_load(stream)
                else:
                    configuration = yaml.load(stream)
            if configuration is None:
                configuration = {}

        self.config = configuration if configuration is not None else {}
        self.db = db
        if isinstance(scope, Scope):
            self.scope = scope
        else:
            self.scope = Scope(scope)

        AbstractWorkbenchModel.__init__(self, name=name.replace('_','').replace(' ',''))
        self.uncertainties = self.scope._x_list
        self.levers = self.scope._l_list
        self.constants = self.scope._c_list
        self.outcomes = self.scope._m_list

        self.metamodel_id = metamodel_id

    def __getstate__(self):
        # don't pickle the db connection
        return dict((k, v) for (k, v) in self.__dict__.items() if (k != 'db'))

    @abc.abstractmethod
    def setup(self, params):
        """
        Configure the core model with the experiment variable values
        
        
        Args:
            params (dict): experiment variables including both exogenous 
                uncertainty and policy levers
                
        Raises:
            KeyError: if experiment variable defined is not supported
                by the core model        
        """     
 
    @abc.abstractmethod
    def get_experiment_archive_path(self, experiment_id: int) -> str:
        """
        Returns path to store model run outputs
        
        Can be useful for long model runs if additional measures will be
        defined at a later time (e.g. link volumes). 
        
        Both the scope name and experiment id can be used to create the 
        folder path. 
        
        Args:
            experiment_id (int):
                experiment id integer (row id of experiment in database)
                
        Returns:
            str: model result path (no trailing backslashes)
        """     
    
    @abc.abstractmethod
    def run(self):
        """
        Initiates the core model run
        
        Model should be 'setup' first
                
        Raises:
            UserWarning: If model is not properly setup
        """     
    
    @abc.abstractmethod
    def post_process(self, params, measure_names, output_path=None):
        """
        Runs post processors associated with measures.

        The model should have previously been prepared using
        the `setup` method.

        Args:
            params (dict):
                Dictionary of experiment variables - indices
                are variable names, values are the experiment settings
            measure_names (List[str]):
                List of measures to be processed
            output_path (str):
                Path to model outputs - if set to none
                will use local values

        Raises:
            KeyError:
                If post process is not available for specified
                measure
        """
    
    @abc.abstractmethod
    def load_measures(self, measure_names, output_path=None) -> dict:
        """
        Import selected measures into dataframe
        
        Imports measures from active scenario
        
        Args:
            measure_names (List[str]): List of measures to be processed
            output_path (str): Path to model output locations
        
        Returns:
            dict of measure name and values from active scenario
        
        Raises:
            KeyError: If post process is not available for specified
                measure
        """           
        

    @abc.abstractmethod
    def archive(self, params, model_results_path, experiment_id:int=0):
        """
        Copies model outputs to archive location
        
        Args:
            params (dict): Dictionary of experiment variables
            model_results_path (str): archive path
            experiment_id (int, optional): The id number for this experiment.
        
        """

    def read_experiments(
            self,
            design_name,
            db=None,
            only_pending=False,
    ):
        """
        Reads results from a design of experiments from the database.

        Args:
            design_name (str): The name of the design to load.
            db (Database, optional): The Database from which to read experiments.
                If no db is given, the default `db` for this model is used.
            only_pending (bool, default False): If True, only pending
                experiments (which have no performance measure results
                stored in the database) are returned.

        Returns:
            pandas.DataFrame:
                A DataFrame that contains all uncertainties, levers, and measures
                for the experiments.

        Raises:
            ValueError:
                If there is no Database connection `db` set.
        """
        db = db if db is not None else self.db
        if db is None:
            raise ValueError('no database to read from')

        return self.ensure_dtypes(
            db.read_experiment_all(self.scope.name, design_name, only_pending=only_pending)
        )

    def read_experiment_parameters(
            self,
            design_name,
            db=None,
            only_pending=False,
    ):
        """
        Reads uncertainties and levers from a design of experiments from the database.

        Args:
            design_name (str): The name of the design to load.
            db (Database, optional): The Database from which to read experiments.
                If no db is given, the default `db` for this model is used.
            only_pending (bool, default False): If True, only pending
                experiments (which have no performance measure results
                stored in the database) are returned.

        Returns:
            pandas.DataFrame:
                A DataFrame that contains all uncertainties, levers, and measures
                for the experiments.

        Raises:
            ValueError:
                If `db` is not given and there is no default
                Database connection set.
        """
        db = db if db is not None else self.db

        if db is None:
            raise ValueError('no database to read from')

        return self.ensure_dtypes(
            db.read_experiment_parameters(self.scope.name, design_name, only_pending=only_pending)
        )

    def read_experiment_measures(
            self,
            design_name,
            experiment_id=None,
            db=None,
    ):
        """
        Reads performace measures from a design of experiments from the database.

        Args:
            design_name (str): The name of the design to load.
            experiment_id (int, optional): The id of the experiment to load.
            db (Database, optional): The Database from which to read experiment(s).
                If no db is given, the default `db` for this model is used.

        Returns:
            pandas.DataFrame:
                A DataFrame that contains all uncertainties, levers, and measures
                for the experiments.

        Raises:
            ValueError:
                If `db` is not given and there is no default
                Database connection set.
        """
        db = db if db is not None else self.db

        if db is None:
            raise ValueError('no database to read from')

        return self.ensure_dtypes(
            db.read_experiment_measures(self.scope.name, design_name, experiment_id)
        )

    def ensure_dtypes(self, df:pd.DataFrame):
        """
        Convert columns of dataframe to correct dtype as needed.

        Args:
            df (pandas.DataFrame): A dataframe with column names
                that are uncertainties, levers, or measures.

        Returns:
            pandas.DataFrame:
                The same data as input, but with dtypes as appropriate.
        """
        return self.scope.ensure_dtypes(df)

    def design_experiments(self, *args, **kwargs):
        """
        Create a design of experiments based on this model.

        Args:
            n_samples_per_factor (int, default 10): The number of samples in the
                design per random factor.
            n_samples (int, optional): The total number of samples in the
                design.  If this value is given, it overrides `n_samples_per_factor`.
            random_seed (int or None, default 1234): A random seed for reproducibility.
            db (Database, optional): If provided, this design will be stored in the
                database indicated.  If not provided, the `db` for this model will
                be used, if one is set.
            design_name (str, optional): A name for this design, to identify it in the
                database. If not given, a unique name will be generated based on the
                selected sampler.  Has no effect if no `db` is given.
            sampler (str or AbstractSampler, default 'lhs'): The sampler to use for this
                design.
            sample_from ('all', 'uncertainties', or 'levers'): Which scope components
                from which to sample.  Components not sampled are set at their default
                values in the design.

        Returns:
            pandas.DataFrame: The resulting design.
        """
        if 'scope' in kwargs:
            kwargs.pop('scope')

        if 'db' not in kwargs:
            kwargs['db'] = self.db

        from ..experiment import experimental_design
        return experimental_design.design_experiments(self.scope, *args, **kwargs)

    def run_experiments(
            self,
            design:pd.DataFrame=None,
            evaluator=None,
            *,
            design_name=None,
            db=None,
    ):
        """
        Runs a design of combined experiments using this model.

        A combined experiment includes a complete set of input values for
        all exogenous uncertainties (a Scenario) and all policy levers
        (a Policy). Unlike the perform_experiments function in the EMA Workbench,
        this method pairs each Scenario and Policy in sequence, instead
        of running all possible combinations of Scenario and Policy.

        Args:
            design (pandas.DataFrame, optional): experiment definitions
                given as a DataFrame, where each exogenous uncertainties and
                policy levers is given as a column, and each row is an experiment.
            evaluator (ema_workbench.Evaluator, optional): Optionally give an
                evaluator instance.  If not given, a default SequentialEvaluator
                will be instantiated.
            design_name (str, optional): The name of a design of experiments to
                load from the database.  This design is only used if
                `experiment_parameters` is None.
            db (Database, optional): The database to use for loading and saving experiments.
                If none is given, the default database for this model is used.
                If there is no default db, and none is given here,
                the results are not stored in a database.

        Returns:
            pandas.DataFrame:
                A DataFrame that contains all uncertainties, levers, and measures
                for the experiments.

        Raises:
            ValueError:
                If there are no experiments defined.  This includes
                the situation where `design` is given but no database is
                available.

        """

        from ema_workbench import Scenario, Policy, perform_experiments

        # catch user gives only a design, not experiment_parameters
        if isinstance(design, str) and design_name is None:
            design_name, design = design, None

        if design_name is None and design is None:
            raise ValueError(f"must give design_name or design")

        if db is None:
            db = self.db

        if design_name is not None and design is None:
            if db is None:
                raise ValueError(f'cannot load design "{design_name}", there is no db')
            design = db.read_experiment_parameters(self.scope.name, design_name)

        if design.empty:
            raise ValueError(f"no experiments available")

        scenarios = [
            Scenario(**dict(zip(self.scope.get_uncertainty_names(), i)))
            for i in design[self.scope.get_uncertainty_names()].itertuples(index=False,
                                                                           name='ExperimentX')
        ]

        policies = [
            Policy(f"Incognito{n}", **dict(zip(self.scope.get_lever_names(), i)))
            for n,i in enumerate(design[self.scope.get_lever_names()].itertuples(index=False,
                                                                                 name='ExperimentL'))
        ]

        if not evaluator:
            from ema_workbench import SequentialEvaluator
            evaluator = SequentialEvaluator(self)

        experiments, outcomes = perform_experiments(self, scenarios=scenarios, policies=policies,
                                                    zip_over={'scenarios', 'policies'}, evaluator=evaluator)
        experiments.index = design.index



        outcomes = pd.DataFrame.from_dict(outcomes)
        outcomes.index = design.index

        if db is not None:
            db.write_experiment_measures(self.scope.name, self.metamodel_id, outcomes)

        return self.ensure_dtypes(pd.concat([
            experiments.drop(columns=['scenario','policy','model']),
            outcomes
        ], axis=1, sort=False))


    def run_experiments_from_design(self, design_name='lhs', archive=True):
        """
        Runs a design of experiments through this core model

        For each experiment, the core model is called to:

            1. set experiment variables
            2. run the experiment
            3. run post-processors associated with specified
               performance measures
            4. (optionally) archive model outputs
            5. record performance measures to database

        Args:
            design_name (str): experiment design type:
                'uni' - generated by univariate sensitivity test design
                'lhs' - generated by latin hypercube sample design
            archive (bool): option to call core_model archive function
                that copies outputs to an archive location
        Raises:
            UserWarning: If there are no experiments associated with
                this type.

        """
        _logger.debug("run_experiments_from_design read_experiment_parameters")

        if design_name is None:
            possible_designs = self.db.read_design_names(self.scope.name)
            if len(possible_designs) == 0:
                raise ValueError('no experimental designs found')
            if len(possible_designs) > 1:
                raise ValueError('multiple experimental designs found, '
                                 'must explicitly give design name')
            design_name = possible_designs[0]

        ex_xl = self.db.read_experiment_parameters(self.scope.name, design_name)
        if ex_xl.empty is True:
            raise UserWarning("No experiments available of design {0}"
                              .format(design_name))

        m_names = [m.name for m in self.scope._m_list]

        m_out = pd.DataFrame()

        for ex_id, xl in ex_xl.iterrows():

            _logger.debug(f"run_core_model setup {ex_id}")
            self.setup(xl.to_dict())

            _logger.debug(f"run_core_model run {ex_id}")
            self.run()

            _logger.debug(f"run_core_model post_process {ex_id}")
            self.post_process(xl.to_dict(), m_names)

            _logger.debug(f"run_core_model wrap up {ex_id}")
            m_di = self.load_measures(m_names)
            m_df = pd.DataFrame(m_di, index=[ex_id])

            _logger.debug(f"run_core_model write db {ex_id}")
            self.db.write_experiment_measures(self.scope.name, self.metamodel_id, m_df)
            m_out = pd.concat([m_out, m_df])

            if archive:
                _logger.debug(f"run_core_model archive {ex_id}")
                self.archive(xl, self.get_experiment_archive_path(ex_id))

    def create_metamodel_from_data(
            self,
            experiment_inputs:pd.DataFrame,
            experiment_outputs:pd.DataFrame,
            output_transforms: dict = None,
            metamodel_id:int = None,
            include_measures = None,
            exclude_measures = None,
            db = None,
            random_state = None,
    ):
        """
        Create a MetaModel from a set of input and output observations.

        Args:
            experiment_inputs (pandas.DataFrame): This dataframe
                should contain all of the experimental inputs, including
                values for each uncertainty, level, and constant.
            experiment_outputs (pandas.DataFrame): This dataframe
                should contain all of the experimental outputs, including
                a column for each performance measure. The index
                for the outputs should match the index for the
                `experiment_inputs`, so that the I-O matches row-by-row.
            output_transforms (dict): A mapping of performance measure
                transforms to use in meta-model estimation and application.
            metamodel_id (int, optional): An identifier for this meta-model.
                If not given, a unique id number will be created randomly.
            include_measures (Collection[str], optional): If provided, only
                output performance measures with names in this set will be included.
            exclude_measures (Collection[str], optional): If provided, only
                output performance measures with names not in this set will be included.
            db (Database, optional): The database to use for loading and saving metamodels.
                If none is given, the default database for this model is used.
                If there is no default db, and none is given here,
                the metamodel is not stored in a database.
            random_state (int, optional): A random state to use in the metamodel
                regression fitting.

        Returns:
            MetaModel:
                a callable object that, when called as if a
                function, accepts keyword arguments as inputs and
                returns a dictionary of (measure name: value) pairs.
        """
        from .core_python import PythonCoreModel
        from .meta_model import MetaModel

        db = db if db is not None else self.db

        experiment_inputs = self.ensure_dtypes(experiment_inputs)

        if metamodel_id is None:
            if db is not None:
                scope_name = self.scope.name
                metamodel_id = db.get_new_metamodel_id(scope_name)
            else:
                metamodel_id = np.random.randint(1,2**63,dtype='int64')

        if include_measures is not None:
            experiment_outputs = experiment_outputs[[i for i in include_measures
                                                     if i in experiment_outputs.columns]]
        if exclude_measures is not None:
            experiment_outputs = experiment_outputs.drop(exclude_measures, axis=1)

        disabled_outputs = [i for i in self.scope.get_measure_names()
                            if i not in experiment_outputs.columns]

        func = MetaModel(experiment_inputs, experiment_outputs,
                         output_transforms, disabled_outputs, random_state)

        scope_ = self.scope.duplicate(strip_measure_transforms=True)

        return PythonCoreModel(
            func,
            configuration = None,
            scope=scope_,
            safe=True,
            db = self.db,
            name=self.name+"Meta",
            metamodel_id=metamodel_id,
        )

    def create_metamodel_from_design(
            self,
            design_name:str,
            metamodel_id:int = None,
            include_measures=None,
            exclude_measures=None,
            db=None,
            random_state=None,
    ):
        """
        Create a MetaModel from a set of input and output observations.

        Args:
            design_name (str): The name of the design to use.
            metamodel_id (int, optional): An identifier for this meta-model.
                If not given, a unique id number will be created randomly.
            include_measures (Collection[str], optional): If provided, only
                output performance measures with names in this set will be included.
            exclude_measures (Collection[str], optional): If provided, only
                output performance measures with names not in this set will be included.
            random_state (int, optional): A random state to use in the metamodel
                regression fitting.

        Returns:
            MetaModel:
                a callable object that, when called as if a
                function, accepts keyword arguments as inputs and
                returns a dictionary of (measure name: value) pairs.

        Raises:
            ValueError: If the named design still has pending experiments.
        """
        db = db if db is not None else self.db

        if db is not None:
            check_df = db.read_experiment_parameters(self.scope.name, design_name, only_pending=True)
            if not check_df.empty:
                from ..exceptions import PendingExperimentsError
                raise PendingExperimentsError(f'design "{design_name}" has pending experiments')

        experiment_inputs = self.db.read_experiment_parameters(self.scope.name, design_name)
        experiment_outputs = self.db.read_experiment_measures(self.scope.name, design_name)

        transforms = {
            i.name: i.metamodeltype
            for i in self.scope.get_measures()
        }

        return self.create_metamodel_from_data(
            experiment_inputs,
            experiment_outputs,
            transforms,
            metamodel_id=metamodel_id,
            include_measures=include_measures,
            exclude_measures=exclude_measures,
            db=db,
            random_state=random_state,
        )


    def get_feature_scores(
            self,
            design,
            return_raw=False,
    ):
        """
        Calculate feature scores based on a design of experiments.

        Args:
            design (str or pandas.DataFrame): The name of the design of experiments
                to use for feature scoring, or a pandas.DataFrame containing the
                experimental design and results.
            return_raw (bool, default False): Whether to return a raw pandas.DataFrame
                containing the computed feature scores, instead of a formatted heatmap
                table.

        Returns:
            xmle.Elem or pandas.DataFrame:
                Returns a rendered SVG as xml, or a DataFrame,
                depending on the `return_raw` argument.

        This function internally uses feature_scoring from the EMA Workbench, which in turn
        scores features using the "extra trees" regression approach.
        """
        from ema_workbench.analysis import feature_scoring
        from ..viz import heatmap_table
        import pandas

        if isinstance(design, str):
            inputs = self.read_experiment_parameters(design)
            outcomes = self.read_experiment_measures(design)
        elif isinstance(design, pandas.DataFrame):
            inputs = design[[c for c in design.columns if c in self.scope.get_parameter_names()]]
            outcomes = design[[c for c in design.columns if c in self.scope.get_measure_names()]]
        else:
            raise TypeError('must name design or give DataFrame')

        fs = feature_scoring.get_feature_scores_all(inputs, outcomes)
        if return_raw:
            return fs
        return heatmap_table(
            fs.T,
            xlabel='Model Parameters', ylabel='Performance Measures',
            title='Feature Scoring' + (f' [{design}]' if design else ''),
        )

    def robust_optimize(
            self,
            robustness_functions,
            scenarios,
            evaluator=None,
            nfe=10000,
            convergence=None,
            constraints=None,
            # epsilons=None,
            **kwargs,
    ):
        """
        Perform robust optimization.

        Args:
            robustness_functions (Collection[Measure]): A collection of
                aggregate statistical performance measures.
            scenarios : int, or collection
            evaluator : Evaluator instance
            algorithm : platypus Algorithm instance
            nfe : int
            constraints : list
            kwargs : any additional arguments will be passed on to algorithm

        Raises
        ------
        AssertionError if robustness_function is not a ScalarOutcome,
        if robustness_funcion.kind is INFO, or
        if robustness_function.function is None

        robustness functions are scalar outcomes, kind should be MINIMIZE or
        MAXIMIZE, function is the robustness function you want to use.

        """


        if evaluator is None:
            from ema_workbench.em_framework import SequentialEvaluator
            evaluator = SequentialEvaluator(self)

        from ema_workbench.em_framework.samplers import sample_uncertainties, sample_levers

        if isinstance(scenarios, int):
            n_scenarios = scenarios
            scenarios = sample_uncertainties(self, n_scenarios)

        # if epsilons is None:
        #     epsilons = [0.05, ] * len(robustness_functions)
        #
        with evaluator:
            robust_results = evaluator.robust_optimize(
                robustness_functions,
                scenarios,
                nfe=nfe,
                constraints=constraints,
                # epsilons=epsilons,
                convergence=convergence,
                **kwargs,
            )

        if isinstance(robust_results, tuple) and len(robust_results) == 2:
            robust_results, result_convergence = robust_results
        else:
            result_convergence = None

        robust_results = self.ensure_dtypes(robust_results)

        if result_convergence is None:
            return robust_results
        else:
            return robust_results, result_convergence

