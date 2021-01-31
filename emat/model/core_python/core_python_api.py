# -*- coding: utf-8 -*-
import yaml
import os
import time
import inspect
import pandas

from typing import Union, Mapping, Callable, Collection
from ...workbench.em_framework import Model as WorkbenchModel

from ...scope.scope import Scope
from ...database.database import Database
from ...model.core_model import AbstractCoreModel
from ...model.core_python.core_python_examples import Dummy

from ...util.docstrings import copydoc


# def filter_dict(dict_to_filter, thing_with_kwargs):
#     sig = inspect.signature(thing_with_kwargs)
#     filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
#     filtered_dict = {filter_key:dict_to_filter[filter_key] for filter_key in filter_keys}
#     return filtered_dict



class PythonCoreModel(AbstractCoreModel, WorkbenchModel):
    """
    An EMAT interface for a core model that is a Python function.

    Args:
        function (callable):
            The Python function to be evaluated.  This function
            must accept keyword arguments that include all
            of the uncertainties and levers, and return a dict
            that contains key-value pairs that map the names
            of performance measures to the computed performance
            measure outputs.
        configuration (str, dict, or None):
            The configuration for this core model. This can be
            passed as a dict, or as a str which gives the filename
            of a yaml file that will be loaded. A core model that
            is a stand-alone Python function will not often not
            require any configuration.
        scope (Scope or str):
            The Scope for this exploratory analysis. Can be given as
            an explicit Scope object, or as a str which gives the
            filename of a yaml file that will be loaded.
        safe (bool):
            Load the configuration yaml file in 'safe' mode.
            This can be disabled if the configuration requires
            custom Python types or is otherwise not compatible with
            safe mode. Loading configuration files with safe mode
            off is not secure and should not be done with files from
            untrusted sources.
        db (Database): An optional default Database to store experiments
            and results.
        name (str): A name for this model, given as an alphanumeric string.
            The name is required by ema_workbench operations.
            If not given, the name of the function is extracted, or
            failing that, "EMAT" is used.
        metamodel_id: An identifier for this model, if it is a meta-model.
            Defaults to 0 (i.e., not a meta-model).
    """

    xl_di = {}

    def __init__(self,
                 function:Callable,
                 configuration:Union[str,Mapping,None]=None,
                 scope:Union[Scope,str]=None,
                 safe:bool=True,
                 db:Database=None,
                 name:str='EMAT',
                 metamodel_id=None,
                 ):
        if scope is None:
            raise ValueError('must give scope')

        if name == 'EMAT':
            try:
                _name = self.function.__name__
            except:
                pass
            else:
                if _name.isalnum():
                    name = _name
                elif _name.replace("_","").replace(" ","").isalnum():
                    name = _name.replace("_","").replace(" ","")

        AbstractCoreModel.__init__(self, configuration, scope, safe, db, metamodel_id=metamodel_id)

        self.archive_path = self.config.get('archive_path', None)

        if self.archive_path is not None:
            os.makedirs(self.archive_path, exist_ok=True)

        # If no archive path is given, a temporary directory is created.
        # All archive files will be lost when this CoreDummy is deleted.
        if self.archive_path is None:
            import tempfile
            self._temp_archive = tempfile.TemporaryDirectory()
            self.archive_path = self._temp_archive.name

        WorkbenchModel.__init__(self, name, function)

    def __repr__(self):
        content = []
        if len(self.scope._c_list):
            content.append(f"{len(self.scope._c_list)} constants")
        if len(self.scope._x_list):
            content.append(f"{len(self.scope._x_list)} uncertainties")
        if len(self.scope._l_list):
            content.append(f"{len(self.scope._l_list)} levers")
        if len(self.scope._m_list):
            content.append(f"{len(self.scope._m_list)} measures")
        metamodel_tag = "" if (self.metamodel_id==0 or self.metamodel_id is None) else f", metamodel_id={self.metamodel_id}"
        return f'<emat.PythonCoreModel "{self.name}"{metamodel_tag} with {", ".join(content)}>'

    @copydoc(AbstractCoreModel.setup)
    def setup(self, params):
        self.xl_di = params
    
    @copydoc(AbstractCoreModel.get_experiment_archive_path)
    def get_experiment_archive_path(self, experiment_id=None, makedirs=False, parameters=None):
        ''' Path is defined with scope name and experiment id '''
        if experiment_id is None:
            if parameters is None:
                raise ValueError("must give `experiment_id` or `parameters`")
            db = getattr(self, 'db', None)
            if db is not None:
                experiment_id = db.get_experiment_id(self.scope.name, parameters)
        mod_results_path = os.path.join(
            self.archive_path,
            f"scp_{self.scope.name}",
            f"exp_{experiment_id}"
        )
        if makedirs:
            os.makedirs(mod_results_path, exist_ok=True)
        return mod_results_path

    @copydoc(AbstractCoreModel.run)
    def run(self):
        self.outcomes_output = self.function(**self.xl_di)

    @copydoc(AbstractCoreModel.post_process)
    def post_process(self, params, measure_names, output_path=None):
        """Not used for PythonCoreModel objects."""
    
    @copydoc(AbstractCoreModel.load_measures)
    def load_measures(
            self,
            measure_names: Collection[str]=None,
            *,
            rel_output_path=None,
            abs_output_path=None,
    ):

        result = self.outcomes_output

        if measure_names is None:
            return result.copy()

        pm_dict = {}
        for pm in measure_names:
            if pm not in result.keys():
                raise KeyError('Measure {0} not supported'.format(pm))
            pm_dict[pm] = result[pm]

        return pm_dict

    @copydoc(AbstractCoreModel.archive)
    def archive(self, params, model_results_path=None, experiment_id:int=0):
        """archive only records experiment values"""
        if not os.path.exists(model_results_path):
            os.makedirs(model_results_path)

        # record experiment definitions
        xl_df = pandas.DataFrame(params, index=[experiment_id])
        xl_df.to_csv(model_results_path + r'_def.csv')

    def run_experiment(self, experiment):
        """
        Running a single instantiated model experiment.

        The results are passed through the performance measure
        processing steps to generate results.

        Args:
            experiment (dict-like)

        Returns:
            dict
        """
        self.outcomes_output = super().run_experiment(experiment)
        return self.outcomes_output

    def __getattr__(self, item):
        """
        Pass through getattr to the function.
        """

        try:
            f = object.__getattribute__(self, 'function')
            if hasattr(f, item):
                return getattr(f, item)
        except:
            pass
        raise AttributeError(item)