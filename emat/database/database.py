# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:11:05 2018

@author: mmilkovits

Abstract Base Class for data storage format

"""

import abc
import pandas as pd
from contextlib import contextmanager

class Database(abc.ABC):
    
    """ Abstract Base Class for EMAT data storage
    
    Database constains the design experiments, meta-model parameters, 
    and the core and meta-model results (performance measures)
    """

    def __init__(self, readonly=False):
        self.readonly = readonly
        self.__locked = False

    @property
    @contextmanager
    def lock(self):
        """Context manager to temporarily mark this database as locked."""
        self.__locked = True
        yield
        self.__locked = False

    @property
    def is_locked(self):
        return self.readonly or self.__locked

    def get_db_info(self):
        """
        Get a short string describing this Database

        Returns:
            str
        """
        return "no info available"

    @abc.abstractmethod
    def init_xlm(self, parameter_list, measure_list):
        """
        Initialize or extend set of experiment variables and measures

        Initialize database with universe of risk variables,
        policy variables, and performance measures. All variables and measures
        defined in scopes must be defined in this set.
        This method only needs to be run
        once after creating a new database.

        Args:
            parameter_list (List[tuple]):
                Experiment variable tuples (variable name, type)
                where variable name is a string and
                type is 'uncertainty', 'lever', or 'constant'
            measure_list (List[tuple]):
                Performance measure tuples (name, transform), where
                name is a string and transform is a defined transformation
                used in metamodeling, currently supported include {'log', None}.

        """
    
    @abc.abstractmethod
    def _write_scope(self, scope_name, sheet, scp_xl, scp_m, content):
        """
        Save the emat scope information to the database.

        Generally users should not call this function directly,
        use `store_scope` instead.

        Args:
            scope_name (str):
                The scope name, used to identify experiments,
                performance measures, and results associated with this model.
                Multiple scopes can be stored in the same database.
            sheet (str):
                Yaml file name with scope definition.
            scp_xl (List[str]):
                Scope parameter names - both uncertainties and policy levers
            scp_m (List[str]):
                Scope performance measure names
            content (Scope, optional):
                Scope object to be pickled and stored in the database.
        Raises:
            KeyError:
                If scope name already exists, the scp_vars are not
                available, or the performance measures are not initialized
                in the database.

        """

    @abc.abstractmethod
    def update_scope(self, scope):
        """
        Update the emat scope information in the database.

        Args:
            scope (Scope): scope to update
        """

    @abc.abstractmethod
    def store_scope(self, scope):
        """
        Save an emat.Scope directly to the database.

        Args:
            scope (Scope): The scope object to store.
        Raises:
            KeyError: If scope name already exists.
        """

    @abc.abstractmethod
    def read_scope(self, scope_name=None):
        """
        Load the pickled scope from the database.

        Args:
            scope_name (str, optional):
                The name of the scope to load.  If not
                given and there is only one scope stored
                in the database, that scope is loaded. If not
                given and there are multiple scopes stored in
                the database, a KeyError is raised.

        Returns:
            Scope

        Raises:
            KeyError: If a name is given but is not found in
                the database, or if no name is given but there
                is more than one scope stored.
        """

    @abc.abstractmethod
    def add_scope_meas(self, scope_name, scp_m):
        """Update the set of performance measures associated with the scope
        
        Use this function when the core model runs are complete to add
        performance measures to the scope and post-process against the 
        archived results
          
        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run
            scp_m (List[str]): scope performance measures
        Raises:
            KeyError: If scope name does not exist or the 
                performance measures are not initialized in the database.
        
        """     
    
    @abc.abstractmethod
    def delete_scope(self, scope_name):
        """Delete the scope from the database
        
        Deletes the scope as well as any experiments and results associated
        with the scope
        
        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run"""  
        
    @abc.abstractmethod
    def write_experiment_parameters(
            self,
            scope_name,
            design_name,
            xl_df,
    ):
        """
        Write experiment definitions the the database.
        
        This method records values for each experiment parameter,
        for each experiment in a design of one or more experiments.
        
        Args:
            scope_name (str):
                A scope name, used to identify experiments,
                performance measures, and results associated with this
                exploratory analysis. The scope with this name should
                already have been stored in this database.
            design_name (str):
                An experiment design name. This name should be unique
                within the named scope, and typically will include a
                reference to the design sampler, for example:
                'uni' - generated by univariate sensitivity test design
                'lhs' - generated by latin hypercube sample design
                The design_name is used primarily to load groups of
                related experiments together.
            xl_df (pandas.Dataframe):
                The columns of this DataFrame are the experiment
                parameters (i.e. policy levers, uncertainties, and
                constants), and each row is an experiment.

        Returns:
            list: the experiment id's of the newly recorded experiments

        Raises:
            UserWarning: If scope name does not exist
            TypeError: If not all scope variables are defined in the 
                exp_def
        """     

    def write_experiment_parameters_1(
            self,
            scope_name,
            design_name: str,
            *args,
            **kwargs
    ):
        """
        Write experiment definitions for a single experiment.

        This method records values for each experiment parameter,
        for a single experiment only.

        Args:
            scope_name (str):
                A scope name, used to identify experiments,
                performance measures, and results associated with this
                exploratory analysis. The scope with this name should
                already have been stored in this database.
            design_name (str):
                An experiment design name. This name should be unique
                within the named scope, and typically will include a
                reference to the design sampler, for example:
                'uni' - generated by univariate sensitivity test design
                'lhs' - generated by latin hypercube sample design
                The design_name is used primarily to load groups of
                related experiments together.
            *args, **kwargs (Mapping[s]):
                A dictionary where the keys are experiment parameter names
                (i.e. policy levers, uncertainties, and constants), and
                values are the the parameter values for this experiment.
                Subsequent positional or keyword arguments are used to update
                the parameters.

        Returns:
            int: The experiment id of the newly recorded experiments

        Raises:
            UserWarning: If scope name does not exist
            TypeError: If not all scope variables are defined in the
                exp_def
        """
        parameters = {}
        for a in args:
            if a is not None:
                parameters.update(a)
        parameters.update(kwargs)
        xl_df = pd.DataFrame(parameters, index=[0])
        result = self.write_experiment_parameters(scope_name, design_name, xl_df)
        return result[0]


    @abc.abstractmethod
    def read_experiment_parameters(
            self,
            scope_name,
            design_name=None,
            only_pending=False,
            design=None,
            *,
            experiment_ids=None,
            ensure_dtypes=True,
    ):
        """
        Read experiment definitions from the database.
        
        Read the values for each experiment parameter per experiment.
        
        Args:
            scope_name (str):
                A scope name, used to identify experiments,
                performance measures, and results associated with this
                exploratory analysis.
            design_name (str, optional): If given, only experiments
                associated with both the scope and the named design
                are returned, otherwise all experiments associated
                with the scope are returned.
            only_pending (bool, default False): If True, only pending
                experiments (which have no performance measure results
                stored in the database) are returned.
            design (str, optional): Deprecated.  Use design_name.
            experiment_ids (Collection, optional):
                A collection of experiment id's to load.  If given,
                both `design_name` and `only_pending` are ignored.
            ensure_dtypes (bool, default True): If True, the scope
                associated with these experiments is also read out
                of the database, and that scope file is used to
                format experimental data consistently (i.e., as
                float, integer, bool, or categorical).

        Returns:
            emat.ExperimentalDesign:
                The experiment parameters are returned in a subclass
                of a normal pandas.DataFrame, which allows attaching
                the `design_name` as meta-data to the DataFrame.

        Raises:
            ValueError: if `scope_name` is not stored in this database
        """    

    @abc.abstractmethod
    def write_experiment_measures(
            self,
            scope_name,
            source,
            m_df,
            run_ids=None,
            experiment_id=None,
    ):
        """
        Write experiment results to the database.
        
        Write the performance measure results for each experiment 
        in the scope - if the scope does not exist, nothing is recorded.

        Note that the design_name is not required to write experiment
        measures, as the individual experiments from any design are
        uniquely identified by the experiment id's.
        
        Args:
            scope_name (str):
                A scope name, used to identify experiments,
                performance measures, and results associated with this
                exploratory analysis. The scope with this name should
                already have been stored in this database.
            source (int):
                An indicator of performance measure source. This should
                be 0 for a bona-fide run of the associated core models,
                or some non-zero metamodel_id number.
            m_df (pandas.DataFrame):
                The columns of this DataFrame are the performance
                measure names, and row indexes are the experiment id's.
            run_ids (pandas.Index, optional):
                Provide an optional index of universally unique run ids
                (UUIDs) for these results. The UUIDs can be used to help
                identify problems and organize model runs.

        Raises:
            UserWarning: If scope name does not exist        
        """         

    def write_experiment_run_status(
            self,
            scope_name,
            run_id,
            experiment_id,
            msg,
    ):
        """
        Write experiment status to the database.

        Parameters
        ----------
        scope_name : str
        run_id : UUID
        experiment_id : int
        msg : str
            The status to write.
        """

    def read_experiment_run_status(
            self,
            scope_name,
            design_name=None,
            *,
            experiment_id=None,
            experiment_ids=None,
    ):
        """
        Read experiment definitions from the database.

        Read the values for each experiment parameter per experiment.

        Args:
            scope_name (str):
                A scope name, used to identify experiments,
                performance measures, and results associated with this
                exploratory analysis.
            design_name (str, optional): If given, only experiments
                associated with both the scope and the named design
                are returned, otherwise all experiments associated
                with the scope are returned.
            experiment_ids (int, optional):
                A single experiment id to check.  If given,
                `design_name` is ignored.
            experiment_ids (int or Collection[int], optional):
                A collection of experiment id's to check.  If given,
                `design_name` is ignored.

        Returns:
            emat.ExperimentalDesign:
                The experiment run statuses are returned in a subclass
                of a normal pandas.DataFrame, which allows attaching
                the `design_name` as meta-data to the DataFrame.

        Raises:
            ValueError: if `scope_name` is not stored in this database
        """
        raise NotImplementedError

    @abc.abstractmethod
    def read_experiment_all(
            self,
            scope_name,
            design_name=None,
            source=None,
            *,
            only_pending=False,
            only_incomplete=False,
            only_complete=False,
            only_with_measures=False,
            ensure_dtypes=True,
            with_run_ids=False,
            runs=None,
    ):
        """
        Read experiment definitions and results
        
        Read the values from each experiment variable and the 
        results for each performance measure per experiment.
        
        Args:
            scope_name (str):
                A scope name, used to identify experiments,
                performance measures, and results associated with this
                exploratory analysis.
            design_name (str or Collection[str], optional):
                The experimental design name (a single `str`) or
                a collection of design names to read.
            source (int, optional): The source identifier of the
                experimental outcomes to load.  If not given, but
                there are only results from a single source in the
                database, those results are returned.  If there are
                results from multiple sources, an error is raised.
            only_pending (bool, default False): If True, only pending
                experiments (which have no performance measure results
                stored in the database) are returned. Experiments that
                have any results, even if only partial results, are
                excluded.
            only_incomplete (bool, default False): If True, only incomplete
                experiments (which have at least one missing performance
                measure result that is not stored in the database) are
                returned.  Only complete experiments (that have every
                performance measure populated) are excluded.
            only_complete (bool, default False): If True, only complete
                experiments (which have no missing performance measure
                results stored in the database) are returned.
            only_with_measures (bool, default False): If True, only
                experiments with at least one stored performance measure
                are returned.
            ensure_dtypes (bool, default True): If True, the scope
                associated with these experiments is also read out
                of the database, and that scope file is used to
                format experimental data consistently (i.e., as
                float, integer, bool, or categorical).
            with_run_ids (bool, default False): Whether to use a
                two-level pd.MultiIndex that includes both the
                experiment_id (which always appears in the index)
                as well as the run_id (which only appears in the
                index if this argument is set to True).
            runs ({None, 'all', 'valid', 'invalid'}, default None):
                By default, this method returns the one and only
                valid model run matching the given `design_name`
                and `source` (if any) for any experiment, and fails
                if there is more than one such valid run. Set this to
                'valid' or 'invalid' to get all valid or invalid model
                runs (instead of raising an exception). Set to 'all' to get
                everything, including both valid and invalidated results.

        Returns:
            emat.ExperimentalDesign:
                The experiment parameters are returned in a subclass
                of a normal pandas.DataFrame, which allows attaching
                the `design_name` as meta-data to the DataFrame.

        Raises:
            ValueError
                When no source is given but the database contains
                results from multiple sources.
        """

    @abc.abstractmethod
    def read_experiment_measures(
            self,
            scope_name,
            design_name=None,
            experiment_id=None,
            source=None,
            design=None,
            runs=None,
            formulas=True,
            with_validity=False,
    ):
        """
        Read experiment results from the database.

        Args:
            scope_name (str or Scope):
                A scope or just its name, used to identify experiments,
                performance measures, and results associated with this
                exploratory analysis.
            design_name (str, optional): If given, only experiments
                associated with both the scope and the named design
                are returned, otherwise all experiments associated
                with the scope are returned.
            experiment_id (int, optional): The id of the experiment to
                retrieve.  If omitted, get all experiments matching the
                named scope and design.
            source (int, optional): The source identifier of the
                experimental outcomes to load.  If not given, but
                there are only results from a single source in the
                database, those results are returned.  If there are
                results from multiple sources, an error is raised.
            design (str): Deprecated, use `design_name`.
            runs ({None, 'all', 'valid', 'invalid'}, default None):
                By default, this method fails if there is more than
                one valid model run matching the given `design_name`
                and `source` (if any) for any experiment.  Set this to
                'valid' or 'invalid' to get all valid or invalid model
                runs (instead of raising an exception). Set to 'all' to get
                everything, including both valid and invalidated results.
            formulas (bool, default True): If the scope includes
                formulaic measures (computed directly from other
                measures) then compute these values and include them in
                the results.

        Returns:
            results (pandas.DataFrame): performance measures

        Raises:
            ValueError
                When the database contains multiple sets of results
                matching the given `design_name` and/or `source`
                (if any) for any experiment.
        """

    @abc.abstractmethod
    def read_experiment_measure_sources(
            self,
            scope_name,
            design_name=None,
            experiment_id=None,
            design=None,
    ):
        """
        Read all source ids from the results stored in the database.

        Args:
            scope_name (str):
                A scope name, used to identify experiments,
                performance measures, and results associated with this
                exploratory analysis.
            design_name (str, optional): If given, only experiments
                associated with both the scope and the named design
                are returned, otherwise all experiments associated
                with the scope are returned.
            experiment_id (int, optional): The id of the experiment to
                retrieve.  If omitted, get all experiments matching the
                named scope and design.
            design (str): Deprecated, use `design_name`.

        Returns:
            List[Int]: performance measure source ids
        """


    @abc.abstractmethod
    def delete_experiments(self, scope_name, design_name=None, design=None):
        """
        Delete experiment definitions and results.
        
        The method removes the linkage between experiments and the
        identified experimental design.  Experiment parameters and results
        are only removed if they are also not linked to any other experimental
        design stored in the database.
        
        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run
            design_name (str): Only experiments
                associated with both the scope and the named design
                are deleted.
            design (str): Deprecated, use `design_name`.
        """


    @abc.abstractmethod
    def delete_experiment_measures(
            self,
            experiment_ids=None,
    ):
        """
        Delete experiment performance measure results.

        The method removes only the performance measures, not the
        parameters.  This can be useful if a set of corrupted model
        results was stored in the database.

        Args:
            experiment_ids (Collection, optional):
                A collection of experiment id's for which measures shall
                be deleted.  Note that no scope or design are given here,
                experiments must be individually identified.

        """


    @abc.abstractmethod
    def write_experiment_all(self, scope_name, design_name, source, xlm_df):
        """
        Write experiment definitions and results
        
        Writes the values from each experiment variable and the 
        results for each performance measure per experiment
        
        Args:
            scope_name (str):
                A scope name, used to identify experiments,
                performance measures, and results associated with this
                exploratory analysis. The scope with this name should
                already have been stored in this database.
            design_name (str):
                An experiment design name. This name should be unique
                within the named scope, and typically will include a
                reference to the design sampler, for example:
                'uni' - generated by univariate sensitivity test design
                'lhs' - generated by latin hypercube sample design
                The design_name is used primarily to load groups of
                related experiments together.
            source (int):
                An indicator of performance measure source. This should
                be 0 for a bona fide run of the associated core models,
                or some non-zero metamodel_id number.
            xlm_df (pandas.Dataframe):
                The columns of this DataFrame are the experiment
                parameters (i.e. policy levers, uncertainties, and
                constants) and performance measures, and each row
                is an experiment.

        Raises:
            DesignExistsError: If scope and design already exist
            TypeError: If not all scope variables are defined in the 
                experiment
        """     


    @abc.abstractmethod
    def read_scope_names(self, design_name=None) -> list:
        """A list of all available scopes in the database.

        Args:
            design_name (str, optional): If a design name, is given, only
                scopes containing a design with this name are returned.

        Returns:
            list
        """

    @abc.abstractmethod
    def read_design_names(self, scope_name:str) -> list:
        """A list of all available designs for a given scope.

        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run
        """

    @abc.abstractmethod
    def read_experiment_id(self, scope_name, *args, **kwargs):
        """
        Read the experiment id previously defined in the database

        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run
            parameters (dict): keys are experiment parameters, values are the
                experimental values to look up.  Subsequent positional or keyword
                arguments are used to update parameters.

        Returns:
            int: the experiment id of the identified experiment

        Raises:
            ValueError: If scope name does not exist
            ValueError: If multiple experiments match an experiment definition.
                This can happen, for example, if the definition is incomplete.
        """


    @abc.abstractmethod
    def read_experiment_ids(self, scope_name, xl_df):
        """
        Read the experiment ids previously defined in the database.

        This method is used to recover the experiment id, if the
        set of parameter values is known but the id of the experiment
        is not known.

        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run
            xl_df (pandas.DataFrame): columns are experiment parameters,
                each row is a full experiment

        Returns:
            list: the experiment id's of the identified experiments

        Raises:
            ValueError: If scope name does not exist
            ValueError: If multiple experiments match an experiment definition.
                This can happen, for example, if the definition is incomplete.
        """


    @abc.abstractmethod
    def read_uncertainties(self, scope_name:str) -> list:
        """A list of all uncertainties for a given scope.

        Args:
            scope_name (str): scope name
        """

    @abc.abstractmethod
    def read_levers(self, scope_name:str) -> list:
        """A list of all levers for a given scope.

        Args:
            scope_name (str): scope name
        """

    @abc.abstractmethod
    def read_constants(self, scope_name:str) -> list:
        """A list of all constants for a given scope.

        Args:
            scope_name (str): scope name
        """

    @abc.abstractmethod
    def read_measures(self, scope_name:str) -> list:
        """A list of all performance measures for a given scope.

        Args:
            scope_name (str): scope name
        """

    @abc.abstractmethod
    def write_metamodel(self, scope_name, metamodel, metamodel_id=None, metamodel_name=''):
        """Store a meta-model in the database

         Args:
            scope_name (str): scope name
            metamodel (emat.MetaModel): The meta-model to be stored.
                If a PythonCoreModel containing a MetaModel is given,
                the MetaModel will be extracted.
            metamodel_id (int, optional): A unique id number for this
                metamodel.  If no id number is given and it cannot be
                inferred from `metamodel`, a unique id number
                will be created.
            metamodel_name (str, optional): A name for this meta-model.
                If no name is given and it cannot be
                inferred from `metamodel`, an empty string is used.
       """


    @abc.abstractmethod
    def read_metamodel(self, scope_name, metamodel_id=None):
        """Retrieve a meta-model from the database.

        Args:
            scope_name (str): scope name
            metamodel_id (int, optional): A unique id number for this
                metamodel.  If not given but there is exactly one
                metamodel stored for the given scope, that metamodel
                will be returned.

        Returns:
            PythonCoreModel: The meta-model, ready to use
        """

    @abc.abstractmethod
    def read_metamodel_ids(self, scope_name):
        """A list of all metamodel id's for a given scope.

        Args:
            scope_name (str): scope name
        """

    @abc.abstractmethod
    def get_new_metamodel_id(self, scope_name):
        """Get a new unused metamodel id for a given scope.

        Args:
            scope_name (str): scope name

        Returns:
            int
        """

    @abc.abstractmethod
    def read_box(self, scope_name: str, box_name: str, scope=None):
        """
        Read a Box from the database.

        Args:
            scope_name (str):
                The name of the scope from which to read the box.
            box_name (str):
                The name of the box to read.
            scope (Scope, optional):
                The Scope to assign to the Box that is returned.
                If not given, no Scope object is assigned to the
                box.

        Returns:
            Box
        """

    @abc.abstractmethod
    def read_box_names(self, scope_name: str):
        """
        Get the names of all boxes associated with a particular scope.

        Args:
            scope_name (str):
                The name of the scope from which to read the Box names.

        Returns:
            list[str]
        """

    @abc.abstractmethod
    def read_box_parent_name(self, scope_name: str, box_name:str):
        """
        Get the name of the parent box for a particular box in the database

        Args:
            scope_name (str):
                The name of the scope from which to read the Box parent.
            box_name (str):
                The name of the box from which to read the parent.

        Returns:
            str or None:
                If the identified box has a parent, this is the name of that
                parent, otherwise None is returned.

        """

    @abc.abstractmethod
    def read_box_parent_names(self, scope_name: str):
        """
        Get the name of the parent box for each box in the database.

        Args:
            scope_name (str):
                The name of the scope from which to read Box parents.

        Returns:
            dict
                A dictionary, with keys giving Box names and values
                giving the respective Box parent names.

        """

    @abc.abstractmethod
    def read_boxes(self, scope_name: str=None, scope=None):
        """
        Read Boxes from the database.

        Args:
            scope_name (str, optional):
                The name of the scope from which to load Boxes. This
                is used exclusively to identify the Boxes to load from
                the database, and the scope by this name is not attached
                to the Boxes, unless `scope` is given, in which case this
                argument is ignored.
            scope (Scope, optional):
                The scope to assign to the Boxes.  If not given,
                no Scope object is assigned.

        Returns:
            Boxes
        """

    @abc.abstractmethod
    def write_box(self, box, scope_name=None):
        """
        Write a single box to the database.

        Args:
            box (Box):
                The Box to write to the database.
            scope_name (str, optional):
                The scope name to use when writing to the database. If
                the `boxes` has a particular scope assigned, the name
                of that scope is used.

        Raises:
            ValueError:
                If the `box` has a particular scope assigned, and
                `scope_name` is given but it is not the same name
                of the assigned scope.

        """

    @abc.abstractmethod
    def write_boxes(self, boxes, scope_name=None):
        """
        Write Boxes to the database.

        Args:
            boxes (Boxes):
                The collection of Boxes to write to the database.
            scope_name (str, optional):
                The scope name to use when writing to the database. If
                the `boxes` has a particular scope assigned, the name
                of that scope is used.

        Raises:
            ValueError:
                If the `boxes` has a particular scope assigned, and
                `scope_name` is given but it is not the same name
                of the assigned scope.

        """

    @abc.abstractmethod
    def new_run_id(
            self,
            scope_name=None,
            parameters=None,
            location=None,
            experiment_id=None,
            source=0,
            **extra_attrs,
    ):
        """
        Create a new run_id in the database.

        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run
            parameters (dict): keys are experiment parameters, values are the
                experimental values to look up.  Subsequent positional or keyword
                arguments are used to update parameters.
            location (str or True, optional): An identifier for this location
                (i.e. this computer).  If set to True, the name of this node
                is found using the `platform` module.
            experiment_id (int, optional): The experiment id associated
                with this run.  If given, the parameters are ignored.
            source (int, default 0): The metamodel_id of the source for this
                run, or 0 for a core model run.

        Returns:
            Tuple[Int,Int]:
                The run_id and experiment_id of the identified experiment

        Raises:
            ValueError: If scope name does not exist
            ValueError: If multiple experiments match an experiment definition.
                This can happen, for example, if the definition is incomplete.
        """

    def info(self, stream=None):
        """
        Print info about scopes and designs in this database.
        """
        if stream is None:
            import sys
            stream = sys.stdout
        print(f"<emat.{self.__class__.__name__}>", file=stream)
        scope_names = self.read_scope_names()
        for scope_name in scope_names:
            print(f"scope: {scope_name}:", file=stream)
            design_names = self.read_design_names(scope_name)
            if design_names:
                print(f"  designs:", file=stream)
                for design_name in design_names:
                    print(f"    - {design_name}", file=stream)
            else:
                print(f"  no designs", file=stream)
