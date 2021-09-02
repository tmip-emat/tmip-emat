import numpy as np
import pandas as pd
import boto3
from addicty import Dict
from typing import Mapping
from .stores import DictStore, ScopeStore
from ..database import Database
from ...exceptions import ReadOnlyDatabaseError
from ...util.deduplicate import reindex_duplicates
from ...util.loggers import get_module_logger

_logger = get_module_logger(__name__)



class SubkeyStore:

    def __init__(self, value_class, *subkey_names):
        ###print(f'SubkeyStore.__init__({", ".join(str(i) for i in subkey_names)})')
        self._value_class = value_class
        self.keydir = "/".join(str(i) for i in subkey_names)

    def __set_name__(self, owner, name):
        # self : SubkeyStore
        # owner : parent class that will have `self` as a member
        # name : the name of the attribute that `self` will be
        ###print(f'SubkeyStore.__set_name__({owner!r}, {name!r})')
        self.public_name = name
        self.private_name = '_subkey_' + name
        if not self.keydir:
            self.keydir = self.public_name

    def __get__(self, obj, objtype=None):
        # self : SubkeyStore
        # obj : instance of parent class that has `self` as a member, or None
        # objtype : class of `obj`
        result = getattr(obj, self.private_name, None)
        if result is None:
            self.__set__(obj, None)
            result = getattr(obj, self.private_name, None)
        result.parent = obj
        return result

    def __set__(self, obj, value):
        # self : SubkeyStore
        # obj : instance of parent class that has `self` as a member
        # value : the new value that is trying to be assigned
        if not (isinstance(value, Mapping) or value is None):
            raise TypeError(f"SubkeyStore must be Mapping not {type(value)}")
        ###print(f"__set__ {obj}, {self.private_name}, {value}")
        if value is None:
            value = {}
        x = self._value_class(obj, self.keydir)
        for k, v in value.items():
            x[k] = v
        x.parent = obj
        setattr(obj, self.private_name, x)

    def __delete__(self, obj):
        # self : SubkeyStore
        # obj : instance of parent class that has `self` as a member
        self.__set__(obj, None)


class Storage(Database):

    scope = SubkeyStore(ScopeStore)
    experiment = SubkeyStore(DictStore)
    design = SubkeyStore(DictStore)

    def __init__(self, bucket):
        self.bucket = bucket
        self.client = boto3.client('s3')
        super().__init__(readonly=False)

    def read_scope_names(self, design_name=None):
        """
        A list of all available scopes in the database.

        Args:
            design_name (str, optional): If a design name, is given, only
                scopes containing a design with this name are returned.

        Returns:
            list
        """
        if design_name is not None:
            raise NotImplementedError("cannot filter scopes in key-value storage")
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix='scope/',
        )
        if response.get('Contents'):
            return [
                i.get('Key').replace("scope/", "")
                for i in response.get('Contents')
            ]
        else:
            return []

    def _write_scope(self, scope_name, sheet, scp_xl, scp_m, content):
        raise NotImplementedError

    def add_scope_meas(self, scope_name, scp_m):
        raise NotImplementedError

    def delete_experiment_measures(self, experiment_ids=None):
        raise NotImplementedError

    def delete_experiments(self, scope_name, design_name=None, design=None):
        raise NotImplementedError

    def delete_scope(self, scope_name):
        raise NotImplementedError

    def get_new_metamodel_id(self, scope_name):
        raise NotImplementedError

    def init_xlm(self, parameter_list, measure_list):
        raise NotImplementedError

    def new_run_id(
            self,
            scope_name=None,
            parameters=None,
            location=None,
            experiment_id=None,
            source=0,
            **extra_attrs,
    ):
        raise NotImplementedError

    def read_box(self, scope_name, box_name, scope=None):
        raise NotImplementedError

    def read_box_names(self, scope_name):
        raise NotImplementedError

    def read_box_parent_name(self, scope_name: str, box_name:str):
        raise NotImplementedError

    def read_box_parent_names(self, scope_name: str):
        raise NotImplementedError

    def read_boxes(self, scope_name: str=None, scope=None):
        raise NotImplementedError

    def read_constants(self, scope_name:str):
        raise NotImplementedError

    def read_design_names(self, scope_name):
        """
        A list of all available designs for a given scope.

        Parameters
        ----------
        scope_name : str
            The scope name used to identify experiments, performance measures,
            and results.

        Returns
        -------
        list
        """
        prefix = f"design/{scope_name}/"
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )
        if response.get('Contents'):
            return [
                i.get('Key').replace(prefix, "")
                for i in response.get('Contents')
            ]
        else:
            return []

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
        raise NotImplementedError

    def read_experiment_id(self, scope_name, *args, **kwargs):
        raise NotImplementedError

    def read_experiment_ids(self, scope_name, xl_df):
        raise NotImplementedError

    def read_experiment_measure_sources(
            self,
            scope_name,
            design_name=None,
            experiment_id=None,
            design=None,
    ):
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def read_levers(self, scope_name):
        raise NotImplementedError

    def read_measures(self, scope_name):
        raise NotImplementedError

    def read_metamodel(self):
        raise NotImplementedError

    def read_metamodel_ids(self):
        raise NotImplementedError

    def read_scope(self, scope_name=None):
        if scope_name is None:
            scope_names = self.read_scope_names()
            if len(scope_names) == 1:
                scope_name = scope_names[0]
            elif len(scope_names) == 0:
                raise ValueError("no scopes are stored")
            else:
                raise ValueError("must give scope_name when more than one scope is stored")
        return self.scope[scope_name]

    def read_uncertainties(self, scope_name):
        raise NotImplementedError

    def store_scope(self, scope):
        if self.readonly:
            raise ReadOnlyDatabaseError
        from ...scope.scope import Scope
        if not isinstance(scope, Scope):
            raise TypeError(f"scope must be emat.Scope not {type(scope)}")
        self.scope[scope.name] = scope

    def update_scope(self, scope):
        self.store_scope(scope)

    def write_box(self):
        raise NotImplementedError

    def write_boxes(self):
        raise NotImplementedError

    def write_experiment_all(self):
        raise NotImplementedError

    def write_experiment_measures(
            self,
            scope_name,
            source,
            m_df,
            run_ids=None,
            experiment_id=None,
    ):
        raise NotImplementedError

    def write_experiment_parameters(
            self,
            scope_name,
            design_name,
            xl_df,
            force_ids=False,
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
            xl_df (pandas.DataFrame):
                The columns of this DataFrame are the experiment
                parameters (i.e. policy levers, uncertainties, and
                constants), and each row is an experiment.
            force_ids (bool, default False):
                For the experiment id's saved into the database to match
                the id's in the index of `xl_df`, or raise an error if
                this cannot be completed, either because that id is in
                use for a different experiment, or because this experiment
                is already saved with a different id.

        Returns:
            list: the experiment id's of the newly recorded experiments

        Raises:
            UserWarning: If scope name does not exist
            TypeError: If not all scope variables are defined in the
                exp_def
            ValueError: If `force_ids` is True but the same experiment
                already has a different id.
            sqlite3.IntegrityError: If `force_ids` is True but a different
                experiment is already using the given id.
        """
        if self.readonly:
            raise ReadOnlyDatabaseError
        if design_name is None:
            design_name = 'ad hoc'

        if isinstance(design_name, dict):
            design_name_map = {}
            for k,v in design_name.items():
                if isinstance(v, str):
                    from ...util.seq_grouping import seq_int_group_expander
                    design_name_map[k] = seq_int_group_expander(v)
                else:
                    design_name_map[k] = v
        else:
            design_name_map = {design_name: xl_df.index}

        #scope_name = self._validate_scope(scope_name, 'design_name')


        # get list of experiment variables - except "one"
        # scp_xl = fcur.execute(sq.GET_SCOPE_XL, [scope_name]).fetchall()
        # if len(scp_xl) == 0:
        #     raise UserWarning('named scope {0} not found - experiments will \
        #                           not be recorded'.format(scope_name))

        ### split experiments into novel and duplicate ###
        # first join to existing experiments
        existing_experiments = self.read_experiment_parameters(scope_name, None)
        combined_experiments = pd.concat([
            existing_experiments,
            xl_df.set_index(np.full(len(xl_df), -1, dtype=int)),
        ])
        combined_experiments_reindexed = reindex_duplicates(combined_experiments)
        xl_df_ = combined_experiments_reindexed.iloc[-len(xl_df):]
        novel_flag = xl_df_.index.isin([-1])
        novel_experiments = xl_df.loc[novel_flag]
        duplicate_experiments = xl_df_.loc[~novel_flag]

        ex_ids = []
        ex_ids.extend(novel_experiments.index)
        ex_ids.extend(duplicate_experiments.index)

        # write experiment id's to S3
        # using the id's as provided in the experiments dataframe
        design_experiments = Dict(experiment_ids=ex_ids)
        design_experiments.dump(f"s3://{self.bucket}/design/{scope_name}/{design_name}")

        for ex_id_as_input, row in novel_experiments.iterrows():
            Dict(**row).dump(f"s3://{self.bucket}/experiment/{scope_name}/{ex_id_as_input}")

        return ex_ids

    def write_metamodel(self, scope_name, metamodel, metamodel_id=None, metamodel_name=''):
        raise NotImplementedError


