import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from addicty import Dict
from typing import Mapping
from ..kvstore.stores import DictStore, ScopeStore
from ..kvstore.storage import SubkeyStore, Storage
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


class DynamoDB(Storage):

    # scope = SubkeyStore(ScopeStore)
    # experiment = SubkeyStore(DictStore)
    # design = SubkeyStore(DictStore)

    def init_experiments(self, tablename='emat_experiments'):
        """
        Initialize the experiments table in DynamoDB.
        """
        self.experiments_tablename = tablename
        try:
            response = self._dynamo_client.create_table(
                AttributeDefinitions=[
                    {
                        'AttributeName': 'scope_name',
                        'AttributeType': 'S',
                    },
                    {
                        'AttributeName': 'experiment_id',
                        'AttributeType': 'N',
                    },
                ],
                KeySchema=[
                    {
                        'AttributeName': 'scope_name',
                        'KeyType': 'HASH',
                    },
                    {
                        'AttributeName': 'experiment_id',
                        'KeyType': 'RANGE',
                    },
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5,
                },
                TableName=tablename,
            )

        except ClientError as error:
            if error.response['Error']['Code'] != 'ResourceInUseException':
                raise


    def _dynamo_experiment_key(self, scope_name, experiment_id):
        return {
            'scope_name': {'S': str(scope_name)},
            'experiment_id': {'N': str(experiment_id)},
        }

    def _put_experiment(self, scope_name, experiment_id, experiment):
        """
        Write one experiment to the database.

        Parameters
        ----------
        scope_name : str
        experiment_id : int
        experiment : Mapping
        """
        from .serialization import TypeSerializer
        x = TypeSerializer().serialize(experiment)['M']
        x.update(self._dynamo_experiment_key(scope_name, experiment_id))
        try:
            response = self._dynamo_client.put_item(
                TableName=self.experiments_tablename,
                Item=x,
                ConditionExpression="attribute_not_exists(experiment_id)",
            )
        except Exception as error:
            if error.response['Error']['Code'] == 'ConditionalCheckFailedException':
                raise KeyError(f"experiment_id {experiment_id} already exists "
                               f"for scope {scope_name}")
            else:
                _logger.exception(str(error))
                raise
        else:
            return response

    def _get_experiment(self, scope_name, experiment_id):
        from .serialization import TypeDeserializer
        key = self._dynamo_experiment_key(scope_name, experiment_id)
        x = self._dynamo_client.get_item(
            TableName=self.experiments_tablename,
            Key=key,
        ).get('Item')
        try:
            experiment = TypeDeserializer().deserialize({'M':x})
            experiment.pop('scope_name')
            experiment.pop('experiment_id')
            return experiment
        except Exception as err:
            _logger.exception(str(err))
            return x

    def _check_max_experiment_id(self, scope_name):
        """
        Get the current maximum experiment number.

        Parameters
        ----------
        scope_name : str

        Returns
        -------
        int
        """
        max_id = 0
        response = self._dynamo_client.query(
            TableName=self.experiments_tablename,
            Limit=1,
            ScanIndexForward=False,
            KeyConditionExpression="scope_name = :scope_name",
            ExpressionAttributeValues={":scope_name": {"S": scope_name}},
        )
        items = response.get('Items', [])
        if items:
            from .serialization import TypeDeserializer
            max_id = TypeDeserializer().deserialize(items[0].get('experiment_id'))
        return max_id


    def __init__(self, bucket, local_port=8123):
        if local_port is not None:
            self._dynamo_client = boto3.client('dynamodb', endpoint_url=f'http://localhost:{local_port}')
        else:
            self._dynamo_client = boto3.client('dynamodb')

        super().__init__(bucket=bucket)
        self.init_experiments()

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

    def read_experiment_measures(self):
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
        """
        Read experiment definitions from the database.

        Read the values for each experiment parameter per experiment.

        Parameters
        ----------
        scope_name : str
            A scope name, used to identify experiments,
            performance measures, and results associated with this
            exploratory analysis.
        design_name : str, optional
            Not implemented.
        only_pending : bool
            Not implemented.
        design : str
            Not implemented.
        experiment_ids : Collection, optional
            Not implemented.
        ensure_dtypes : bool, default True
            If True, the scope associated with these experiments is also read
            out of the database, and that scope file is used to format
            experimental data consistently (i.e., as float, integer, bool, or
            categorical).

        Returns
        -------
        emat.ExperimentalDesign:
            The experiment parameters are returned in a subclass
            of a normal pandas.DataFrame, which allows attaching
            the `design_name` as meta-data to the DataFrame.
        """

        if design_name is not None:
            raise NotImplementedError

        if experiment_ids is not None:
            raise NotImplementedError

        if only_pending:
            raise NotImplementedError

        experiment_list = []

        from .serialization import TypeDeserializer
        deserialize = TypeDeserializer().deserialize

        query_args = dict(
            TableName=self.experiments_tablename,
            KeyConditionExpression="scope_name = :scope_name",
            ExpressionAttributeValues={":scope_name": {"S": scope_name}},
        )

        while True:
            # get next page
            response = self._dynamo_client.query(**query_args)
            # handle returned experiments
            for i in response.get('Items',[]):
                x = deserialize({'M':i})
                x.pop("scope_name")
                experiment_list.append(x)
            LastEvaluatedKey = response.get("LastEvaluatedKey", None)
            if LastEvaluatedKey:
                query_args['ExclusiveStartKey'] = LastEvaluatedKey
            else:
                break

        from ...experiment.experimental_design import ExperimentalDesign
        if experiment_list:

            scope = self.read_scope(scope_name)
            column_order = (
                    scope.get_constant_names()
                    + scope.get_uncertainty_names()
                    + scope.get_lever_names()
            )
            xl_df = pd.DataFrame(experiment_list).set_index("experiment_id")
            result = ExperimentalDesign(xl_df[[i for i in column_order if i in xl_df.columns]])
            result.design_name = design_name
            if ensure_dtypes:
                scope = self.read_scope(scope_name)
                result = scope.ensure_dtypes(result)

        else:
            result = ExperimentalDesign()
            result.design_name = design_name

        return result


    def read_levers(self, scope_name):
        raise NotImplementedError

    def read_measures(self, scope_name):
        raise NotImplementedError

    def read_metamodel(self):
        raise NotImplementedError

    def read_metamodel_ids(self):
        raise NotImplementedError

    def read_uncertainties(self, scope_name):
        raise NotImplementedError

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

        ### split experiments into novel and duplicate ###
        # first join to existing experiments
        _logger.critical("read_experiment_parameters")
        existing_experiments = self.read_experiment_parameters(scope_name)
        combined_experiments = pd.concat([
            existing_experiments,
            xl_df.set_index(np.full(len(xl_df), -1, dtype=int)),
        ])
        _logger.critical("reindex_duplicates")
        combined_experiments_reindexed = reindex_duplicates(combined_experiments)
        xl_df_ = combined_experiments_reindexed.iloc[-len(xl_df):]
        novel_flag = xl_df_.index.isin([-1])
        novel_experiments = xl_df.loc[novel_flag]
        duplicate_experiments = xl_df_.loc[~novel_flag]

        _logger.critical("novel_ids")
        novel_id_start = self._check_max_experiment_id(scope_name)+1
        novel_ids = pd.RangeIndex(novel_id_start, novel_id_start+len(novel_experiments))
        novel_experiments.index = novel_ids

        ex_ids = []
        ex_ids.extend(novel_experiments.index)
        ex_ids.extend(duplicate_experiments.index)

        # write experiment id's to S3
        # using the id's as provided in the experiments dataframe
        _logger.critical("write experiment_ids to s3")
        design_experiments = Dict(experiment_ids=ex_ids)
        design_experiments.dump(f"s3://{self.bucket}/design/{scope_name}/{design_name}")

        for ex_id_as_input, row in novel_experiments.iterrows():
            _logger.critical(f"put {ex_id_as_input}")
            self._put_experiment(scope_name, ex_id_as_input, dict(**row))

        return ex_ids

    def write_metamodel(self, scope_name, metamodel, metamodel_id=None, metamodel_name=''):
        raise NotImplementedError


