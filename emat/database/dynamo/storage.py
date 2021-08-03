import time
import uuid

import numpy as np
import pandas as pd
import boto3
import secrets
import gzip
from botocore.exceptions import ClientError
from addicty import Dict
from typing import Mapping
from ..kvstore.stores import DictStore, ScopeStore
from ..kvstore.storage import SubkeyStore, Storage
from ..database import Database
from ...exceptions import ReadOnlyDatabaseError
from ...util.deduplicate import reindex_duplicates
from ...util.loggers import get_module_logger
from ...util.uid import uuid_time, uuid6

_logger = get_module_logger(__name__)


def _aws_error_code(error):
    try:
        return error.response['Error']['Code']
    except Exception:
        return None


# class SubkeyStore:
#
#     def __init__(self, value_class, *subkey_names):
#         ###print(f'SubkeyStore.__init__({", ".join(str(i) for i in subkey_names)})')
#         self._value_class = value_class
#         self.keydir = "/".join(str(i) for i in subkey_names)
#
#     def __set_name__(self, owner, name):
#         # self : SubkeyStore
#         # owner : parent class that will have `self` as a member
#         # name : the name of the attribute that `self` will be
#         ###print(f'SubkeyStore.__set_name__({owner!r}, {name!r})')
#         self.public_name = name
#         self.private_name = '_subkey_' + name
#         if not self.keydir:
#             self.keydir = self.public_name
#
#     def __get__(self, obj, objtype=None):
#         # self : SubkeyStore
#         # obj : instance of parent class that has `self` as a member, or None
#         # objtype : class of `obj`
#         result = getattr(obj, self.private_name, None)
#         if result is None:
#             self.__set__(obj, None)
#             result = getattr(obj, self.private_name, None)
#         result.parent = obj
#         return result
#
#     def __set__(self, obj, value):
#         # self : SubkeyStore
#         # obj : instance of parent class that has `self` as a member
#         # value : the new value that is trying to be assigned
#         if not (isinstance(value, Mapping) or value is None):
#             raise TypeError(f"SubkeyStore must be Mapping not {type(value)}")
#         ###print(f"__set__ {obj}, {self.private_name}, {value}")
#         if value is None:
#             value = {}
#         x = self._value_class(obj, self.keydir)
#         for k, v in value.items():
#             x[k] = v
#         x.parent = obj
#         setattr(obj, self.private_name, x)
#
#     def __delete__(self, obj):
#         # self : SubkeyStore
#         # obj : instance of parent class that has `self` as a member
#         self.__set__(obj, None)


class DynamoDB(Database):


    def init_scopes(self, tablename='emat_scopes'):
        """
        Initialize the experiments table in DynamoDB.
        """
        self.scopes_tablename = tablename
        try:
            response = self._dynamo_client.create_table(
                AttributeDefinitions=[
                    {
                        'AttributeName': 'scope_name',
                        'AttributeType': 'S',
                    },
                ],
                KeySchema=[
                    {
                        'AttributeName': 'scope_name',
                        'KeyType': 'HASH',
                    },
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5,
                },
                TableName=tablename,
            )

        except ClientError as error:
            if _aws_error_code(error) != 'ResourceInUseException':
                raise

    def init_designs(self, tablename='emat_designs'):
        """
        Initialize the experiments table in DynamoDB.
        """
        self.designs_tablename = tablename
        try:
            response = self._dynamo_client.create_table(
                AttributeDefinitions=[
                    {
                        'AttributeName': 'scope_name',
                        'AttributeType': 'S',
                    },
                    {
                        'AttributeName': 'design_name',
                        'AttributeType': 'S',
                    },
                ],
                KeySchema=[
                    {
                        'AttributeName': 'scope_name',
                        'KeyType': 'HASH',
                    },
                    {
                        'AttributeName': 'design_name',
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
            if _aws_error_code(error) != 'ResourceInUseException':
                raise


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
            if _aws_error_code(error) != 'ResourceInUseException':
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
        experiment_id = int(experiment_id)
        if experiment_id <= 0:
            raise ValueError("experiment_id must be a positive integer")

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
            errcode = _aws_error_code(error)
            if errcode == 'ConditionalCheckFailedException':
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

    def _put_scope(self, scope, scope_name=None):
        if scope_name is None:
            scope_name = scope.name

        x = {
            'scope_name': {'S': str(scope_name)},
            'content': {'B': gzip.compress(scope.dump().encode())},
        }
        x.update(self._dynamo_experiment_key(scope_name, 0))
        response = self._dynamo_client.put_item(
            TableName=self.scopes_tablename,
            Item=x,
        )
        return response

    def _get_scope(self, scope_name):
        from .serialization import TypeDeserializer
        key = {
            'scope_name': {'S': str(scope_name)},
        }
        x = self._dynamo_client.get_item(
            TableName=self.scopes_tablename,
            Key=key,
        ).get('Item')
        try:
            content = TypeDeserializer().deserialize({'M':x})['content']
            from ...scope.scope import Scope
            return Scope(None, gzip.decompress(bytes(content)).decode())
        except Exception as err:
            _logger.exception(str(err))
            return x

    def _put_design(self, scope_name, design_name, experiment_ids):
        x = {
            'scope_name': str(scope_name),
            'design_name': str(design_name),
            'experiment_ids': set(experiment_ids),
        }
        from .serialization import TypeSerializer
        x = TypeSerializer().serialize(x)['M']
        response = self._dynamo_client.put_item(
            TableName=self.designs_tablename,
            Item=x,
        )
        return response

    def _get_design(self, scope_name, design_name):
        """

        Parameters
        ----------
        scope_name : str
        design_name : str

        Returns
        -------
        set
            The experiment id's in this design
        """
        x = {
            'scope_name': str(scope_name),
            'design_name': str(design_name),
        }
        from .serialization import TypeSerializer
        x = TypeSerializer().serialize(x)['M']
        response = self._dynamo_client.get_item(
            TableName=self.designs_tablename,
            Key=x,
        )
        try:
            content = response['Item']['experiment_ids']
        except KeyError:
            raise KeyError(f"design_name {design_name} not found")
        else:
            from .serialization import TypeDeserializer
            return TypeDeserializer().deserialize(content)


    def init_results(self, tablename='emat_experiment_results'):
        """
        Initialize the experiment results table in DynamoDB.
        """
        self.experiment_results_tablename = tablename
        try:
            response = self._dynamo_client.create_table(
                AttributeDefinitions=[
                    {
                        'AttributeName': 'scope_name',
                        'AttributeType': 'S',
                    },
                    {
                        'AttributeName': 'ex_run_id',
                        'AttributeType': 'B',
                    },
                ],
                KeySchema=[
                    {
                        'AttributeName': 'scope_name',
                        'KeyType': 'HASH',
                    },
                    {
                        'AttributeName': 'ex_run_id',
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
            if _aws_error_code(error) != 'ResourceInUseException':
                raise

    def _dynamo_experiment_result_key(self, scope_name, experiment_id, run_id):
        if isinstance(run_id, uuid.UUID):
            run_id_bytes = run_id.bytes
        else:
            run_id_bytes = bytes(run_id)
        return {
            'scope_name': {'S': scope_name},
            'ex_run_id': {'B': int(experiment_id).to_bytes(4, 'big') + run_id_bytes},
        }

    def _put_experiment_result(self, scope_name, experiment_id, run_id, results, overwrite=False, bucket=None):
        """
        Write one experiment run results to the database.

        Parameters
        ----------
        scope_name : str
        experiment_id : int
        run_id : uuid.UUID
        results : Mapping
        overwrite : bool, default False
        bucket : str, optional
            Also backup results storage to s3 in this bucket
        """
        from .serialization import TypeSerializer

        if bucket is not None:
            try:
                s3_filename = f"{scope_name}/{experiment_id}/{run_id}"
                Dict(results).dump(f"s3://{bucket}/results/{s3_filename[:1014]}")
            except Exception:
                _logger.exception("FAILED TO STORE BACKUP DATA IN S3")

        x = TypeSerializer().serialize(results)['M']
        if overwrite:
            x.update(self._dynamo_experiment_result_key(scope_name, experiment_id, run_id))
            try:
                kwds = dict(
                    TableName=self.experiment_results_tablename,
                    Item=x,
                )
                if not overwrite:
                    kwds['ConditionExpression'] = "attribute_not_exists(run_id)"
                response = self._dynamo_client.put_item(**kwds)
            except Exception as error:
                errcode = _aws_error_code(error)
                if errcode == 'ConditionalCheckFailedException':
                    raise KeyError(f"experiment_id {experiment_id} "
                                   f"run_id {run_id} already exists "
                                   f"for scope {scope_name}")
                else:
                    _logger.exception(str(error))
                    raise
            else:
                return response
        else:
            key = self._dynamo_experiment_result_key(scope_name, experiment_id, run_id)
            expr_names = {}
            expr_values = {}
            update_str = []
            for n, (ik, iv) in enumerate(x.items()):
                expr_names[f"#AK{n}"] = ik
                expr_values[f":av{n}"] = iv
                update_str.append(f"#AK{n} = :av{n}")
            kwds = dict(
                TableName=self.experiment_results_tablename,
                Key=key,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                UpdateExpression="SET "+", ".join(update_str)
            )
            response = self._dynamo_client.update_item(**kwds)

    def _get_experiment_result(self, scope_name, experiment_id, run_id):
        from .serialization import TypeDeserializer
        key = self._dynamo_experiment_result_key(scope_name, experiment_id, run_id)
        x = self._dynamo_client.get_item(
            TableName=self.experiment_results_tablename,
            Key=key,
        ).get('Item')
        try:
            result = TypeDeserializer().deserialize({'M':x})
            result.pop('scope_name')
            result.pop('ex_run_id')
            return result
        except Exception as err:
            _logger.exception(str(err))
            return x


    def __init__(self, local_port=8123, retries=10, *, bucket=None):

        from botocore.config import Config

        boto_client_kwds = dict(
            config=Config(
                retries={
                    'max_attempts': retries,
                    'mode': 'standard',
                },
            ),
        )
        if local_port is not None:
            boto_client_kwds['endpoint_url'] = f'http://localhost:{local_port}'

        self._dynamo_client = boto3.client('dynamodb', **boto_client_kwds)

        super().__init__()
        self.init_scopes()
        self.init_designs()
        self.init_experiments()
        self.init_results()
        self.bucket = bucket

    def _write_scope(self, scope_name, sheet, scp_xl, scp_m, content):
        raise NotImplementedError

    def store_scope(self, scope):
        if self.readonly:
            raise ReadOnlyDatabaseError
        from ...scope.scope import Scope
        if not isinstance(scope, Scope):
            raise TypeError(f"scope must be emat.Scope not {type(scope)}")
        self._put_scope(scope)

    def update_scope(self, scope):
        self.store_scope(scope)

    def add_scope_meas(self, scope_name, scp_m):
        raise NotImplementedError

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

        scope_names = []

        from .serialization import TypeDeserializer
        deserialize = TypeDeserializer().deserialize

        scan_args = dict(
            TableName=self.scopes_tablename,
        )

        while True:
            # get next page
            response = self._dynamo_client.scan(**scan_args)
            # handle returned experiments
            for i in response.get('Items',[]):
                x = deserialize({'M':i})
                s = x.pop("scope_name")
                scope_names.append(s)
            LastEvaluatedKey = response.get("LastEvaluatedKey", None)
            if LastEvaluatedKey:
                scan_args['ExclusiveStartKey'] = LastEvaluatedKey
            else:
                break
        return scope_names

    def read_scope(self, scope_name=None):
        if scope_name is None:
            scope_names = self.read_scope_names()
            if len(scope_names) == 1:
                scope_name = scope_names[0]
            elif len(scope_names) == 0:
                raise ValueError("no scopes are stored")
            else:
                raise ValueError("must give scope_name when more than one scope is stored")
        return self._get_scope(scope_name)

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
        if self.readonly:
            raise ReadOnlyDatabaseError
        #scope_name = self._validate_scope(scope_name, 'design_name')
        if experiment_id is None:
            raise NotImplementedError("experiment_id cannot be None")
            # if parameters is None:
            #     raise ValueError('must give experiment_id or parameters')
            # experiment_id = self.get_experiment_id(scope_name, parameters)
        run_id = uuid6()
        if location is True:
            import platform
            location = platform.node()
        attrs = dict(
            run_location=location,
            run_source=source,
        )
        self._put_experiment_result(scope_name, experiment_id, run_id, attrs, bucket=self.bucket)
        return run_id, experiment_id

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

        design_names = []

        from .serialization import TypeDeserializer
        deserialize = TypeDeserializer().deserialize

        query_args = dict(
            TableName=self.designs_tablename,
            KeyConditionExpression="scope_name = :scope_name",
            ExpressionAttributeValues={":scope_name": {"S": scope_name}, },
        )

        while True:
            # get next page
            response = self._dynamo_client.query(**query_args)
            # handle returned experiments
            for i in response.get('Items',[]):
                x = deserialize({'M':i})
                design_names.append(x.pop("design_name"))
            LastEvaluatedKey = response.get("LastEvaluatedKey", None)
            if LastEvaluatedKey:
                query_args['ExclusiveStartKey'] = LastEvaluatedKey
            else:
                break
        return design_names



        # prefix = f"design/{scope_name}/"
        # response = self.client.list_objects_v2(
        #     Bucket=self.bucket,
        #     Prefix=prefix,
        # )
        # if response.get('Contents'):
        #     return [
        #         i.get('Key').replace(prefix, "")
        #         for i in response.get('Contents')
        #     ]
        # else:
        #     return []

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
        """
        Read experiment results from the database.

        Parameters
        ----------
        scope_name : str or Scope
            A scope or just its name, used to identify experiments,
            performance measures, and results associated with this
            exploratory analysis.
        design_name : str, optional
            If given, only experiments associated with both the scope and the
            named design are returned, otherwise all experiments associated
            with the scope are returned.
        experiment_id : int, optional
            The id of the experiment to retrieve.  If omitted, get all
            experiments matching the named scope and design.
        source : int, optional
            The source identifier of the experimental outcomes to load.  If
            not given, but there are only results from a single source in the
            database, those results are returned.  If there are results from
            multiple sources, an error is raised.
        design : str
            Deprecated, use `design_name`.
        runs : {None, 'all', 'valid', 'invalid'}, default None
            By default, this method fails if there is more than one valid
            model run matching the given `design_name` and `source` (if any)
            for any experiment.  Set this to 'valid' or 'invalid' to get all
            valid or invalid model runs (instead of raising an exception).
            Set to 'all' to get everything, including both valid and
            invalidated results.
        formulas : bool, default True
            If the scope includes formulaic measures (computed directly from
            other measures) then compute these values and include them in
            the results.

        Returns
        -------
        results : pandas.DataFrame
            performance measures

        Raises
        ------
        ValueError
            When the database contains multiple sets of results
            matching the given `design_name` and/or `source`
            (if any) for any experiment.
        """

        if not isinstance(scope_name, str):
            scope_name = scope_name.name

        if design_name is not None:
            # TODO: this is not very efficient, pinging the Dynamo for every row.
            # maybe batch it?
            ex_ids = self._get_design(scope_name, design_name)
            measures = []
            for ex_id in ex_ids:
                measures.append(self.read_experiment_measures(
                    scope_name,
                    experiment_id=ex_id,
                    source=source,
                    runs=runs,
                    formulas=formulas,
                    with_validity=with_validity,
                ))
            return pd.concat(measures)

        if source is not None:
            raise NotImplementedError

        experiment_list = []

        from .serialization import TypeDeserializer
        deserialize = TypeDeserializer().deserialize

        query_args = dict(
            TableName=self.experiment_results_tablename,
            KeyConditionExpression="scope_name = :scope_name",
            ExpressionAttributeValues={":scope_name": {"S": scope_name}},
        )

        if runs == 'valid':
            query_args['FilterExpression'] = "attribute_not_exists(invalidated)"
        elif runs == 'invalid':
            query_args['FilterExpression'] = "attribute_exists(invalidated)"

        if experiment_id is not None:
            key_cond = query_args['KeyConditionExpression']
            key_cond = key_cond + " AND ex_run_id BETWEEN :ex_id_low AND :ex_id_high"
            query_args['KeyConditionExpression'] = key_cond
            attr_vals = {}
            attr_vals.update(query_args['ExpressionAttributeValues'])
            experiment_id_bytes = int(experiment_id).to_bytes(4, 'big')
            attr_vals[":ex_id_low"] = {'B': experiment_id_bytes + bytes(16)}
            attr_vals[":ex_id_high"] = {'B': experiment_id_bytes + b'\xFF'*16}
            query_args['ExpressionAttributeValues'] = attr_vals

        while True:
            # get next page
            response = self._dynamo_client.query(**query_args)
            # handle returned experiments
            for i in response.get('Items',[]):
                x = deserialize({'M':i})
                x.pop("scope_name")
                x_ids = bytes(x.pop("ex_run_id"))
                x['experiment_id'] = int(x_ids[:4].hex(), 16)
                x['run_id'] = uuid.UUID(bytes=x_ids[4:])
                # 'ex_run_id': {'B': experiment_id.to_bytes(4, 'big') + run_id_bytes}
                experiment_list.append(x)
            LastEvaluatedKey = response.get("LastEvaluatedKey", None)
            if LastEvaluatedKey:
                query_args['ExclusiveStartKey'] = LastEvaluatedKey
            else:
                break

        from ...experiment.experimental_design import ExperimentalDesign
        if experiment_list:

            scope = self.read_scope(scope_name)
            column_order = scope.get_measure_names()
            xl_df = pd.DataFrame(experiment_list)
            xl_df = xl_df.set_index(["experiment_id", "run_id"])
            result = ExperimentalDesign(xl_df[[i for i in column_order if i in xl_df.columns]])
            result.design_name = design_name

        else:
            result = ExperimentalDesign()
            result.design_name = design_name

        if runs is None:
            df = result.reset_index()
            df["_run_timestamp_"] = df[["run_id"]].applymap(uuid_time)
            df = df.sort_values(by="_run_timestamp_")
            df = df.drop_duplicates(subset="experiment_id", keep='last')
            df = df.set_index(['experiment_id', 'run_id'])
            df = df.drop(columns=['_run_timestamp_'])
            result = ExperimentalDesign(df)
            result.design_name = design_name

        return result

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
            KeyConditionExpression="scope_name = :scope_name AND experiment_id > :zero",
            ExpressionAttributeValues={":scope_name": {"S": scope_name}, ":zero": {"N": "0"}},
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
            m_df (pandas.DataFrame or dict):
                The columns of this DataFrame are the performance
                measure names, and row indexes are the experiment id's.
                If given as a `dict` instead of a DataFrame, the keys are
                treated as columns and an `experiment_id` must be provided.
            run_ids (pandas.Index, optional):
                Provide an optional index of universally unique run ids
                (UUIDs) for these results. The UUIDs can be used to help
                identify problems and organize model runs.
            experiment_id (int, optional):
                Provide an experiment_id.  This is only used if the
                `m_df` is provided as a dict instead of a DataFrame

        Raises:
            UserWarning: If scope name does not exist
        """
        if self.readonly:
            raise ReadOnlyDatabaseError

        if experiment_id is not None:
            if not isinstance(m_df, dict):
                raise ValueError("only give an experiment_id with a dict as `m_df`")
            m_df = pd.DataFrame(m_df, index=[experiment_id])

        # split run_ids from multiindex
        if m_df.index.nlevels == 2:
            if run_ids is not None:
                raise ValueError('run_ids cannot be given when they are embedded in m_df.index')
            run_ids = m_df.index.get_level_values(1)
            m_df.index = m_df.index.get_level_values(0)

        if run_ids is None:
            # generate new run_ids if none is given
            run_ids = []
            for experiment_id in m_df.index:
                run_ids.append(uuid6())

        scope = self.read_scope(scope_name)
        scp_m = scope.get_measure_names()

        for measure_name in scp_m:
            dataseries = None
            if measure_name not in m_df.columns:
                formula = getattr(scope[measure_name], 'formula', None)
                if formula:
                    m_df[measure_name] = m_df.eval(formula).rename(measure_name)

        for run_id, (ex_id, row) in zip(run_ids, m_df.iterrows()):
            _logger.critical(f"put {ex_id}.{run_id} results")
            self._put_experiment_result(scope_name, ex_id, run_id, dict(**row), bucket=self.bucket)


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
        from .serialization import TypeSerializer, TypeDeserializer
        combined_experiments = combined_experiments.applymap(
            lambda x: TypeDeserializer().deserialize(TypeSerializer().serialize(x))
        )
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
        _logger.critical("write experiment_ids to dynamo designs")
        self._put_design(scope_name, design_name, ex_ids)

        for ex_id_as_input, row in novel_experiments.iterrows():
            _logger.critical(f"put {ex_id_as_input}")
            self._put_experiment(scope_name, ex_id_as_input, dict(**row))

        return ex_ids

    def write_metamodel(self, scope_name, metamodel, metamodel_id=None, metamodel_name=''):
        raise NotImplementedError


