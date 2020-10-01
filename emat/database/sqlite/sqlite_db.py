"""sqlite_db:
    Methods for creating and deleting a sqlite3 database for emat.
    A Sqlite3 database is a single file.
    The class knows the set of sql files needed to create the necessary tables
"""

import os
from typing import List
import sqlite3
import atexit
import pandas as pd
import warnings
from typing import AbstractSet
import numpy as np
import uuid

from . import sql_queries as sq
from ..database import Database
from ...util.deduplicate import reindex_duplicates
from ...exceptions import DatabaseVersionWarning, DatabaseVersionError

from ...util.loggers import get_module_logger
_logger = get_module_logger(__name__)
import logging

from ...util.docstrings import copydoc

class SQLiteDB(Database):
    """
    SQLite implementation of the :class:`Database` abstract base class.

    Args:
        database_path (str, optional): file path and name of database file
            If not given, a database is initialized in-memory.
        initialize (bool or 'skip', default False):
            Whether to initialize emat database file.  The value of this argument
            is ignored if `database_path` is not given (as in-memory databases
            must always be initialized).  If given as 'skip' then no setup
            scripts are run, and it is assumed that all relevant tables already
            exist in the database.
        readonly (bool, default False):
            Whether to open the database connection in readonly mode.
        check_same_thread (bool, default True):
            By default, check_same_thread is True and only the creating thread
            may use the connection. If set False, the returned connection may be
            shared across multiple threads.  The dask distributed evaluator has
            workers that run code in a separate thread from the model class object,
            so setting this to False is necessary to enable SQLite connections on
            the workers.
    """

    def __init__(
            self,
            database_path=":memory:",
            initialize=False,
            readonly=False,
            check_same_thread=True,
    ):

        if database_path[-3:] == '.gz':
            import tempfile, os, shutil, gzip
            if not os.path.isfile(database_path):
                raise FileNotFoundError(database_path)
            self._tempdir = tempfile.TemporaryDirectory()
            tempfilename = os.path.join(self._tempdir.name, os.path.basename(database_path[:-3]))
            with open(tempfilename, "wb") as tmp:
                shutil.copyfileobj(gzip.open(database_path), tmp)
            database_path = tempfilename

        self.database_path = database_path
        self.readonly = readonly

        if self.database_path == ":memory:":
            initialize = True
        # in order:
        self.modules = {}
        if initialize == 'skip':
            self.conn = self.__create(
                [],
                wipe=False,
                check_same_thread=check_same_thread,
            )
        elif initialize:
            self.conn = self.__create(
                ["emat_db_init.sql", "meta_model.sql"],
                wipe=True,
                check_same_thread=check_same_thread,
            )
        elif readonly:
            self.conn = sqlite3.connect(
                f'file:{database_path}?mode=ro',
                uri=True,
                check_same_thread=check_same_thread,
            )
        else:
            self.conn = self.__create(
                ["emat_db_init.sql", "meta_model.sql"],
                wipe=False,
                check_same_thread=check_same_thread,
            )
        if not readonly:
            self.conn.execute("PRAGMA foreign_keys = ON")
            with self.conn:
                self.conn.cursor().execute(sq.SET_VERSION_DATABASE)
                self.conn.cursor().execute(sq.SET_MINIMUM_VERSION_DATABASE)

        # Warn if opening a database that requires a more recent version of tmip-emat.
        try:
            _min_ver = list(self.conn.cursor().execute(sq.GET_MINIMUM_VERSION_DATABASE))
            if len(_min_ver):
                _min_ver_number = _min_ver[0][0]
                from ... import __version__
                ver = (np.asarray([int(i) for i in __version__.split(".")])
                       @ np.asarray([1000000,1000,1]))
                if _min_ver_number > ver:
                    warnings.warn(
                        f"Database requires emat version {_min_ver_number}",
                        category=DatabaseVersionWarning,
                    )
        except:
            pass

        # update old databases
        if 'design' in self._debug_query(table='ema_experiment')['name'].to_numpy():
            self.update_database(sq.UPDATE_DATABASE_ema_design_experiment)

        if 'measure_run' not in self._debug_query(table='ema_experiment_measure')['name'].to_numpy():
            self.update_database(sq.UPDATE_DATABASE_ema_experiment_measure_ADD_measure_run)

        atexit.register(self.conn.close)


    def __create(self, filenames, wipe=False, check_same_thread=None):
        """
        Call sql files to create sqlite database file
        """
       
        # close connection and delete file if exists
        if self.database_path != ":memory:" and wipe:
            self.__delete_database()
        try:
            conn = sqlite3.connect(self.database_path, check_same_thread=check_same_thread)
        except sqlite3.OperationalError as err:
            raise sqlite3.OperationalError(f'error on connecting to {self.database_path}') from err
        with conn:
            cur = conn.cursor()

            for filename in filenames:
                _logger.debug("running script " + filename)
                contents = (
                    self.__read_sql_file(
                        os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            filename
                        )
                    )
                )
                cur.executescript(contents)

        return conn

    def update_database(self, queries, on_error='ignore'):
        """
        Update database for compatability with tmip-emat 0.4
        """
        if self.readonly:
            raise DatabaseVersionError("cannot open or update an old database in readonly")
        else:
            warnings.warn(
                f"updating database file",
                category=DatabaseVersionWarning,
            )
        with self.conn:
            cur = self.conn.cursor()
            for u in queries:
                try:
                    cur.execute(u)
                except:
                    if on_error in ('log','raise'):
                        _logger.error(f"SQL Query:\n{u}\n")
                    if on_error == 'raise':
                        raise


    def __repr__(self):
        scopes = self.read_scope_names()
        if len(scopes) == 1:
            return f'<emat.SQLiteDB with scope "{scopes[0]}">'
        elif len(scopes) == 0:
            return f'<emat.SQLiteDB with no scopes>'
        elif len(scopes) <= 4:
            return f'<emat.SQLiteDB with {len(scopes)} scopes: "{", ".join(scopes[0])}">'
        else:
            return f'<emat.SQLiteDB with {len(scopes)} scopes>'

    def __read_sql_file(self, filename):
        """
        helper function to load sql files to create database
        """
        sql_file_path = filename
        _logger.debug(sql_file_path)
        with open(sql_file_path, 'r') as fil:
            all_lines = fil.read()
        return all_lines
        
    def __delete_database(self):
        """
        Delete the sqlite database file
        """
        if os.path.exists(self.database_path):
            os.remove(self.database_path)

    def _debug_query(self, qry=None, table=None):
        if qry is None and table is not None:
            qry = f"PRAGMA table_info({table});"
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(qry)
            try:
                cols = ([i[0] for i in cur.description])
            except:
                df = None
            else:
                df = pd.DataFrame(cur.fetchall(), columns=cols)
        return df

    def get_db_info(self):
        """
        Get a short string describing this Database

        Returns:
            str
        """
        return f"SQLite @ {self.database_path}"

    @copydoc(Database.init_xlm)
    def init_xlm(self, parameter_list: List[tuple], measure_list: List[tuple]):

        with self.conn:
            cur = self.conn.cursor()

            # experiment variables - description and type (risk or strategy)
            for xl in parameter_list:
                cur.execute(sq.CONDITIONAL_INSERT_XL, xl)

            # performance measures - description
            for m in measure_list:
                cur.execute(sq.CONDITIONAL_INSERT_M, m)


    @copydoc(Database.write_scope)
    def write_scope(self, scope_name, sheet, scp_xl, scp_m, content=None):

        with self.conn:
            cur = self.conn.cursor()

            if content is not None:
                import gzip, cloudpickle
                blob = gzip.compress(cloudpickle.dumps(content))
            else:
                blob = None

            try:
                cur.execute(sq.INSERT_SCOPE, [scope_name, sheet, blob])
            except sqlite3.IntegrityError:
                raise KeyError(f'scope named "{scope_name}" already exists')

            for xl in scp_xl:
                cur.execute(sq.INSERT_SCOPE_XL, [scope_name, xl])
                if cur.rowcount < 1:
                    raise KeyError('Experiment Variable {0} not present in database'
                                   .format(xl))

            for m in scp_m:
                cur.execute(sq.INSERT_SCOPE_M, [scope_name, m])
                if cur.rowcount < 1:
                    raise KeyError('Performance measure {0} not present in database'
                                   .format(m))


    @copydoc(Database.store_scope)
    def store_scope(self, scope):
        return scope.store_scope(self)

    @copydoc(Database.read_scope)
    def read_scope(self, scope_name=None):

        cur = self.conn.cursor()

        if scope_name is None:
            scope_names = self.read_scope_names()
            if len(scope_names) > 1:
                raise ValueError("multiple scopes stored in database, you must identify one:\n -"+"\n -".join(scope_names))
            elif len(scope_names) == 0:
                raise ValueError("no scopes stored in database")
            else:
                scope_name = scope_names[0]
        try:
            blob = cur.execute(sq.GET_SCOPE, [scope_name]).fetchall()[0][0]
        except IndexError:
            raise KeyError(f"scope '{scope_name}' not found")
        if blob is None:
            return blob
        import gzip, cloudpickle
        try:
            return cloudpickle.loads(gzip.decompress(blob))
        except ModuleNotFoundError as err:
            if "ema_workbench" in err.msg:
                import sys
                from ... import workbench
                sys.modules['ema_workbench'] = workbench
                return cloudpickle.loads(gzip.decompress(blob))
            else:
                raise


    @copydoc(Database.write_metamodel)
    def write_metamodel(self, scope_name, metamodel=None, metamodel_id=None, metamodel_name=''):

        with self.conn:
            if metamodel is None and hasattr(scope_name, 'scope'):
                # The metamodel was the one and only argument,
                # and it embeds the Scope.
                metamodel = scope_name
                scope_name = metamodel.scope.name

            scope_name = self._validate_scope(scope_name, None)

            # Do not store PythonCoreModel, store the metamodel it wraps
            from ...model.core_python import PythonCoreModel
            from ...model.meta_model import MetaModel
            if isinstance(metamodel, PythonCoreModel) and isinstance(metamodel.function, MetaModel):
                metamodel_name = metamodel_name or metamodel.name
                if metamodel_id is None:
                    metamodel_id = metamodel.metamodel_id
                metamodel = metamodel.function

            # Get a new id if needed
            if metamodel_id is None:
                metamodel_id = self.get_new_metamodel_id(scope_name)

            # Don't pickle-zip a null model
            if metamodel is None:
                blob = metamodel
            else:
                import gzip, cloudpickle
                blob = gzip.compress(cloudpickle.dumps(metamodel))

            try:
                cur = self.conn.cursor()
                cur.execute(sq.INSERT_METAMODEL_PICKLE,
                             [scope_name, metamodel_id, metamodel_name, blob])
            except sqlite3.IntegrityError:
                raise KeyError(f'metamodel_id {metamodel_id} for scope "{scope_name}" already exists')

        return metamodel_id

    @copydoc(Database.read_metamodel)
    def read_metamodel(self, scope_name, metamodel_id=None):
        scope_name = self._validate_scope(scope_name, None)

        if metamodel_id is None:
            candidate_ids = self.read_metamodel_ids(scope_name)
            if len(candidate_ids) == 1:
                metamodel_id = candidate_ids[0]
            elif len(candidate_ids) == 0:
                raise ValueError(f'no metamodels for scope "{scope_name}" are stored')
            else:
                raise ValueError(f'{len(candidate_ids)} metamodels for scope "{scope_name}" are stored')

        cur = self.conn.cursor()
        query_result = cur.execute(
            sq.GET_METAMODEL_PICKLE,
            [scope_name, metamodel_id]
        ).fetchall()
        try:
            name, blob = query_result[0]
        except IndexError:
            raise KeyError(f"no metamodel_id {metamodel_id} for scope named '{scope_name}'")

        import gzip, pickle
        mm = pickle.loads(gzip.decompress(blob))

        scope = self.read_scope(scope_name)

        from ...model.core_python import PythonCoreModel
        return PythonCoreModel(
            mm,
            configuration=None,
            scope=scope,
            safe=True,
            db=self,
            name=name,
            metamodel_id=metamodel_id,
        )

    @copydoc(Database.read_metamodel_ids)
    def read_metamodel_ids(self, scope_name):
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)
        metamodel_ids = [i[0] for i in cur.execute(sq.GET_METAMODEL_IDS,
                                                         [scope_name] ).fetchall()]
        return metamodel_ids

    @copydoc(Database.get_new_metamodel_id)
    def get_new_metamodel_id(self, scope_name):
        with self.conn:
            scope_name = self._validate_scope(scope_name, None)
            cur = self.conn.cursor()
            metamodel_id = [i[0] for i in cur.execute(sq.GET_NEW_METAMODEL_ID,).fetchall()][0]
            self.write_metamodel(scope_name, None, metamodel_id)
            return metamodel_id


    @copydoc(Database.add_scope_meas)
    def add_scope_meas(self, scope_name, scp_m):
        with self.conn:
            cur = self.conn.cursor()
            scope_name = self._validate_scope(scope_name, None)

            # test that scope exists
            saved_m = cur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()
            if len(saved_m) == 0:
                raise KeyError('named scope does not exist')

            for m in scp_m:
                if m not in saved_m:
                    cur.execute(sq.INSERT_SCOPE_M, [scope_name, m])
                    if cur.rowcount < 1:
                        raise KeyError('Performance measure {0} not present in database'
                                       .format(m))


    @copydoc(Database.delete_scope) 
    def delete_scope(self, scope_name):
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(sq.DELETE_SCOPE, [scope_name])

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
            xl_df (pandas.DataFrame):
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
        if design_name is None:
            design_name = 'ad hoc'

        with self.conn:
            scope_name = self._validate_scope(scope_name, 'design_name')
            # local cursor because we'll depend on lastrowid
            fcur = self.conn.cursor()

            # get list of experiment variables - except "one"
            scp_xl = fcur.execute(sq.GET_SCOPE_XL, [scope_name]).fetchall()
            if len(scp_xl) == 0:
                raise UserWarning('named scope {0} not found - experiments will \
                                      not be recorded'.format(scope_name))

            fcur.execute(sq.INSERT_DESIGN, [scope_name, design_name])

            ### split experiments into novel and duplicate ###
            # first join to existing experiments
            existing_experiments = self.read_experiment_parameters(scope_name, None)
            combined_experiments = pd.concat([
                existing_experiments,
                xl_df.set_index(np.full(len(xl_df), -1, dtype=int)),
            ])
            combined_experiments = reindex_duplicates(combined_experiments)
            xl_df_ = combined_experiments.iloc[-len(xl_df):]
            novel_flag = xl_df_.index.isin([-1])
            novel_experiments = xl_df_.loc[novel_flag]
            duplicate_experiments = xl_df_.loc[~novel_flag]

            ex_ids = []

            for _, row in novel_experiments.iterrows():
                # create new experiment and get id
                fcur.execute(sq.INSERT_EXPERIMENT, [scope_name])
                ex_id = fcur.lastrowid
                # set each from experiment defitinion
                for xl in scp_xl:
                    try:
                        value = row[xl[0]]
                        fcur.execute(sq.INSERT_EX_XL, [ex_id, value, xl[0]])
                    except TypeError:
                        _logger.error(f'Experiment definition missing {xl[0]} variable')
                        raise

                # Add this experiment id to this design
                ex_ids.append(ex_id)
                try:
                    fcur.execute(sq.INSERT_DESIGN_EXPERIMENT, [scope_name, design_name, ex_id])
                except Exception as err:
                    _logger.error(str(err))
                    _logger.error(f"scope_name, design_name, ex_id= {scope_name, design_name, ex_id}")
                    raise

            for ex_id in duplicate_experiments.index:
                # Add this experiment id to this design
                ex_ids.append(ex_id)
                try:
                    fcur.execute(sq.INSERT_DESIGN_EXPERIMENT, [scope_name, design_name, ex_id])
                except Exception as err:
                    _logger.error(str(err))
                    _logger.error(f"scope_name, design_name, ex_id= {scope_name, design_name, ex_id}")
                    raise

            return ex_ids

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
        scope_name = self._validate_scope(scope_name, 'design_name')
        parameters = {}
        for a in args:
            if a is not None:
                parameters.update(a)
        parameters.update(kwargs)
        xl_df = pd.DataFrame(parameters, index=[0])
        result = self.read_experiment_ids(scope_name, xl_df)
        return result[0]

    def get_experiment_id(self, scope_name=None, *args, **kwargs):
        """
        Read or create an experiment id in the database.

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
        scope_name = self._validate_scope(scope_name, 'design_name')
        from ...exceptions import MissingIdWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=MissingIdWarning)
            ex_id = self.read_experiment_id(scope_name, *args, **kwargs)
        if ex_id is None:
            parameters = self.read_scope(scope_name).get_parameter_defaults()
            for a in args:
                if a is not None:
                    parameters.update(a)
            parameters.update(kwargs)
            df = pd.DataFrame(parameters, index=[0])
            ex_id = self.write_experiment_parameters(scope_name, None, df)[0]
        return ex_id

    def new_run_id(self, scope_name=None, *args, location=None, **kwargs):
        """
        Create a new run_id in the database.

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
        scope_name = self._validate_scope(scope_name, 'design_name')
        ex_id = self.get_experiment_id(scope_name, *args, **kwargs)
        run_id = uuid.uuid1()
        if location is True:
            import platform
            location = platform.node()
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                sq.NEW_EXPERIMENT_RUN,
                dict(
                    run_id=run_id.bytes,
                    experiment_id=ex_id,
                    run_location=location,
                )
            )
            return run_id, ex_id



    def read_experiment_ids(
            self,
            scope_name,
            xl_df,
    ):
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
        # local cursor
        fcur = self.conn.cursor()
        ex_ids = []
        missing_ids = 0

        try:
            # get list of experiment variables - except "one"
            scp_xl = fcur.execute(sq.GET_SCOPE_XL, [scope_name]).fetchall()
            if len(scp_xl) == 0:
                raise ValueError('named scope {0} not found - experiment ids \
                                      not available'.format(scope_name))


            for row in xl_df.itertuples(index=False, name=None):

                candidate_ids = None

                # get all ids by value
                for par_name, par_value in zip(xl_df.columns, row):
                    possible_ids = set([i[0] for i in fcur.execute(
                        sq.GET_EXPERIMENT_IDS_BY_VALUE,
                        [scope_name, par_name, par_value],
                    ).fetchall()])
                    if candidate_ids is None:
                        candidate_ids = possible_ids
                    else:
                        candidate_ids &= possible_ids

                if len(candidate_ids) > 1:
                    raise ValueError('multiple matching experiment ids found')
                elif len(candidate_ids) == 1:
                    ex_ids.append(candidate_ids.pop())
                else:
                    missing_ids +=1
                    ex_ids.append(None)
        finally:
            fcur.close()

        if missing_ids:
            from ...exceptions import MissingIdWarning
            warnings.warn(f'missing {missing_ids} ids', category=MissingIdWarning)
        return ex_ids

    def read_all_experiment_ids(
            self,
            scope_name,
            design_name=None,
    ):
        """
        Read the experiment ids previously defined in the database

        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run
            design_name (str or None): experiment design name.  Set to None
                to find experiments across all designs.

        Returns:
            list: the experiment id's of the identified experiments

        Raises:
            ValueError: If scope name does not exist

        """

        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, 'design_name')
        if design_name is None:
            experiment_ids = [i[0] for i in cur.execute(sq.GET_EXPERIMENT_IDS_ALL,
                                                             [scope_name] ).fetchall()]
        else:
            experiment_ids = [i[0] for i in cur.execute(sq.GET_EXPERIMENT_IDS_IN_DESIGN,
                                                             [scope_name, design_name] ).fetchall()]
        return experiment_ids

    def _validate_scope(self, scope_name, design_parameter_name="design_name"):
        """Validate the scope argument to a function."""
        known_scopes = self.read_scope_names()

        if len(known_scopes) == 0:
            raise ValueError(f'there are no stored scopes')

        # None, but there is only one scope in the DB, so use it.
        if scope_name is None or scope_name is 0:
            if len(known_scopes) == 1:
                return known_scopes[0]
            raise ValueError(f'there are {len(known_scopes)} scopes, must identify scope explicitly')

        if scope_name not in known_scopes:
            oops = self.read_scope_names(design_name=scope_name)
            if oops:
                if design_parameter_name:
                    des = f", {design_parameter_name}='{oops[0]}'"
                else:
                    des = ''
                raise ValueError(f'''no scope named "{scope_name}", did you mean '''
                                 f'''"...(scope='{scope_name}'{des},...)"?''')
            raise ValueError(f'no scope named "{scope_name}" is stored in the database, only {known_scopes}')

        return scope_name

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

        if design is not None:
            if design_name is None:
                design_name = design
                warnings.warn("the `design` argument is deprecated for "
                              "read_experiment_parameters, use `design_name`", DeprecationWarning)
            elif design != design_name:
                raise ValueError("cannot give both `design_name` and `design`")

        scope_name = self._validate_scope(scope_name, 'design_name')
        cur = self.conn.cursor()

        if experiment_ids is not None:
            query = sq.GET_EX_XL_IDS_IN
            if isinstance(experiment_ids, int):
                experiment_ids = [experiment_ids]
            subquery = ",".join(f"?{n+2}" for n in range(len(experiment_ids)))
            query = query.replace("???", subquery)
            xl_df = pd.DataFrame(cur.execute(query, [scope_name, *experiment_ids]).fetchall())
        elif only_pending:
            if design_name is None:
                xl_df = pd.DataFrame(cur.execute(
                    sq.GET_EX_XL_ALL_PENDING, [scope_name, ]).fetchall())
            else:
                xl_df = pd.DataFrame(cur.execute(
                    sq.GET_PENDING_EXPERIMENT_PARAMETERS, [scope_name, design_name]
                ).fetchall())
        else:
            if design_name is None:
                xl_df = pd.DataFrame(cur.execute(
                        sq.GET_EX_XL_ALL, [scope_name, ]).fetchall())
            else:
                xl_df = pd.DataFrame(cur.execute(
                        sq.GET_EXPERIMENT_PARAMETERS, [scope_name, design_name]
                ).fetchall())
        if xl_df.empty is False:
            xl_df = xl_df.pivot(index=0, columns=1, values=2)
        xl_df.index.name = 'experiment'
        xl_df.columns.name = None

        column_order = (
                self.read_constants(scope_name)
                + self.read_uncertainties(scope_name)
                + self.read_levers(scope_name)
        )

        from ...experiment.experimental_design import ExperimentalDesign
        result = ExperimentalDesign(xl_df[[i for i in column_order if i in xl_df.columns]])
        result.design_name = design_name
        if ensure_dtypes:
            scope = self.read_scope(scope_name)
            if scope is not None:
                result = scope.ensure_dtypes(result)
        return result

    def write_experiment_measures(
            self,
            scope_name,
            source,
            m_df,
            run_ids=None,
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

        with self.conn:
            cur = self.conn.cursor()
            scope_name = self._validate_scope(scope_name, None)
            scp_m = cur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()

            if len(scp_m) == 0:
                raise UserWarning('named scope {0} not found - experiments will \
                                      not be recorded'.format(scope_name))

            if run_ids is None:
                run_ids = pd.Index([None]*len(m_df))

            for m in scp_m:
                if m[0] in m_df.columns:
                    for (ex_id, value), uid in zip(m_df[m[0]].iteritems(),run_ids):
                        if isinstance(uid, uuid.UUID):
                            uid = uid.bytes
                        # index is experiment id
                        try:
                            if not pd.isna(m[0]):
                                cur.execute(
                                    sq.INSERT_EX_M,
                                    dict(
                                        experiment_id=ex_id,
                                        measure_value=value,
                                        measure_source=source,
                                        measure_name=m[0],
                                        measure_run=uid,
                                    )
                                )
                        except:
                            _logger.error(f"Error saving {value} to m {m[0]} for ex {ex_id}")
                            raise

    def write_ex_m_1(
            self,
            scope_name,
            source,
            ex_id,
            m_name,
            m_value,
            run_id=None,
    ):
        """Write a single performance measure result for an experiment

        Write the performance measure result for an experiment
        in the scope - if the scope does not exist, nothing is recorded

        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run
            source (int): indicator of performance measure source
                (0 = core model or non-zero = meta-model id)
            ex_id (int): experiment id
            m_name (str): performance measure name
            m_value (numeric): performance measure value
            run_ids (uuid, optional):
                Provide an optional universally unique run id
                (UUID) for these results. The UUID can be used to help
                identify problems and organize model runs.

        Raises:
            UserWarning: If scope name does not exist
        """
        with self.conn:
            scope_name = self._validate_scope(scope_name, None)
            cur = self.conn.cursor()
            scp_m = cur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()

            if len(scp_m) == 0:
                raise UserWarning('named scope {0} not found - experiments will \
                                      not be recorded'.format(scope_name))

            for m in scp_m:
                if m[0] == m_name:
                    try:
                        if not pd.isna(m[0]):
                            cur.execute(
                                sq.INSERT_EX_M,
                                dict(
                                    experiment_id=ex_id,
                                    measure_value=m_value,
                                    measure_source=source,
                                    measure_name=m[0],
                                    measure_run=run_id,
                                )
                            )
                    except:
                        _logger.error(f"Error saving {m_value} to m {m[0]} for ex {ex_id}")
                        raise

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
        scope_name = self._validate_scope(scope_name, 'design_name')
        cur = self.conn.cursor()
        if design_name is None:
            if source is None:
                ex_xlm = pd.DataFrame(cur.execute(sq.GET_EX_XLM_ALL,
                                                       [scope_name,]).fetchall())
            else:
                ex_xlm = pd.DataFrame(cur.execute(sq.GET_EX_XLM_ALL_BYSOURCE,
                                                       [scope_name,source]).fetchall())
        elif isinstance(design_name, str):
            if source is None:
                ex_xlm = pd.DataFrame(cur.execute(
                    sq.GET_EXPERIMENT_PARAMETERS_AND_MEASURES,
                    [scope_name, design_name],
                ).fetchall())
            else:
                ex_xlm = pd.DataFrame(cur.execute(
                    sq.GET_EXPERIMENT_PARAMETERS_AND_MEASURES_BYSOURCE,
                    [scope_name, design_name, source]
                ).fetchall())
        else:
            if source is None:
                ex_xlm = pd.concat([
                    pd.DataFrame(cur.execute(
                        sq.GET_EXPERIMENT_PARAMETERS_AND_MEASURES,
                        [scope_name, dn],
                    ).fetchall())
                    for dn in design_name
                ])
            else:
                ex_xlm = pd.concat([
                    pd.DataFrame(cur.execute(
                        sq.GET_EXPERIMENT_PARAMETERS_AND_MEASURES_BYSOURCE,
                        [scope_name, dn, source],
                    ).fetchall())
                    for dn in design_name
                ])
        if ex_xlm.empty is False:
            ex_xlm = ex_xlm.pivot(index=0, columns=1, values=2)
        ex_xlm.index.name = 'experiment'
        ex_xlm.columns.name = None

        if only_incomplete:
            import numpy, pandas
            retain = numpy.zeros(len(ex_xlm), dtype=bool)
            for meas_name in self.read_measures(scope_name):
                if meas_name not in ex_xlm.columns:
                    retain[:] = True
                    break
                else:
                    retain[:] |= pandas.isna(ex_xlm[meas_name])
            ex_xlm = ex_xlm.loc[retain, :]

        column_order = (
                self.read_constants(scope_name)
                + self.read_uncertainties(scope_name)
                + self.read_levers(scope_name)
                + self.read_measures(scope_name)
        )
        result = ex_xlm[[i for i in column_order if i in ex_xlm.columns]]

        if only_complete:
            result = result[~result.isna().any(axis=1)]
        elif only_incomplete:
            result = result[result.isna().any(axis=1)]

        if only_with_measures:
            result_measures = result[[i for i in self.read_measures(scope_name) if i in result.columns]]
            result = result[~result_measures.isna().all(axis=1)]
        elif only_pending:
            result_measures = result[[i for i in self.read_measures(scope_name) if i in result.columns]]
            result = result[result_measures.isna().all(axis=1)]

        if ensure_dtypes:
            scope = self.read_scope(scope_name)
            if scope is not None:
                result = scope.ensure_dtypes(result)

        from ...experiment.experimental_design import ExperimentalDesign
        result = ExperimentalDesign(result)
        result.design_name = design_name

        return result

    def read_experiment_measures(
            self,
            scope_name,
            design_name=None,
            experiment_id=None,
            source=None,
            design=None,
            index_type='experiment_id',
    ):
        """
        Read experiment results from the database.

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
            source (int, optional): The source identifier of the
                experimental outcomes to load.  If not given, but
                there are only results from a single source in the
                database, those results are returned.  If there are
                results from multiple sources, an error is raised.
            design (str): Deprecated, use `design_name`.

        Returns:
            results (pandas.DataFrame): performance measures

        Raises:
            ValueError
                When no source is given but the database contains
                results from multiple sources.
        """

        if design is not None:
            if design_name is None:
                design_name = design
                warnings.warn("the `design` argument is deprecated for "
                              "read_experiment_parameters, use `design_name`", DeprecationWarning)
            elif design != design_name:
                raise ValueError("cannot give both `design_name` and `design`")

        if index_type not in ('experiment_id', 'run_id'):
            raise ValueError("index_type must be 'experiment_id' or 'run_id'")

        scope_name = self._validate_scope(scope_name, 'design_name')
        if design_name is None:
            if experiment_id is None:
                sql = sq.GET_EXPERIMENT_MEASURES_ALL
                arg = dict(scope_name=scope_name)
                if source is not None:
                    sql += ' AND measure_source = @measure_source'
                    arg['measure_source'] = source
            else:
                sql = sq.GET_EXPERIMENT_MEASURES_BY_ID_ALL
                arg = dict(
                    scope_name=scope_name,
                    experiment_id=experiment_id,
                )
                if source is not None:
                    sql += ' AND measure_source = @measure_source'
                    arg['measure_source'] = source
        else:
            if experiment_id is None:
                sql = sq.GET_EXPERIMENT_MEASURES
                arg = dict(
                    scope_name=scope_name,
                    design_name=design_name,
                )
                if source is not None:
                    sql = sql.replace("/*source*/", ' AND measure_source = @measure_source')
                    arg['measure_source'] = source
            else:
                sql = sq.GET_EXPERIMENT_MEASURES_BY_ID
                arg = dict(
                    scope_name=scope_name,
                    design_name=design_name,
                    experiment_id=experiment_id,
                )
                if source is not None:
                    sql = sql.replace("/*source*/", ' AND measure_source = @measure_source')
                    arg['measure_source'] = source

        if index_type == 'run_id':
            sql = sql.replace(
                "eem.experiment_id, --index_type",
                "COALESCE(eer.run_id, 'Ex'||printf('%05d', eem.experiment_id)), --index_type",
            )
        cur = self.conn.cursor()
        ex_m = pd.DataFrame(cur.execute(sql, arg).fetchall())
        if ex_m.empty is False:
            ex_m = ex_m.pivot(index=0, columns=1, values=2)
        ex_m.index.name = 'experiment'
        if index_type == 'run_id':
            ex_m.index = [
                (uuid.UUID(bytes=i) if isinstance(i,bytes) else i)
                for i in ex_m.index
            ]
            ex_m.index.name = 'run'
        ex_m.columns.name = None

        column_order = (
                self.read_measures(scope_name)
        )

        return ex_m[[i for i in column_order if i in ex_m.columns]]

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
        if design is not None:
            if design_name is None:
                design_name = design
                warnings.warn("the `design` argument is deprecated for "
                              "read_experiment_parameters, use `design_name`", DeprecationWarning)
            elif design != design_name:
                raise ValueError("cannot give both `design_name` and `design`")

        scope_name = self._validate_scope(scope_name, 'design_name')
        if design_name is None:
            sql = sq.GET_EXPERIMENT_MEASURE_SOURCES
            arg = {'scope_name':scope_name}
            if experiment_id is not None:
                sql += " AND experiment_id = @experiment_id"
                arg['experiment_id'] = experiment_id
        else:
            sql = sq.GET_EXPERIMENT_MEASURE_SOURCES_BY_DESIGN
            arg = {'scope_name': scope_name, 'design_name':design_name, }
            if experiment_id is not None:
                sql += " AND experiment_id = @experiment_id"
                arg['experiment_id'] = experiment_id
        cur = self.conn.cursor()
        return [i[0] for i in cur.execute(sql, arg).fetchall()]

    def delete_experiments(
            self,
            scope_name,
            design_name=None,
            design=None,
    ):
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
        with self.conn:
            if design is not None:
                if design_name is None:
                    design_name = design
                    warnings.warn("the `design` argument is deprecated for "
                                  "read_experiment_parameters, use `design_name`", DeprecationWarning)
                elif design != design_name:
                    raise ValueError("cannot give both `design_name` and `design`")
            scope_name = self._validate_scope(scope_name, 'design_name')
            cur = self.conn.cursor()
            cur.execute(sq.DELETE_DESIGN_EXPERIMENTS, [scope_name, design_name])
            cur.execute(sq.DELETE_LOOSE_EXPERIMENTS, [scope_name,])

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
        with self.conn:
            cur = self.conn.cursor()
            bindings = ",".join( ["?"] * len(experiment_ids) )
            cur.execute(
                sq.DELETE_MEASURES_BY_EXPERIMENT_ID.replace('?',bindings),
                experiment_ids,
            )

    def write_experiment_all(
            self,
            scope_name,
            design_name,
            source,
            xlm_df,
            run_ids=None,
    ):
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
            xlm_df (pandas.DataFrame):
                The columns of this DataFrame are the experiment
                parameters (i.e. policy levers, uncertainties, and
                constants) and performance measures, and each row
                is an experiment.
            run_ids (pandas.Index, optional):
                Provide an optional index of universally unique run ids
                (UUIDs) for these results. The UUIDs can be used to help
                identify problems and organize model runs.

        Raises:
            UserWarning: If scope and design already exist
            TypeError: If not all scope variables are defined in the
                experiment
        """

        with self.conn:
            scope_name = self._validate_scope(scope_name, 'design_name')
            fcur = self.conn.cursor()

            exist = pd.DataFrame(fcur.execute(
                sq.GET_EXPERIMENT_PARAMETERS_AND_MEASURES,
                [scope_name, design_name],
            ).fetchall())
            if exist.empty is False:
                raise UserWarning(
                    'scope {0} with design {1} found, ' 
                    'must be deleted before recording'
                        .format(scope_name, design_name)
                )

            # get list of experiment variables
            scp_xl = fcur.execute(sq.GET_SCOPE_XL, [scope_name]).fetchall()
            scp_m = fcur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()

            experiment_ids = self.write_experiment_parameters(
                scope_name,
                design_name,
                xlm_df[[z[0] for z in scp_xl]],
            )
            xlm_df.index = experiment_ids
            self.write_experiment_measures(
                scope_name,
                source,
                xlm_df[[z[0] for z in scp_m]],
                run_ids=run_ids,
            )

        
    @copydoc(Database.read_scope_names)
    def read_scope_names(self, design_name=None) -> list:
        cur = self.conn.cursor()
        if design_name is None:
            scopes = [i[0] for i in cur.execute(sq.GET_SCOPE_NAMES ).fetchall()]
        else:
            scopes = [i[0] for i in cur.execute(sq.GET_SCOPES_CONTAINING_DESIGN_NAME,
                                                     [design_name] ).fetchall()]
        return scopes

    @copydoc(Database.read_design_names)
    def read_design_names(self, scope_name:str) -> list:
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)
        designs = [i[0] for i in cur.execute(sq.GET_DESIGN_NAMES, [scope_name] ).fetchall()]
        return designs

    @copydoc(Database.read_uncertainties)
    def read_uncertainties(self, scope_name:str) -> list:
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)
        scp_x = cur.execute(sq.GET_SCOPE_X, [scope_name]).fetchall()
        return [i[0] for i in scp_x]

    @copydoc(Database.read_levers)
    def read_levers(self, scope_name:str) -> list:
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)
        scp_l = cur.execute(sq.GET_SCOPE_L, [scope_name]).fetchall()
        return [i[0] for i in scp_l]

    @copydoc(Database.read_constants)
    def read_constants(self, scope_name:str) -> list:
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)
        scp_c = cur.execute(sq.GET_SCOPE_C, [scope_name]).fetchall()
        return [i[0] for i in scp_c]

    @copydoc(Database.read_measures)
    def read_measures(self, scope_name: str) -> list:
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)
        scp_m = cur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()
        return [i[0] for i in scp_m]

    @copydoc(Database.read_box)
    def read_box(self, scope_name: str, box_name: str, scope=None):
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)

        from ...scope.box import Box
        parent = self.read_box_parent_name(scope_name, box_name)
        box = Box(name=box_name, scope=scope,
                  parent=parent)

        for row in cur.execute(sq.GET_BOX_THRESHOLDS, [scope_name, box_name]):
            par_name, t_value, t_type = row
            if t_type == -2:
                box.set_lower_bound(par_name, t_value)
            elif t_type == -1:
                box.set_upper_bound(par_name, t_value)
            elif t_type == 0:
                box.relevant_features.add(par_name)
            elif t_type >= 1:
                box.add_to_allowed_set(par_name, t_value)

        return box

    @copydoc(Database.read_box_names)
    def read_box_names(self, scope_name: str):
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)
        names = cur.execute(sq.GET_BOX_NAMES, [scope_name]).fetchall()
        return [i[0] for i in names]

    @copydoc(Database.read_box_parent_name)
    def read_box_parent_name(self, scope_name: str, box_name:str):
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)
        try:
            return cur.execute(sq.GET_BOX_PARENT_NAME, [scope_name, box_name]).fetchall()[0][0]
        except IndexError:
            return None

    @copydoc(Database.read_box_parent_names)
    def read_box_parent_names(self, scope_name: str):
        cur = self.conn.cursor()
        scope_name = self._validate_scope(scope_name, None)
        names = cur.execute(sq.GET_BOX_PARENT_NAMES, [scope_name]).fetchall()
        return {i[0]:i[1] for i in names}

    @copydoc(Database.read_boxes)
    def read_boxes(self, scope_name: str=None, scope=None):
        from ...scope.box import Boxes
        if scope is not None:
            scope_name = scope.name
        scope_name = self._validate_scope(scope_name, None)
        names = self.read_box_names(scope_name)
        u = Boxes(scope=scope)
        for name in names:
            box = self.read_box(scope_name, name, scope=scope)
            u.add(box)
        return u

    @copydoc(Database.write_box)
    def write_box(self, box, scope_name=None):
        with self.conn:
            from ...scope.box import Box, Bounds
            assert isinstance(box, Box)

            try:
                scope_name_ = box.scope.name
            except AttributeError:
                scope_name_ = scope_name
            if scope_name is not None and scope_name != scope_name_:
                raise ValueError("scope_name mismatch")
            scope_name = scope_name_
            scope_name = self._validate_scope(scope_name, None)
            cur = self.conn.cursor()

            p_ = set(self.read_uncertainties(scope_name) + self.read_levers(scope_name))
            m_ = set(self.read_measures(scope_name))

            if box.parent_box_name is None:
                cur.execute(sq.INSERT_BOX, [scope_name, box.name])
            else:
                cur.execute(sq.INSERT_SUBBOX, [scope_name, box.name,
                                                    box.parent_box_name])

            for t_name, t_vals in box._thresholds.items():

                if t_name in p_:
                    sql_cl = sq.CLEAR_BOX_THRESHOLD_P
                    sql_in = sq.SET_BOX_THRESHOLD_P
                elif t_name in m_:
                    sql_cl = sq.CLEAR_BOX_THRESHOLD_M
                    sql_in = sq.SET_BOX_THRESHOLD_M
                else:
                    warnings.warn(f"{t_name} not identifiable as parameter or measure")
                    continue

                cur.execute(sql_cl, [scope_name, box.name, t_name])

                if isinstance(t_vals, Bounds):
                    if t_vals.lowerbound is not None:
                        cur.execute(sql_in, [scope_name, box.name, t_name, t_vals.lowerbound, -2])
                    if t_vals.upperbound is not None:
                        cur.execute(sql_in, [scope_name, box.name, t_name, t_vals.upperbound, -1])
                elif isinstance(t_vals, AbstractSet):
                    for n, t_val in enumerate(t_vals, start=1):
                        cur.execute(sql_in, [scope_name, box.name, t_name, t_val, n])
                else:
                    raise NotImplementedError(str(type(t_vals)))

            for t_name in box.relevant_features:

                if t_name in p_:
                    sql_cl = sq.CLEAR_BOX_THRESHOLD_P
                    sql_in = sq.SET_BOX_THRESHOLD_P
                elif t_name in m_:
                    sql_cl = sq.CLEAR_BOX_THRESHOLD_M
                    sql_in = sq.SET_BOX_THRESHOLD_M
                else:
                    warnings.warn(f"{t_name} not identifiable as parameter or measure")
                    continue

                cur.execute(sql_cl, [scope_name, box.name, t_name])
                cur.execute(sql_in, [scope_name, box.name, t_name, None, 0])


    @copydoc(Database.write_boxes)
    def write_boxes(self, boxes, scope_name=None):
        with self.conn:
            if boxes.scope is not None:
                if scope_name is not None and scope_name != boxes.scope.name:
                    raise ValueError('scope name mismatch')
                scope_name = boxes.scope.name
            for box in boxes.values():
                self.write_box(box, scope_name)

    def log(self, message, level=logging.INFO):
        """
        Log a message into the SQLite database

        Args:
            message (str): A message to log, will be stored verbatim.
            level (int): A logging level, can be used to filter messages.
        """
        message = str(message)
        with self.conn:
            cur = self.conn.cursor()
            cur.execute("INSERT INTO ema_log(level, content) VALUES (?,?)", [level, str(message)])
        _logger.log(level, message)

    def merge_log(self, other):
        """
        Merge the log from another SQLiteDB into this database log.

        Args:
            other (emat.SQLiteDB): Source of log to merge.
        """
        if not hasattr(other, 'conn'):
            return
        with self.conn:
            with other.conn:
                selfc = self.conn.cursor()
                otherc = other.conn.cursor()
                other_q = f"""
                SELECT
                    timestamp, level, content 
                FROM
                    ema_log 
                ORDER BY
                    rowid
                """
                for timestamp, level, content in otherc.execute(other_q):
                        selfc.execute(
                            """
                            INSERT INTO ema_log(timestamp, level, content) 
                            SELECT ?1, ?2, ?3
                            WHERE NOT EXISTS(
                                SELECT 1 FROM ema_log 
                                WHERE timestamp=?1 AND level=?2 AND content=?3
                            )
                            """,
                            [timestamp, level, content],
                        )

                    # if not selfc.execute(
                    #     "SELECT count(*) FROM ema_log WHERE timestamp=? AND level=? AND content=?",
                    #     [timestamp, level, content],
                    # ).fetchall()[0][0]:
                    #     selfc.execute(
                    #         "INSERT INTO ema_log(timestamp, level, content) VALUES (?,?,?)",
                    #         [timestamp, level, content],
                    #     )

    def print_log(self, file=None, limit=20, order="DESC", level=logging.INFO, like=None):
        """
        Print logged messages from the SQLite database

        Args:
            file (file-like object):
                Where to print the output, defaults to the
                current sys.stdout.
            limit (int or None, default 20):
                Maximum number of messages to print
            order ({'DESC','ASC'}):
                Print log messages in descending or ascending
                chronological order.
            level (int):
                A logging level, only messages logged at this
                level or higher are printed.
        """
        if order.upper() not in ("ASC", "DESC"):
            raise ValueError("order must be ASC or DESC")
        if not isinstance(level, int):
            raise ValueError("level must be integer")
        if like:
            qry = f"""
            SELECT
                datetime(timestamp, 'localtime'), content 
            FROM
                ema_log 
            WHERE
                level >= {level} 
                AND content LIKE '{like}'
            ORDER BY
                timestamp {order}, rowid {order}
            """
        else:
            qry = f"""
            SELECT
                datetime(timestamp, 'localtime'), content 
            FROM
                ema_log 
            WHERE
                level >= {level} 
            ORDER BY
                timestamp {order}, rowid {order}
            """
        if limit is not None:
            qry += f" LIMIT {limit}"
        with self.conn:
            cur = self.conn.cursor()
            for row in cur.execute(qry):
                print(" - ".join(row), file=file)


    def mark_run_invalid(
            self,
            run_id
    ):
        """
        Mark a particular run_id as invalid.

        Args:
            run_id (str): The run to mark as invalid.
        """
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
            UPDATE 
                ema_experiment_run
            SET
                run_valid = FALSE
            WHERE
                run_id = @run_id
            ''', dict(run_id=run_id))
            if cursor.rowcount:
                self.log(f"MARKED AS INVALID RUN_ID {run_id}")
            else:
                _logger.warn(f"FAILED TO MARK AS INVALID RUN_ID {run_id}")


    def merge_database(
            self,
            other,
            force=False,
            dryrun=False,
            on_conflict='ignore',
            max_diffs_in_log=50,
    ):
        """
        Merge results from another database.

        Only results from a matching identical scope are
        merged.  Metamodels are copied without checking
        if they are duplicate.

        Args:
            other (emat.Database):
                The other database from which to draw data.
            force (bool, default False):
                By default, database results are merged only
                when the scopes match exactly between the current
                and `other` database files.  Setting `force` to
                `True` will merge results from the `other`
                database file as long as the scope names match,
                the names of the the scoped parameters match,
                and there is some overlap in the names of the
                scoped performance measures.
            dryrun (bool, default False):
                Allows a dry run of the merge, to check how many
                experiments would be merged, without actually
                importing any data into the current database.
            on_conflict ({'ignore','replace'}, default 'ignore'):
                When corresponding performance measures for the
                same experiment exist in both the current and
                `other` databases, the merge will either ignore
                the updated value or replace it with the value
                from the other database.  This only applies to
                conflicts in performance measures; conflicts in
                parameters always result in distinct experiments.
            max_diffs_in_log (int, default 50):
                When there are fewer than this many changes in
                a set of experiments, the individual experiment_ids
                that have been changed are reported.

        """
        assert isinstance(other, Database)
        from ...util.deduplicate import count_diff_rows, report_diff_rows
        other_db_path = getattr(other, 'database_path', None)
        if other_db_path:
            self.log(f"merging from database at {other_db_path}")
        for scope_name in other.read_scope_names():
            if scope_name not in self.read_scope_names():
                self.log(f"not merging scope name {scope_name}, name does not match current database")
                continue
            scope_self = self.read_scope(scope_name)
            scope_other = other.read_scope(scope_name)
            if force:
                # Force merge if parameter names match, and there are some measures in common
                if set(scope_self.get_parameter_names()) != set(scope_other.get_parameter_names()):
                    self.log(f"not merging scope name {scope_name}, parameter names do not match current database")
                    continue
                common_measure_names = list(set(scope_self.get_measure_names()) & set(scope_other.get_measure_names()))
                if len(common_measure_names) == 0:
                    self.log(f"not merging scope name {scope_name}, measure names do not overlap current database")
                    continue
            else:
                # Unforced merge only when scopes are identical
                if scope_self != scope_other:
                    self.log(f"not merging scope name {scope_name}, content does not match current database")
                    continue
                common_measure_names = scope_self.get_measure_names()

            self.log(f"merging scope name {scope_name}")

            # transfer metamodels
            source_mapping = {}
            other_metamodel_ids = other.read_metamodel_ids(scope_name)
            for other_metamodel_id in other_metamodel_ids:
                other_metamodel = other.read_metamodel(scope_name, other_metamodel_id)
                new_id = self.get_new_metamodel_id(scope_name)
                other_metamodel.metamodel_id = new_id
                if not dryrun:
                    self.write_metamodel(
                        scope_name,
                        metamodel=other_metamodel,
                        metamodel_id=new_id,
                    )
                source_mapping[other_metamodel_id] = new_id

            # transfer experiments
            design_names = other.read_design_names(scope_name)
            for design_name in design_names:
                xl_df = other.read_experiment_parameters(scope_name, design_name)
                proposed_design_name = design_name
                design_match = False
                experiment_id_map = None
                if proposed_design_name in self.read_design_names(scope_name):
                    xl_df_self = self.read_experiment_parameters(scope_name, design_name)
                    if not xl_df.reset_index(drop=True).equals(
                            xl_df_self.reset_index(drop=True)
                    ):
                        n = 2
                        while proposed_design_name in self.read_design_names(scope_name):
                            proposed_design_name = f"{design_name}_{n}"
                            n += 1
                    else:
                        design_match = True
                        experiment_id_map = pd.Series(data=xl_df_self.index, index=xl_df.index)
                if not design_match:
                    # transfer parameters
                    if proposed_design_name == design_name:
                        self.log(f"experiment parameters updates for {scope_name}/{design_name}")
                    else:
                        self.log(f"experiment parameters updates for {scope_name}/{design_name} -> {proposed_design_name}")
                    if not dryrun:
                        self.write_experiment_parameters(scope_name, proposed_design_name, xl_df)
                sources = other.read_experiment_measure_sources(scope_name, design_name)
                # transfer measures
                for source in sources:
                    if source == 0:
                        # source is core model, make updates but do not overwrite non-null values
                        m_df0 = self.read_experiment_measures(scope_name, design_name, source=source)
                        m_df1 = other.read_experiment_measures(scope_name, design_name, source=source)[common_measure_names]
                        if experiment_id_map is not None:
                            m_df1.index = m_df1.index.map(experiment_id_map)
                        df0copy = m_df0.copy()
                        if on_conflict == 'replace':
                            m_df = m_df1.combine_first(m_df0)
                        else:
                            m_df = m_df0.combine_first(m_df1)
                        n_diffs = count_diff_rows(m_df, df0copy)
                        self.log(f"experiment measures {n_diffs} updates "
                                 f"for {scope_name}/{design_name}")
                        if n_diffs < max_diffs_in_log:
                            list_diffs, list_adds, list_drops = report_diff_rows(m_df, df0copy)
                            if list_diffs:
                                self.log(f"  {scope_name}/{design_name} experiment measures updated: "
                                         +", ".join(str(_) for _ in list_diffs))
                            if list_adds:
                                self.log(f"  {scope_name}/{design_name} experiment measures added: "
                                         +", ".join(str(_) for _ in list_adds))
                            if list_drops: # should be none
                                self.log(f"  {scope_name}/{design_name} experiment measures dropped: "
                                         +", ".join(str(_) for _ in list_drops))
                        if not dryrun:
                            self.write_experiment_measures(scope_name, 0, m_df)
                    else: # source is not core model, just copy
                        m_df = other.read_experiment_measures(scope_name, design_name, source=source)[common_measure_names]
                        if not dryrun:
                            self.write_experiment_measures(scope_name, source_mapping[source], m_df)

        # merge logs
        if isinstance(other, SQLiteDB):
            self.merge_log(other)
