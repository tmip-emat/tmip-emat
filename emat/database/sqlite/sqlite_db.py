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

from . import sql_queries as sq
from ..database import Database
from ...util.deduplicate import reindex_duplicates

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
        initialize (bool, default False):
            Whether to initialize emat database file.  The value of this argument
            is ignored if `database_path` is not given (as in-memory databases
            must always be initialized).

    """

    def __init__(self, database_path: str=":memory:", initialize: bool=False, readonly=False):

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

        if self.database_path == ":memory:":
            initialize = True
        # in order:
        self.modules = {}
        if initialize:
            self.conn = self.__create(["emat_db_init.sql", "meta_model.sql"], wipe=True)
        elif readonly:
            self.conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
        else:
            self.conn = self.__create(["emat_db_init.sql", "meta_model.sql"], wipe=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.cursor().execute(sq.SET_VERSION_DATABASE)
        atexit.register(self.conn.close)


    def __create(self, filenames, wipe=False):
        """
        Call sql files to create sqlite database file
        """
       
        # close connection and delete file if exists
        if self.database_path != ":memory:" and wipe:
            self.__delete_database()
        try:
            conn = sqlite3.connect(self.database_path)
        except sqlite3.OperationalError as err:
            raise sqlite3.OperationalError(f'error on connecting to {self.database_path}') from err
        with conn:
            cur = conn.cursor()

            for filename in filenames:
                _logger.info("running script " + filename)
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

    def update_database(self):
        """
        Update database for compatability with tmip-emat 0.4
        """
        with self.conn:
            cur = self.conn.cursor()
            for u in sq.UPDATE_DATABASE:
                try:
                    cur.execute(u)
                except:
                    _logger.error(f"SQL Query:\n{u}\n")
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
        return cloudpickle.loads(gzip.decompress(blob))

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
            design_name (str or None): experiment design name.  Set to None
                to find experiments across all designs. If the experiment is
                not found and no design_name is given, it will be assigned to
                a design named 'ad_hoc'.
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
        ex_id = self.read_experiment_id(scope_name, *args, **kwargs)
        if ex_id is None:
            parameters = self.read_scope(scope_name).get_parameter_defaults()
            for a in args:
                parameters.update(a)
            parameters.update(kwargs)
            df = pd.DataFrame(parameters, index=[0])
            ex_id = self.write_experiment_parameters(scope_name, None, df)[0]
        return ex_id

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
        return result

    def write_experiment_measures(
            self,
            scope_name,
            source,
            m_df,
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

            for m in scp_m:
                if m[0] in m_df.columns:
                    for ex_id, value in m_df[m[0]].iteritems():
                        # index is experiment id
                        try:
                            cur.execute(
                                    sq.INSERT_EX_M,
                                    [ex_id, value, source, m[0]])
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
                        cur.execute(
                            sq.INSERT_EX_M,
                            [ex_id, m_value, source, m[0]])
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

        scope_name = self._validate_scope(scope_name, 'design_name')
        if design_name is None:
            if experiment_id is None:
                sql = sq.GET_EX_M_ALL
                arg = [scope_name]
                if source is not None:
                    sql += ' AND ema_experiment_measure.measure_source =?2'
                    arg.append(source)
            else:
                sql = sq.GET_EX_M_BY_ID_ALL
                arg = [scope_name, experiment_id]
                if source is not None:
                    sql += ' AND ema_experiment_measure.measure_source =?3'
                    arg.append(source)
        else:
            if experiment_id is None:
                sql = sq.GET_EXPERIMENT_MEASURES
                arg = [scope_name, design_name]
                if source is not None:
                    sql = sql.replace("/*source*/", ' AND ema_experiment_measure.measure_source =?3')
                    arg.append(source)
            else:
                sql = sq.GET_EXPERIMENT_MEASURES_BY_ID
                arg = [scope_name, design_name, experiment_id]
                if source is not None:
                    sql = sql.replace("/*source*/", ' AND ema_experiment_measure.measure_source =?4')
                    arg.append(source)
        cur = self.conn.cursor()
        ex_m = pd.DataFrame(cur.execute(sql, arg).fetchall())
        if ex_m.empty is False:
            ex_m = ex_m.pivot(index=0, columns=1, values=2)
        ex_m.index.name = 'experiment'
        ex_m.columns.name = None

        column_order = (
                self.read_measures(scope_name)
        )

        return ex_m[[i for i in column_order if i in ex_m.columns]]

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

    def write_experiment_all(
            self,
            scope_name,
            design_name,
            source,
            xlm_df,
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

    def print_log(self, file=None, limit=20, order="DESC", level=logging.INFO):
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
        qry = f"SELECT datetime(timestamp, 'localtime'), content FROM ema_log WHERE level >= {level} ORDER BY timestamp {order}"
        if limit is not None:
            qry += f" LIMIT {limit}"
        with self.conn:
            cur = self.conn.cursor()
            for row in cur.execute(qry):
                print(" - ".join(row), file=file)
