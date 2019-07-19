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
from typing import AbstractSet

from . import sql_queries as sq
from ..database import Database

from ...util.loggers import get_module_logger
_logger = get_module_logger(__name__)

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

    def __init__(self, database_path: str=":memory:", initialize: bool=False):

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
        self.filenames = [
            "scope.sql", "exp_design.sql", "meta_model.sql"
        ]
        self.modules = {}
        if initialize:
            self.conn = self.__create()
        else:
            self.conn = sqlite3.connect(database_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cur = self.conn.cursor()
        atexit.register(self.conn.close)


    def __create(self):
        """
        Call sql files to create sqlite database file
        """
       
        # close connection and delete file if exists
        if self.database_path != ":memory:":
            self.__delete_database()
        try:
            conn = sqlite3.connect(self.database_path)
        except sqlite3.OperationalError as err:
            raise sqlite3.OperationalError(f'error on connecting to {self.database_path}') from err
        cur = conn.cursor()

        for filename in self.filenames:
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
        
        conn.commit()

        return conn
            
    def __repr__(self):
        scopes = self.read_scope_names()
        if len(scopes) == 1:
            return f'<emat.SQLiteDB with scope "{scopes[0]}">'
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
        # experiment variables - description and type (risk or strategy)
        for xl in parameter_list:
            self.cur.execute(sq.CONDITIONAL_INSERT_XL, xl)
            
        # performance measures - description
        for m in measure_list:
            self.cur.execute(sq.CONDITIONAL_INSERT_M, m)
            
        self.conn.commit()

    @copydoc(Database.write_scope)
    def write_scope(self, scope_name, sheet, scp_xl, scp_m, content=None):

        if content is not None:
            import gzip, cloudpickle
            blob = gzip.compress(cloudpickle.dumps(content))
        else:
            blob = None

        try:
            self.cur.execute(sq.INSERT_SCOPE, [scope_name, sheet, blob])
        except sqlite3.IntegrityError:
            raise KeyError(f'scope named "{scope_name}" already exists')
        
        for xl in scp_xl:
            self.cur.execute(sq.INSERT_SCOPE_XL, [scope_name, xl])
            if self.cur.rowcount < 1: 
                raise KeyError('Experiment Variable {0} not present in database'
                               .format(xl))
            
        for m in scp_m:
            self.cur.execute(sq.INSERT_SCOPE_M, [scope_name, m])    
            if self.cur.rowcount < 1: 
                raise KeyError('Performance measure {0} not present in database'
                               .format(m))
    
        self.conn.commit()

    @copydoc(Database.read_scope)
    def read_scope(self, scope_name):
        try:
            blob = self.cur.execute(sq.GET_SCOPE, [scope_name]).fetchall()[0][0]
        except IndexError:
            blob = None
        if blob is None:
            return blob
        import gzip, cloudpickle
        return cloudpickle.loads(gzip.decompress(blob))

    @copydoc(Database.write_metamodel)
    def write_metamodel(self, scope_name, metamodel=None, metamodel_id=None, metamodel_name=''):

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
            self.cur.execute(sq.INSERT_METAMODEL_PICKLE,
                         [scope_name, metamodel_id, metamodel_name, blob])
        except sqlite3.IntegrityError:
            raise KeyError(f'metamodel_id {metamodel_id} for scope "{scope_name}" already exists')

        self.conn.commit()


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

        name, blob = self.cur.execute(sq.GET_METAMODEL_PICKLE,
                               [scope_name, metamodel_id]).fetchall()[0]
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
        scope_name = self._validate_scope(scope_name, None)
        metamodel_ids = [i[0] for i in self.cur.execute(sq.GET_METAMODEL_IDS,
                                                         [scope_name] ).fetchall()]
        return metamodel_ids

    @copydoc(Database.get_new_metamodel_id)
    def get_new_metamodel_id(self, scope_name):
        scope_name = self._validate_scope(scope_name, None)
        metamodel_id = [i[0] for i in self.cur.execute(sq.GET_NEW_METAMODEL_ID,).fetchall()][0]
        self.write_metamodel(scope_name, None, metamodel_id)
        return metamodel_id


    @copydoc(Database.add_scope_meas)
    def add_scope_meas(self, scope_name, scp_m):
        scope_name = self._validate_scope(scope_name, None)

        # test that scope exists
        saved_m = self.cur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()
        if len(saved_m) == 0: 
            raise KeyError('named scope does not exist')
            
        for m in scp_m:
            if m not in saved_m:
                self.cur.execute(sq.INSERT_SCOPE_M, [scope_name, m])    
                if self.cur.rowcount < 1: 
                    raise KeyError('Performance measure {0} not present in database'
                                   .format(m))
    
        self.conn.commit()
        
    @copydoc(Database.delete_scope) 
    def delete_scope(self, scope_name):
        self.cur.execute(sq.DELETE_SCOPE, [scope_name])

    @copydoc(Database.write_experiment_parameters)
    def write_experiment_parameters(self, scope_name, design_name: str, xl_df: pd.DataFrame):
        scope_name = self._validate_scope(scope_name, 'design_name')
        # local cursor because we'll depend on lastrowid
        fcur = self.conn.cursor()
        
        # get list of experiment variables - except "one"     
        scp_xl = fcur.execute(sq.GET_SCOPE_XL, [scope_name]).fetchall()
        if len(scp_xl) == 0:
            raise UserWarning('named scope {0} not found - experiments will \
                                  not be recorded'.format(scope_name))

        ex_ids = []

        for index, row in xl_df.iterrows():
            # create new experiment and get id
            fcur.execute(sq.INSERT_EX, [design_name, scope_name])
            ex_id = fcur.lastrowid
            ex_ids.append(ex_id)
                
            # set each from experiment defitinion 
            for xl in scp_xl:
                try:
                    value = row[xl[0]]
                    fcur.execute(sq.INSERT_EX_XL, [ex_id, value, xl[0]])
                except TypeError:
                    _logger.error(f'Experiment definition missing {xl[0]} variable')
                    raise

        self.conn.commit()
        fcur.close()
        return ex_ids

    def read_experiment_id(self, scope_name, design_name: str, *args, **kwargs):
        """Read the experiment id previously defined in the database

        Args:
            scope_name (str): scope name, used to identify experiments,
                performance measures, and results associated with this run
            design_name (str or None): experiment design name.  Set to None
                to find experiments across all designs.
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
        if design_name is not None and not isinstance(design_name, str):
            parameters.update(design_name)
            design_name = None
        for a in args:
            parameters.update(a)
        parameters.update(kwargs)
        xl_df = pd.DataFrame(parameters, index=[0])
        result = self.read_experiment_ids(scope_name, design_name, xl_df)
        return result[0]

    @copydoc(Database.read_experiment_ids)
    def read_experiment_ids(self, scope_name, design_name: str, xl_df: pd.DataFrame):

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
                    if design_name is None:
                        possible_ids = set([i[0] for i in fcur.execute(
                            sq.GET_EXPERIMENT_IDS_BY_VALUE,
                            [scope_name, par_name, par_value],
                        ).fetchall()])
                    else:
                        possible_ids = set([i[0] for i in fcur.execute(
                            sq.GET_EXPERIMENT_IDS_BY_DESIGN_AND_VALUE,
                            [scope_name, design_name, par_name, par_value],
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
            import warnings
            warnings.warn(f'missing {missing_ids} ids')
        return ex_ids

    def read_all_experiment_ids(self, scope_name:str, design_name:str=None):
        """Read the experiment ids previously defined in the database

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
        
        
        scope_name = self._validate_scope(scope_name, 'design_name')
        if design_name is None:
            experiment_ids = [i[0] for i in self.cur.execute(sq.GET_EXPERIMENT_IDS_ALL,
                                                             [scope_name] ).fetchall()]
        else:
            experiment_ids = [i[0] for i in self.cur.execute(sq.GET_EXPERIMENT_IDS_IN_DESIGN,
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

    @copydoc(Database.read_experiment_parameters)
    def read_experiment_parameters(self, scope_name: str, design:str=None, only_pending:bool=False)-> pd.DataFrame:

        scope_name = self._validate_scope(scope_name, 'design')

        if only_pending:
            if design is None:
                xl_df = pd.DataFrame(self.cur.execute(
                    sq.GET_EX_XL_ALL_PENDING, [scope_name, ]).fetchall())
            else:
                xl_df = pd.DataFrame(self.cur.execute(
                    sq.GET_EX_XL_PENDING, [scope_name, design]).fetchall())
        else:
            if design is None:
                xl_df = pd.DataFrame(self.cur.execute(
                        sq.GET_EX_XL_ALL, [scope_name, ]).fetchall())
            else:
                xl_df = pd.DataFrame(self.cur.execute(
                        sq.GET_EX_XL, [scope_name, design]).fetchall())
        if xl_df.empty is False:
            xl_df = xl_df.pivot(index=0, columns=1, values=2)
        xl_df.index.name = 'experiment'
        xl_df.columns.name = None

        column_order = (
                self.read_constants(scope_name)
                + self.read_uncertainties(scope_name)
                + self.read_levers(scope_name)
        )

        return xl_df[[i for i in column_order if i in xl_df.columns]]

    @copydoc(Database.write_experiment_measures)
    def write_experiment_measures(self,
                   scope_name,
                   source: int,
                   m_df: pd.DataFrame):
        scope_name = self._validate_scope(scope_name, None)
        scp_m = self.cur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()
        
        if len(scp_m) == 0:
            raise UserWarning('named scope {0} not found - experiments will \
                                  not be recorded'.format(scope_name))        

        for m in scp_m:
            if m[0] in m_df.columns:
                for ex_id, value in m_df[m[0]].iteritems():
                    # index is experiment id
                    try:
                        self.cur.execute(
                                sq.INSERT_EX_M,
                                [ex_id, value, source, m[0]])
                    except:
                        _logger.error(f"Error saving {value} to m {m[0]} for ex {ex_id}")
                        raise

        self.conn.commit()

    def write_ex_m_1(self,
                     scope_name,
                     source: int,
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
        scope_name = self._validate_scope(scope_name, None)
        scp_m = self.cur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()

        if len(scp_m) == 0:
            raise UserWarning('named scope {0} not found - experiments will \
                                  not be recorded'.format(scope_name))

        for m in scp_m:
            if m[0] == m_name:
                try:
                    self.cur.execute(
                        sq.INSERT_EX_M,
                        [ex_id, m_value, source, m[0]])
                except:
                    _logger.error(f"Error saving {m_value} to m {m[0]} for ex {ex_id}")
                    raise


    @copydoc(Database.read_experiment_all)
    def read_experiment_all(self, scope_name, design_name, only_pending=False) ->pd.DataFrame:
        scope_name = self._validate_scope(scope_name, 'design')
        if design_name is None:
            ex_xlm = pd.DataFrame(self.cur.execute(sq.GET_EX_XLM_ALL,
                                                   [scope_name,]).fetchall())
        elif isinstance(design_name, str):
            ex_xlm = pd.DataFrame(self.cur.execute(sq.GET_EX_XLM,
                                                   [scope_name,
                                                    design_name]).fetchall())
        else:
            ex_xlm = pd.concat([
                pd.DataFrame(self.cur.execute(sq.GET_EX_XLM, [scope_name, dn]).fetchall())
                for dn in design_name
            ])
        if ex_xlm.empty is False:
            ex_xlm = ex_xlm.pivot(index=0, columns=1, values=2)
        ex_xlm.index.name = 'experiment'
        ex_xlm.columns.name = None

        if only_pending:
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
        return ex_xlm[[i for i in column_order if i in ex_xlm.columns]]

    @copydoc(Database.read_experiment_measures)
    def read_experiment_measures(self, scope_name: str, design: str, experiment_id=None) ->pd.DataFrame:
        scope_name = self._validate_scope(scope_name, 'design')
        if design is None:
            if experiment_id is None:
                ex_m = pd.DataFrame(
                    self.cur.execute(
                        sq.GET_EX_M_ALL,
                        [scope_name]
                    ).fetchall())
            else:
                ex_m = pd.DataFrame(
                    self.cur.execute(
                        sq.GET_EX_M_BY_ID_ALL,
                        [scope_name, experiment_id]
                    ).fetchall())
        else:
            if experiment_id is None:
                ex_m = pd.DataFrame(
                    self.cur.execute(
                        sq.GET_EX_M,
                        [scope_name, design]
                    ).fetchall())
            else:
                ex_m = pd.DataFrame(
                    self.cur.execute(
                        sq.GET_EX_M_BY_ID,
                        [scope_name, design, experiment_id]
                    ).fetchall())
        if ex_m.empty is False:
            ex_m = ex_m.pivot(index=0, columns=1, values=2)
        ex_m.index.name = 'experiment'
        ex_m.columns.name = None

        column_order = (
                self.read_measures(scope_name)
        )

        return ex_m[[i for i in column_order if i in ex_m.columns]]

    @copydoc(Database.delete_experiments)
    def delete_experiments(self, scope_name: str, design: str):
        scope_name = self._validate_scope(scope_name, 'design')
        self.cur.execute(sq.DELETE_EX, [scope_name, design])
        self.conn.commit()
        
    @copydoc(Database.write_experiment_all)
    def write_experiment_all(self,
                     scope_name, 
                     design: str, 
                     source: int,
                     xlm_df: pd.DataFrame):
        scope_name = self._validate_scope(scope_name, 'design')
        fcur = self.conn.cursor()
        
        exist = pd.DataFrame(fcur.execute(sq.GET_EX_XLM, 
                                          [scope_name,
                                          design]).fetchall())
        if exist.empty is False:
            raise UserWarning('scope {0} with design {1} found \
                                  must be deleted before recording'
                                  .format(scope_name, design))
        
        # get list of experiment variables     
        scp_xl = fcur.execute(sq.GET_SCOPE_XL, [scope_name]).fetchall()
        scp_m = fcur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()
        
        for index, row in xlm_df.iterrows():
            # create new experiment and get id
            fcur.execute(sq.INSERT_EX, [design, scope_name])
            ex_id = fcur.lastrowid
        
             # set each from experiment defitinion 
            for xl in scp_xl:
                try:
                    xl_value = row[xl[0]]
                    fcur.execute(sq.INSERT_EX_XL, 
                                 [ex_id, xl_value, xl[0]])
                except TypeError:
                    _logger.error(f'Experiment definition missing {xl[0]} variable')
                    raise
              
            for m in scp_m:
                if m[0] in xlm_df.columns:
                    m_value = row[m[0]]
                    self.cur.execute(
                        sq.INSERT_EX_M, [ex_id, m_value, source, m[0]])
        
        fcur.close()
        self.conn.commit()


        
    @copydoc(Database.read_scope_names)
    def read_scope_names(self, design_name=None) -> list:
        if design_name is None:
            scopes = [i[0] for i in self.cur.execute(sq.GET_SCOPE_NAMES ).fetchall()]
        else:
            scopes = [i[0] for i in self.cur.execute(sq.GET_SCOPES_CONTAINING_DESIGN_NAME,
                                                     [design_name] ).fetchall()]
        return scopes

    @copydoc(Database.read_design_names)
    def read_design_names(self, scope_name:str) -> list:
        scope_name = self._validate_scope(scope_name, None)
        designs = [i[0] for i in self.cur.execute(sq.GET_DESIGN_NAMES, [scope_name] ).fetchall()]
        return designs

    @copydoc(Database.read_uncertainties)
    def read_uncertainties(self, scope_name:str) -> list:
        scope_name = self._validate_scope(scope_name, None)
        scp_x = self.cur.execute(sq.GET_SCOPE_X, [scope_name]).fetchall()
        return [i[0] for i in scp_x]

    @copydoc(Database.read_levers)
    def read_levers(self, scope_name:str) -> list:
        scope_name = self._validate_scope(scope_name, None)
        scp_l = self.cur.execute(sq.GET_SCOPE_L, [scope_name]).fetchall()
        return [i[0] for i in scp_l]

    @copydoc(Database.read_constants)
    def read_constants(self, scope_name:str) -> list:
        scope_name = self._validate_scope(scope_name, None)
        scp_c = self.cur.execute(sq.GET_SCOPE_C, [scope_name]).fetchall()
        return [i[0] for i in scp_c]

    @copydoc(Database.read_measures)
    def read_measures(self, scope_name: str) -> list:
        scope_name = self._validate_scope(scope_name, None)
        scp_m = self.cur.execute(sq.GET_SCOPE_M, [scope_name]).fetchall()
        return [i[0] for i in scp_m]

    @copydoc(Database.read_box)
    def read_box(self, scope_name: str, box_name: str, scope=None):
        scope_name = self._validate_scope(scope_name, None)

        from ...scope.box import Box
        parent = self.read_box_parent_name(scope_name, box_name)
        box = Box(name=box_name, scope=scope,
                  parent=parent)

        for row in self.cur.execute(sq.GET_BOX_THRESHOLDS, [scope_name, box_name]):
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
        scope_name = self._validate_scope(scope_name, None)
        names = self.cur.execute(sq.GET_BOX_NAMES, [scope_name]).fetchall()
        return [i[0] for i in names]

    @copydoc(Database.read_box_parent_name)
    def read_box_parent_name(self, scope_name: str, box_name:str):
        scope_name = self._validate_scope(scope_name, None)
        try:
            return self.cur.execute(sq.GET_BOX_PARENT_NAME, [scope_name, box_name]).fetchall()[0][0]
        except IndexError:
            return None

    @copydoc(Database.read_box_parent_names)
    def read_box_parent_names(self, scope_name: str):
        scope_name = self._validate_scope(scope_name, None)
        names = self.cur.execute(sq.GET_BOX_PARENT_NAMES, [scope_name]).fetchall()
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

        p_ = set(self.read_uncertainties(scope_name) + self.read_levers(scope_name))
        m_ = set(self.read_measures(scope_name))

        if box.parent_box_name is None:
            self.cur.execute(sq.INSERT_BOX, [scope_name, box.name])
        else:
            self.cur.execute(sq.INSERT_SUBBOX, [scope_name, box.name,
                                                box.parent_box_name])

        for t_name, t_vals in box.thresholds.items():

            if t_name in p_:
                sql_cl = sq.CLEAR_BOX_THRESHOLD_P
                sql_in = sq.SET_BOX_THRESHOLD_P
            elif t_name in m_:
                sql_cl = sq.CLEAR_BOX_THRESHOLD_M
                sql_in = sq.SET_BOX_THRESHOLD_M
            else:
                import warnings
                warnings.warn(f"{t_name} not identifiable as parameter or measure")
                continue

            self.cur.execute(sql_cl, [scope_name, box.name, t_name])

            if isinstance(t_vals, Bounds):
                if t_vals.lowerbound is not None:
                    self.cur.execute(sql_in, [scope_name, box.name, t_name, t_vals.lowerbound, -2])
                if t_vals.upperbound is not None:
                    self.cur.execute(sql_in, [scope_name, box.name, t_name, t_vals.upperbound, -1])
            elif isinstance(t_vals, AbstractSet):
                for n, t_val in enumerate(t_vals, start=1):
                    self.cur.execute(sql_in, [scope_name, box.name, t_name, t_val, n])
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
                import warnings
                warnings.warn(f"{t_name} not identifiable as parameter or measure")
                continue

            self.cur.execute(sql_cl, [scope_name, box.name, t_name])
            self.cur.execute(sql_in, [scope_name, box.name, t_name, None, 0])


    @copydoc(Database.write_boxes)
    def write_boxes(self, boxes, scope_name=None):
        if boxes.scope is not None:
            if scope_name is not None and scope_name != boxes.scope.name:
                raise ValueError('scope name mismatch')
            scope_name = boxes.scope.name
        for box in boxes.values():
            self.write_box(box, scope_name)
        self.conn.commit()
