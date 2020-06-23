# -*- coding: utf-8 -*-
import yaml
import os
import time
import inspect
import pandas

from typing import Union, Mapping, Callable, Collection
from ..workbench.connectors.excel import ExcelModel

from ..scope.scope import Scope
from ..database.database import Database
from ..model.core_model import AbstractCoreModel


from ..util.docstrings import copydoc



class ExcelCoreModel(AbstractCoreModel, ExcelModel):
    """
    Interface class for a core model in Excel.

    Args:
        wd (Path-like):
            The working directory for the excel model.  This should be
            the directory in which the model resides.  Any supplementary
            data files required for the excel model should also be in
            this directory, and these files should be linked using only
            relative paths.  No other files should appear in this directory,
            as the entire directory will (potentially) be replicated multiple
            times during model execution.
        model_file (str):
            The file name of the excel model.  A file by this name should
            appear in `wd`.
        configuration (str, dict, or None):
            The configuration for this core model. This can be
            passed as a dict, or as a str which gives the filename
            of a yaml file that will be loaded.
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
        db (Database): An optional Database to store experiments and results.
        name (str): A name for this model, given as an alphanumeric string.
            The name is required by ema_workbench operations.
            If not given, the name of the function is extracted, or
            failing that, "EMAT" is used.
        metamodel_id: An identifier for this model, if it is a meta-model.
            Defaults to 0 (i.e., not a meta-model).

    Excel models are *only* available on the Windows operating system.
    Although Excel is also available on other operating systems (i.e.,
    macOS) the necessary automated control API in Python is only
    available on Windows.

    It is also important to note that Excel-based models are processed
    including the entire working directory given by the `wd` argument in
    the constructor, not just based on an Excel workbook in isolation.
    """

    def __init__(self,
                 wd,
                 model_file,
                 configuration: Union[str, Mapping, None] = None,
                 scope: Union[Scope, str] = None,
                 safe: bool = True,
                 db: Database = None,
                 name: str = 'EMAT',
                 metamodel_id=None,
                 ):
        if scope is None:
            raise ValueError('must give scope')

        if name == 'EMAT':
            try:
                _name = os.path.splitext(os.path.split(model_file)[1])[0]
            except:
                pass
            else:
                if _name.isalnum():
                    name = _name
                elif _name.replace("_", "").replace(" ", "").replace(".", "").isalnum():
                    name = _name.replace("_", "").replace(" ", "").replace(".", "")

        AbstractCoreModel.__init__(self, configuration, scope, safe, db, metamodel_id=metamodel_id)

        self.archive_path = self.config.get('archive_path', None)

        if self.archive_path is not None:
            os.makedirs(self.archive_path, exist_ok=True)

        # If no archive path is given, a temporary directory is created.
        # All archive files will be lost when this ExcelCoreModel is deleted.
        if self.archive_path is None:
            import tempfile
            self._temp_archive = tempfile.TemporaryDirectory()
            self.archive_path = self._temp_archive.name

        pointers = {i.name: i.address for i in self.scope._x_list if i.address is not None}
        pointers.update({i.name: i.address for i in self.scope._l_list if i.address is not None})
        pointers.update({i.name: i.address for i in self.scope._c_list if i.address is not None})
        pointers.update({i.name: i.address for i in self.scope._m_list if i.address is not None})

        for k,v in pointers.items():
            if v == 'SKIP':
                pointers[k] = None

        ExcelModel.__init__(self, name, wd=wd, model_file=model_file,
                            default_sheet=None, pointers=pointers, model_def=None)

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
        metamodel_tag = "" if self.metamodel_id == 0 else f", metamodel_id={self.metamodel_id}"
        return f'<emat.PythonCoreModel "{self.name}"{metamodel_tag} with {", ".join(content)}>'

    @copydoc(AbstractCoreModel.setup)
    def setup(self, params):
        """This method is not needed for Excel models."""

    @copydoc(AbstractCoreModel.get_experiment_archive_path)
    def get_experiment_archive_path(self, experiment_id=None, makedirs=False, parameters=None):
        """This method is not needed for Excel models."""

    @copydoc(AbstractCoreModel.run)
    def run(self):
        """This method is not needed for Excel models."""

    @copydoc(AbstractCoreModel.post_process)
    def post_process(self, params, measure_names, output_path=None):
        """This method is not needed for Excel models."""

    @copydoc(AbstractCoreModel.load_measures)
    def load_measures(
            self,
			measure_names: Collection[str]=None,
			*,
			rel_output_path=None,
			abs_output_path=None,
    ):
        """This method is not needed for Excel models."""

    @copydoc(AbstractCoreModel.archive)
    def archive(self, params, model_results_path=None, experiment_id=0):
        """This method is not needed for Excel models."""
