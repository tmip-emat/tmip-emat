# -*- coding: utf-8 -*-

from typing import List, Union, Mapping
import yaml
import os, sys, time, errno
from shutil import copyfile, copy, copytree
import glob
import pandas as pd
import numpy as np
from ...scope.scope import Scope
from ...database.database import Database
from .core_files import FilesCoreModel
from .parsers import TableParser, loc, iloc, loc_sum
from ...util.docstrings import copydoc
from .core_files import copy_model_outputs_1, copy_model_outputs_ext

from ...util.loggers import get_module_logger
_logger = get_module_logger(__name__)


class ODOTModel(FilesCoreModel):
    """
    Setup connections and paths to GBNRTC core model

    Args:
        configuration:
            The configuration for this
            core model. This can be passed as a dict, or as a str
            which gives the filename of a YAML file that will be
            loaded.
        scope:
            The exploration scope, as a Scope object or as
            a str which gives the filename of a YAML file that will be
            loaded.
        safe:
            Load the configuration YAML file in 'safe' mode.
            This can be disabled if the configuration requires
            custom Python types or is otherwise not compatible with
            safe mode. Loading configuration files with safe mode
            off is not secure and should not be done with files from
            untrusted sources.
        db:
            An optional Database to store experiments and results.
        name:
            A name for this model, given as an alphanumeric string.
            The name is required by ema_workbench operations.
            If not given, "GBNRTC" is used.

    """




    def __init__(self,
                 configuration:Union[str,Mapping],
                 scope: Union[Scope, str],
                 safe:bool=True,
                 db:Database=None,
                 name:str='GBNRTC',
                 ):
        super().__init__(
                 configuration=configuration,
                 scope=scope,
                 safe=safe,
                 db=db,
                 name=name,
        )


    def copyeverything(self, src, dst):
        try:
          copytree(src, dst)
        except OSError as exc: # python >2.5
          if exc.errno == errno.ENOTDIR:
              copy(src, dst)
          else: raise

    
    def setup(self, design, scen, path):
        """
        Configure the core model with the experiment variable values

        Args:
            design data table: experiment variables including both exogenous
                uncertainty and policy levers
            scen: the scenario number to be setup within the full design
            path: the starting (clean) model path provided by "SOABM_model_config.yaml"    

        """
        
        # New Scenario directory 
        NewScen = path + '_' + str(scen) 
        
        # check if new directory already exists, and if so don't worry about creating
        if not os.path.isdir(NewScen):
          # Run the scenario to copy the reference directory
          self.copyeverything(path,NewScen) 

        # change working directory to new scenario
        os.chdir(NewScen)

        # Save a csv copy of the experimental parameters      
        design.iloc[scen].to_csv('Emat_Parameters.csv',header=False)
                
        # Call the model run function, which call the bat file
        #self.run()
        
          
    def run(self):
        """
        Runs the SOABM bat file
        
        Model should be prepared using `setup` first.

        """

        _logger.info("Starting model run at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))        

        os.system("RunModel.bat") 
        
    def post_process(self,
                     params: dict,
                     measure_names: List[str],
                     output_path=None):
        """
        Runs post processors associated with measures.

        For the GBNRTC model, this method calls
        TransCAD macros to generate output files.

        The model should have previously been executed using
        the `run` method.

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

        _logger.info("Running post process scripts at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))
        
        if self.tc is None:
            self.start_transcad()
        
        if output_path is None:
            output_path = self.model_path
            
        output_path = output_path.replace("\\", "\\\\") + "\\\\"
        
        pm_done = []
        for pm in measure_names:
            # skip if performance measure handled by other macro
            if pm in pm_done: continue
            
            try:
                func, macro = self.__TRANSCAD_MACRO_BY_PM[pm]
                pm_done += func(self, macro, params, output_path)
            except KeyError:
                _logger.exception(f"Post process method for pm {pm} not available")
                raise
    
    def archive(self, params: dict, model_results_path: str, experiment_id:int=0):
        """
        Copies TransCAD settings, outputs and common input files

        Args:
            params (dict): Dictionary of experiment variables
            model_results_path (str): archive path
            experiment_id (int, optional): The id number for this experiment.

        """

        # create output folder
        if not os.path.exists(model_results_path):
            os.makedirs(model_results_path)
            time.sleep(2)
            os.makedirs(os.path.join(model_results_path, "ModelConfig"))
            time.sleep(2)
            os.makedirs(os.path.join(model_results_path, "Outputs"))
            time.sleep(2)

        # record experiment definitions
        xl_df = pd.DataFrame(params, index=[experiment_id])
        xl_df.to_csv(model_results_path + '.csv')

        self.tc.RunMacro(
            "G30 File Close All",
            self.exp_param_ui)  # good idea incase TransCAD is holding any cards

        # copy model settings
        self.tc.RunMacro("Copy Model Parameters", self.exp_param_ui,
                         self.mod_path_tc +"\\\\",
                         model_results_path.replace("\\", "\\\\") + "\\\\")

        copy_model_outputs_ext(self.model_path, model_results_path, "AM_LinkVolumes")
        copy_model_outputs_ext(self.model_path, model_results_path, "PM_LinkVolumes")
        copy_model_outputs_ext(self.model_path, model_results_path, "MD_LinkVolumes")
        copy_model_outputs_ext(self.model_path, model_results_path, "NT_LinkVolumes")
        copy_model_outputs_ext(self.model_path, model_results_path, "TASN_ONO_pkwk")
        copy_model_outputs_ext(self.model_path, model_results_path, "TASN_ONO_opwk")
        copy_model_outputs_ext(self.model_path, model_results_path, "TASN_ONO_pkdr")
        copy_model_outputs_ext(self.model_path, model_results_path, "TASN_ONO_opdr")
        copy_model_outputs_ext(self.model_path, model_results_path, "pktrips")

        copy_model_outputs_1(self.model_path, model_results_path, "pkdr.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "pkwk.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "opdr.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "opwk.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "skim_walk.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "skim_hwypk.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "skim_hwyop.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "AM_hwytrips.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "PM_hwytrips.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "ModeChoice_Daily_Sum_Trips_pk.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "ModeChoice_Daily_Sum_Trips_op.mtx")
        copy_model_outputs_1(self.model_path, model_results_path, "msa_log.txt")
        
        # copy other files to support performance measures
        for file in glob.glob(
                os.path.join(
                    self.model_path,
                    "EMAExperimentFiles", "PerfMeasSupport", "*"
                )
        ):
            copy(file, os.path.join(model_results_path, "Outputs"))
            
        # copy output summaries (all csv's)
        for file in glob.glob(
                os.path.join(self.model_path, "Outputs", "*.csv")
        ):
            copy(file, os.path.join(model_results_path, "Outputs"))

    # =============================================================================
    #     Experiment variable setting methods
    # =============================================================================

    def __set_simple_evar(self, macro, evar, exp_var):       
        ''' call TransCAD macro and return variable to completed list'''
        self.tc.RunMacro("G30 File Close All", self.exp_param_ui)  
        ret = self.tc.RunMacro(macro, 
                               self.exp_param_ui, 
                               self.mod_path_tc,
                               float(exp_var[evar]))
        if ret != 0:
            raise SystemError("Error {2} in setting {0} to {1}" 
                              .format(evar, exp_var[evar], ret)) 
        return [evar]
        
    

     
    # =============================================================================
    #   Performance measure processing methods
    # =============================================================================
    
    def __pp_path(self, macro, exp_var, out_path):
        '''call TransCAD macro with the path argument'''
        self.tc.RunMacro("G30 File Close All", self.exp_param_ui)  
        self.tc.RunMacro(macro, 
                         self.perf_meas_ui, 
                         out_path)
        return self.__PM_BY_TRANSCAD_MACRO[macro][1]

    def __pp_path_kens(self, macro, exp_var, out_path):
        '''call TransCAD macro with path and kensington arguments'''
        kens = int(exp_var['Kensington Decommissioning'])
                
        self.tc.RunMacro("G30 File Close All", self.exp_param_ui)  
        
        self.tc.RunMacro(macro, 
                         self.perf_meas_ui, 
                         out_path,
                         kens)
        return self.__PM_BY_TRANSCAD_MACRO[macro][1]    
    
  
     
