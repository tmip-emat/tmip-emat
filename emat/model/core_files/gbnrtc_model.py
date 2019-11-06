# -*- coding: utf-8 -*-

from typing import List, Union, Mapping
import yaml
import os, sys, time
from shutil import copyfile, copy
import glob
import pandas as pd
import numpy as np
from ...scope.scope import Scope
from ...database.database import Database
from .core_files import FilesCoreModel
from .parsers import TableParser, loc, iloc, loc_sum
from ...util.docstrings import copydoc
from .core_files import copy_model_outputs_1, copy_model_outputs_ext

try:
    import emat.model.core_files.caliper as cp
except ImportError:
    cp = None

from ...util.loggers import get_module_logger
_logger = get_module_logger(__name__)


class GBNRTCModel(FilesCoreModel):
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

    tc = None
    
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

        self.exp_param_ui = self.config['exp_param_ui']
        self.tdm_ui = self.config['tdm_ui']
        self.perf_meas_ui = self.config['perf_meas_ui']
        self.tc_ver_path = self.config['tc_version_path']

        # build reverse hash table to lookup post-processing macros given a performance measure
        self.__TRANSCAD_MACRO_BY_PM = {}
        for k, v in self.__PM_BY_TRANSCAD_MACRO.items():
            for pm in v[1]:
                self.__TRANSCAD_MACRO_BY_PM[pm] = v[0], k

        # TransCAD's path formatting requirements
        self.mod_path_tc = self.model_path.replace("\\", "\\\\") + "\\\\"

        self._parsers = self.__MEASURE_PARSERS


    def start_transcad(self):
        """
        Launch TransCAD and call methods to reset scenario
        """
        # wack transCAD if it is open (cannot restart model)
        _logger.warn("[(re)starting TransCAD instances]")
        os.system("TASKKILL /F /IM tcw.exe")
        time.sleep(10)
        os.startfile(self.tc_ver_path + 'tcw.exe')
        time.sleep(20)

        self.tc = cp.Gisdk(
            "TransCAD",
            log_file=self.model_path + "TC_log.txt",
            search_path=self.tc_ver_path)
        
        if self.tc is None:
            _logger.error("ERROR: failed to attach to a TransCAD instance")
            sys.exit()
    
    def setup(self, params: dict):
        """
        Configure the core model with the experiment variable values

        Args:
            params (dict): experiment variables including both exogenous
                uncertainty and policy levers

        Raises:
            KeyError: if experiment variable defined is not supported
                by the core model
        """

        self.start_transcad()
        # initilize scenario output space
        ret = self.tc.RunMacro("Restore Defaults", 
                               self.exp_param_ui, 
                               self.mod_path_tc)

        if ret != 0:
            raise SystemError("Error in restore defaults {0}".format(ret))

        variables_done = ['constant']
        for xl in params:
            # skip if variable handled as part of a combined setting
            if xl in variables_done: continue

            _logger.info(f"\t\t\tSetting experiment variable for {xl} to {params[xl]}")
            
            try:
                func, macro = self.__METHOD_MACRO_BY_EVAR[xl]
                variables_done += func(self, macro, xl, params)
            except KeyError:
                _logger.exception("Experiment variable method not available")
                raise
          
    def run(self):
        """
        Launches TransCAD and runs the model

        Model should be prepared using `setup` first.

        Raises:
            UserWarning: If model is not properly setup.
        """

        _logger.info("Starting model run at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))
        
        if self.tc is None:
            raise UserWarning("ERROR: failed to attach to a TransCAD instance")

        self.tc.RunMacro("G30 File Close All", self.tdm_ui)
        self.tc.RunMacro("GBNRTC Model - AutoRun 2050", self.tdm_ui)   
        
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
        
        _logger.info("Archiving model runs at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))
        if self.tc is None:
            self.start_transcad()        

        # create output folder
        if not os.path.exists(model_results_path):
            os.makedirs(model_results_path)
            time.sleep(2)
            os.makedirs(os.path.join(model_results_path, "TAZ"))
            time.sleep(2)
            os.makedirs(os.path.join(model_results_path, "Network"))
            time.sleep(2)
            os.makedirs(os.path.join(model_results_path, "Inputs"))
            time.sleep(2)
            os.makedirs(os.path.join(model_results_path, "Inputs\\Model"))
            time.sleep(2)
            os.makedirs(os.path.join(model_results_path, "Outputs"))
            time.sleep(2)

        # record experiment definitions
        xl_df = pd.DataFrame(list(params.items()),columns=['variable','value'])
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
            copy(file, os.path.join(model_results_path, 
                                    "EMAExperimentFiles", "PerfMeasSupport"))
            
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

    
    def __set_bailey_transit_strategy(self, macro, evar, exp_var): 
        ''' call TransCAD macro and return both strategies to completed list'''
        # bailey transit and micro mobility strategies are interdependent
        tserv = exp_var['Bailey Transit Service']
        mobhub = int(exp_var['Bailey Mobility Hubs'])
        mmob =  int(exp_var['Bailey Micro-Mobility'])
        
        ttime = 0
        thead = 0
        
        if tserv == 'improved headway':
            thead = 1
        elif tserv == 'traffic priority':
            ttime = 1
        elif tserv == 'brt':
            ttime = 1
            thead = 1
            

        self.tc.RunMacro("G30 File Close All", self.exp_param_ui)  
        ret = self.tc.RunMacro(macro, 
                               self.exp_param_ui, 
                               self.mod_path_tc,
                               ttime, mobhub, mmob)
        
        if ret != 0:
            raise SystemError("Error in setting Bailey travel time / Mobility Hubs to {0}/{1}" 
                              .format(ttime, mobhub)) 
            
        self.tc.RunMacro("G30 File Close All", self.exp_param_ui)  
        ret = self.tc.RunMacro("Headway", 
                               self.exp_param_ui, 
                               self.mod_path_tc,
                               thead)
        
        if ret != 0:
            raise SystemError("Error in setting Bailey service headway component to {0}"
                              .format(thead))             
            
        return ['Bailey Transit Service', 'Bailey Mobility Hubs', 'Bailey Micro-Mobility']
        
    
    def __set_kenslrt_strategy(self, macro, evar, exp_var): 
        ''' call TransCAD macro and return both strategies to completed list'''
        # kensington and lrt strategy are interdependent
        kens = int(exp_var['Kensington Decommissioning'])
        lrt = int(exp_var['LRT Extension'])

        self.tc.RunMacro("G30 File Close All", self.exp_param_ui)  
        ret = self.tc.RunMacro(macro, 
                               self.exp_param_ui, 
                               self.mod_path_tc,
                               kens, lrt)
        
        ret = self.tc.RunMacro("VA", 
                               self.exp_param_ui, 
                               self.mod_path_tc,
                               float(exp_var['Shared Mobility']))
        if ret != 0:
            raise SystemError("Error in setting LRT / Kensington to {0}/{1}" 
                              .format(lrt, kens)) 
            
        return ['Kensington Decommissioning', 'LRT Extension']

     
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
    
  
     

    # =============================================================================
    #   Hooks to macros and methods for GBNRTC
    # =============================================================================
                  
    # dictionary of methods and Macros by experiment variable
    __METHOD_MACRO_BY_EVAR = {
        'Land Use - CBD Focus':         (__set_simple_evar, "SE Data"),
        'Freeway Capacity':             (__set_simple_evar, "Roadway Capacity"),
        'Auto IVTT Sensitivity':        (__set_simple_evar, "IVTT Discount"),
        'Shared Mobility':              (__set_simple_evar, "VA"),
        'Kensington Decommissioning':   (__set_kenslrt_strategy, "Set POC Strategies"),
        'LRT Extension':                (__set_kenslrt_strategy, "Set POC Strategies"),
        'Bailey Land Use':              (__set_simple_evar, "Corridor_SED"),
        'Self Parking':                 (__set_simple_evar, "Terminal_Time"),
        'Weather Impacts':              (__set_simple_evar, "Weather_Capacity_Impact"),
        'Bailey Transit Service':       (__set_bailey_transit_strategy, "Transit_Strategies"),
        'Bailey Micro-Mobility':        (__set_bailey_transit_strategy, "Transit_Strategies"),
        'Bailey Mobility Hubs':         (__set_bailey_transit_strategy, "Transit_Strategies"),
        #'Bailey Transit Headway':       (__set_simple_evar, "Headway"),
        'Bailey Reduced Parking':       (__set_simple_evar, "Bailey_Parking"),        
        'Bailey Non-Motorized Facilities':       (__set_simple_evar, "Walk_Time")                
        
    }
    
    __PM_BY_TRANSCAD_MACRO = {
        'VMTVHT' : (__pp_path, 
                    ['Region-wide VMT','Interstate + Expressway + Ramp/Connector VMT',
                    'Major and Minor Arterials VMT','Total Auto VMT',
                    'Total Truck VMT', 'Bailey VMT', 'Bailey VHT', 
                    'Bailey Delay AM','Bailey Delay MD','Bailey Delay PM','Bailey Delay NT']),
        'TripLength' : (__pp_path, 
                        ['AM Trip Time (minutes)','AM Trip Length (miles)',
                        'PM Trip Time (minutes)','PM Trip Length (miles)']),
        'Boardings' : (__pp_path, 
                       ['Peak Walk-to-transit Boarding','Off-Peak Walk-to-transit Boarding',
                       'Peak Drive-to-transit Boarding','Off-Peak Drive-to-transit Boarding',
                       'Total Transit Boardings','Peak Walk-to-transit LRT Boarding',
                       'Off-Peak Walk-to-transit LRT Boarding','Peak Drive-to-transit LRT Boarding',
                       'Off-Peak Drive-to-transit LRT Boarding','Total LRT Boardings']),
        'BaileyBoardings' : (__pp_path, 
                       ['Peak Walk-to-transit Bailey Route Boarding','Peak Drive-to-transit Bailey Route Boarding',
                       'Off-Peak Walk-to-transit Bailey Route Boarding', 'Off-Peak Drive-to-transit Bailey Route Boarding',
                       'Bailey corridor route ridership']),    
        'ModeShare' : (__pp_path, 
                       ['Peak Transit Share','Peak NonMotorized Share',
                       'Off-Peak Transit Share','Off-Peak NonMotorized Share',
                       'Daily Transit Share','Daily NonMotorized Share',
                       'Bailey Peak Transit Share','Bailey Peak NonMotorized Share',
                       'Bailey Off-Peak Transit Share','Bailey Off-Peak NonMotorized Share',
                       'Bailey Transit share','Bailey NonMotorized share',
                       'Regional trips to/from Bailey']),
        'HBW Transit Time' : (__pp_path, 
                              ['Number of Home-based work tours taking <= 45 minutes via transit']),
        'Accessibility to CBD' : (__pp_path, 
                                  ['Households within 30 min of CBD']),
        'Accessibility_Bailey' : (__pp_path, 
                                  ['Employment < 20 transit mins from Bailey']),                                  
        'Downtown to Airport Travel Time' : (__pp_path, 
                                             ['Downtown to Airport Travel Time']),
        'CorridorMeasures' : (__pp_path_kens, 
                              ['Kensington Daily VMT','Kensington Daily VHT',
                              'Kensington_OB PM VMT','Kensington_OB PM VHT',
                              'Kensington_IB AM VMT','Kensington_IB AM VHT',
                              '190 Daily VMT','190 Daily VHT',
                              '190_OB Daily VMT','190_OB Daily VHT',
                              '190_IB Daily VMT','190_IB Daily VHT',
                              '33_west Daily VMT','33_west Daily VHT',
                              'I90_south Daily VMT','I90_south Daily VHT']),
        'OD Flows' : (__pp_path, 
                      ['OD Volume District 1 to 1','OD Volume District 1 to 2',
                      'OD Volume District 1 to 3','OD Volume District 1 to 4',
                      'OD Volume District 1 to 5','OD Volume District 1 to 6',
                      'OD Volume District 1 to 7','OD Volume District 1 to 8',
                      'OD Volume District 1 to 9','OD Volume District 1 to 10'])
    }
    
    __MEASURE_PARSERS = [

        TableParser(
            "transit_board.csv",
            {
                'Peak Walk-to-transit Boarding': loc[0, 'pk_walk_board'],
                'Off-Peak Walk-to-transit Boarding': loc[0, 'op_walk_board'],
                'Peak Drive-to-transit Boarding': loc[0, 'pk_drive_board'],
                'Off-Peak Drive-to-transit Boarding': loc[0, 'op_drive_board'],
                'Total Transit Boardings': loc[0, 'total'],
            }
        ),

        TableParser(
            "transit_board_rail.csv",
            {
                'Peak Walk-to-transit LRT Boarding': loc[0, 'pk_walk_board'],
                'Off-Peak Walk-to-transit LRT Boarding': loc[0, 'op_walk_board'],
                'Peak Drive-to-transit LRT Boarding': loc[0, 'pk_drive_board'],
                'Off-Peak Drive-to-transit LRT Boarding': loc[0, 'op_drive_board'],
                'Total LRT Boardings': loc[0, 'total'],
            }
        ),
        
        TableParser(
            "transit_board_bailey.csv",
            {
                'Peak Walk-to-transit Bailey Route Boarding': loc[0, 'pk_walk_board'],
                'Off-Peak Walk-to-transit Bailey Route Boarding': loc[0, 'op_walk_board'],
                'Peak Drive-to-transit Bailey Route Boarding': loc[0, 'pk_drive_board'],
                'Off-Peak Drive-to-transit Bailey Route Boarding': loc[0, 'op_drive_board'],
                'Bailey corridor route ridership': loc[0, 'total'],
            }
        ),        

        TableParser(
            "SUMMARY_VMTVHT.csv",
            {
                'Region-wide VMT': loc_sum[:, 'Daily Total VMT'],
                'Total Auto VMT': loc_sum[:, 'Daily Auto VMT'],
                'Total Truck VMT': loc_sum[:, 'Daily Truck VMT'],
                'Interstate + Expressway + Ramp/Connector VMT': (
                        loc_sum[1:2, 'Daily Total VMT']
                        + loc_sum[6:8, 'Daily Total VMT']
                        + loc_sum[21:22, 'Daily Total VMT']),
                'Major and Minor Arterials VMT': (
                        loc_sum[3:4, 'Daily Total VMT']
                        + loc[23, 'Daily Total VMT']),
            },
            # The TransCAD macro that creates the file SUMMARY_VMTVHT.csv
            # does not attach a header row with column names, so we
            # need to give column names here to enable the label-based
            # loc getters above. The column names include the first
            # columns, which is a row identifier that will be used as
            # the index of the resulting loaded dataframe, and not column
            # data.
            names=[
                'identifier',
                'Daily Auto VMT', 'Daily Truck VMT',
                'Daily Total VMT', 'Daily Auto VHT',
                'Daily Truck VHT', 'Daily Total VHT','Daily Auto FFVHT',
                'Daily Truck FFVHT', 'Daily Total FFVHT',
                'AM Auto VMT', 'AM Truck VMT', 'AM Total VMT',
                'MD Auto VMT', 'MD Truck VMT', 'MD Total VMT',
                'PM Auto VMT', 'PM Truck VMT', 'PM Total VMT',
                'NT Auto VMT', 'NT Truck VMT', 'NT Total VMT',
                'AM Auto VHT', 'AM Truck VHT', 'AM Total VHT',
                'MD Auto VHT', 'MD Truck VHT', 'MD Total VHT',
                'PM Auto VHT', 'PM Truck VHT', 'PM Total VHT',
                'NT Auto VHT', 'NT Truck VHT', 'NT Total VHT',
                'AM Auto FFVHT', 'AM Truck FFVHT', 'AM Total FFVHT',
                'MD Auto FFVHT', 'MD Truck FFVHT', 'MD Total FFVHT',
                'PM Auto FFVHT', 'PM Truck FFVHT', 'PM Total FFVHT',
                'NT Auto FFVHT', 'NT Truck FFVHT', 'NT Total FFVHT'
            ],
            index_col=0,
            dtype={'identifier':int},
        ),

        TableParser(
            "SELECTION_VMTVHT.csv",
            {
                'Bailey VMT': loc_sum[:, 'Daily Total VMT'],
                'Bailey VHT': loc_sum[:, 'Daily Total VHT'],
                'Bailey Delay AM': loc_sum[:, 'AM Total VHT'],
                'Bailey Delay MD': loc_sum[:, 'MD Total VHT'],
                'Bailey Delay PM': loc_sum[:, 'PM Total VHT'],
                'Bailey Delay NT': loc_sum[:, 'NT Total VHT']                                    
            },
            # The TransCAD macro that creates the file SUMMARY_VMTVHT.csv
            # does not attach a header row with column names, so we
            # need to give column names here to enable the label-based
            # loc getters above.
            names=[
                'identifier',
                'Daily Auto VMT', 'Daily Truck VMT',
                'Daily Total VMT', 'Daily Auto VHT',
                'Daily Truck VHT', 'Daily Total VHT','Daily Auto FFVHT',
                'Daily Truck FFVHT', 'Daily Total FFVHT',
                'AM Auto VMT', 'AM Truck VMT', 'AM Total VMT',
                'MD Auto VMT', 'MD Truck VMT', 'MD Total VMT',
                'PM Auto VMT', 'PM Truck VMT', 'PM Total VMT',
                'NT Auto VMT', 'NT Truck VMT', 'NT Total VMT',
                'AM Auto VHT', 'AM Truck VHT', 'AM Total VHT',
                'MD Auto VHT', 'MD Truck VHT', 'MD Total VHT',
                'PM Auto VHT', 'PM Truck VHT', 'PM Total VHT',
                'NT Auto VHT', 'NT Truck VHT', 'NT Total VHT',
                'AM Auto FFVHT', 'AM Truck FFVHT', 'AM Total FFVHT',
                'MD Auto FFVHT', 'MD Truck FFVHT', 'MD Total FFVHT',
                'PM Auto FFVHT', 'PM Truck FFVHT', 'PM Total FFVHT',
                'NT Auto FFVHT', 'NT Truck FFVHT', 'NT Total FFVHT'
            ],
            index_col=0,
            dtype={'identifier':int},
        ),

        TableParser(
            "trip_length.csv",
            {
                'AM Trip Time (minutes)': loc[0, 'AM Trip Length (minute)'],
                'AM Trip Length (miles)': loc[0, 'AM Trip Length (mile)'],
                'PM Trip Time (minutes)': loc[0, 'PM Trip Length (minute)'],
                'PM Trip Length (miles)': loc[0, 'PM Trip Length (mile)'],
            }
        ),

        TableParser(
            "mode_share.csv",
            {
                "Peak Transit Share"          : loc[0, 'pk_transit_share'],
                "Peak NonMotorized Share"     : loc[0, 'pk_nonmot_share'],
                "Off-Peak Transit Share"      : loc[0, 'op_transit_share'],
                "Off-Peak NonMotorized Share" : loc[0, 'op_nonmot_share'],
                "Daily Transit Share"         : loc[0, 'dly_transit_share'],
                "Daily NonMotorized Share"    : loc[0, 'dly_nonmot_share'],
            }
        ),

        TableParser(
            "bailey_mode_share.csv",
            {
                "Bailey Peak Transit Share"          : loc[0, 'pk_transit_share'],
                "Bailey Peak NonMotorized Share"     : loc[0, 'pk_nonmot_share'],
                "Bailey Off-Peak Transit Share"      : loc[0, 'op_transit_share'],
                "Bailey Off-Peak NonMotorized Share" : loc[0, 'op_nonmot_share'],
                "Bailey Transit share"               : loc[0, 'dly_transit_share'],
                "Bailey NonMotorized share"          : loc[0, 'dly_nonmot_share'],
            }
        ),

        TableParser(
            "bailey_trip_summary.csv",
            {
                "Regional trips to/from Bailey"       : (loc[0, 'pk_total'] + 
                                                        loc[0, 'op_total'])
            }
        ),

        TableParser(
            "HH_within_30min.csv",
            {
                "Households within 30 min of CBD": loc[0, 'HHs within 30 min of CBD'],
            }
        ),

        TableParser(
            "EMP_within_20min_of_BLY.csv",
            {
                "Employment < 20 transit mins from Bailey": loc[0, 'EMPs within 20 min of BLY'],
            }
        ),

        TableParser(
            "HBW_within_45min_transit.csv",
            {
                "Number of Home-based work tours taking <= 45 minutes via transit": loc[0, 'HBW within 45 min'],
            }
        ),

        TableParser(
            "Downtown_Airport_Time.csv",
            {
                "Downtown to Airport Travel Time": loc[0, 'Downtown to Airport Time'],
            }
        ),

        TableParser(
            "District_OD.csv",
            {
                'OD Volume District 1 to 1' : iloc[0,0],
                'OD Volume District 1 to 2' : iloc[0,1],
                'OD Volume District 1 to 3' : iloc[0,2],
                'OD Volume District 1 to 4' : iloc[0,3],
                'OD Volume District 1 to 5' : iloc[0,4],
                'OD Volume District 1 to 6' : iloc[0,5],
                'OD Volume District 1 to 7' : iloc[0,6],
                'OD Volume District 1 to 8' : iloc[0,7],
                'OD Volume District 1 to 9' : iloc[0,8],
                'OD Volume District 1 to 10': iloc[0,9],
             }
        ),

        TableParser(
            "CorridorMeasures.csv",
            {
                'Kensington Daily VMT': loc['Kensington', ' VMT'],
                'Kensington Daily VHT': loc['Kensington', ' VHT'],
                'Kensington_OB PM VMT': loc['Kensington_OB', ' VMT'],
                'Kensington_OB PM VHT': loc['Kensington_OB', ' VHT'],
                'Kensington_IB AM VMT': loc['Kensington_IB', ' VMT'],
                'Kensington_IB AM VHT': loc['Kensington_IB', ' VHT'],
                '190 Daily VMT':        loc['190', ' VMT'],
                '190 Daily VHT':        loc['190', ' VHT'],
                '190_OB Daily VMT':     loc['190_OB', ' VMT'],
                '190_OB Daily VHT':     loc['190_OB', ' VHT'],
                '190_IB Daily VMT':     loc['190_IB', ' VMT'],
                '190_IB Daily VHT':     loc['190_IB', ' VHT'],
                '33_west Daily VMT':    loc['33_west', ' VMT'],
                '33_west Daily VHT':    loc['33_west', ' VHT'],
                'I90_south Daily VMT':  loc['I90_south', ' VMT'],
                'I90_south Daily VHT':  loc['I90_south', ' VHT'],
                'I-90 SL #1 Daily VMT': loc['i190_sel1', ' VMT'],
                'I-90 SL #1 Daily VHT': loc['i190_sel1', ' VHT'],
                'I-90 SL #2 Daily VMT': loc['i190_sel2', ' VMT'],
                'I-90 SL #2 Daily VHT': loc['i190_sel2', ' VHT'],
                'I-90 SL #3 Daily VMT': loc['i190_sel3', ' VMT'],
                'I-90 SL #3 Daily VHT': loc['i190_sel3', ' VHT'],
                'I-90 SL #4 Daily VMT': loc['i190_sel4', ' VMT'],
                'I-90 SL #4 Daily VHT': loc['i190_sel4', ' VHT'],
            },
            index_col=0,
            # The example archive data does not include the screenline
            # data items, so these will by default raise an error.
            # We can skip the error and just retrieve NaN values
            # using the handle_errors argument.
            handle_errors='nan',
        ),

    ]
