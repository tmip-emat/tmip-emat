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
                 name:str='ODOT',
                 ):
        super().__init__(
                 configuration=configuration,
                 scope=scope,
                 safe=safe,
                 db=db,
                 name=name,
        )

        self._parsers = self.__MEASURE_PARSERS

    def copyeverything(self, src, dst):
        try:
          copytree(src, dst)
        except OSError as exc: # python >2.5
          if exc.errno == errno.ENOTDIR:
              copy(src, dst)
          else: raise
              

    def setup(self, params: dict):
        """
        Configure the core model with the experiment variable values

        Args:
            params (dict): experiment variables including both exogenous
                uncertainty and policy levers

        Raises:
            KeyError: if experiment variable defined is not supported               
        """
        # Run the scenario to copy the reference directory
        self.copyeverything(self.config['model_ref'],self.model_path) 

        # change working directory to new scenario
        os.chdir(self.model_path)

        # Save a csv copy of the experimental parameters      
        pd.Series(params).to_csv('Emat_Parameters.csv',header=False)
                
        # Call the bat file to update ABM input files based on EMAT scenarios
        os.system("EMAT_Inputs.bat")
        
          
    def run(self):
        """
        Runs the SOABM bat file
        
        Model should be prepared using `setup` first.

        """

        _logger.info("Starting model run at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))        

        os.system("RunModel.bat") 
        
    def run_mid(self):
        """
        Runs the SOABM bat file
        
        Model should be prepared using `setup` first.

        """

        _logger.info("Re-Starting model run at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))        

        os.system("RunModel_Mid.bat") 
        
    def post_process(self,
                     params: dict,
                     measure_names: List[str],
                     output_path=None):
 
        """
        Runs the EMAT_Process bat file
        The bat file runs an R script which creates all the 
        outputs developed for the TMIP-EMAT evaulation.
        
        Assumes setup and run have been compelted.

        """

        _logger.info("Starting model post process at at {0}".format(time.strftime("%Y_%m_%d %H%M%S")))        

        os.system("EMAT_Process.bat") 
        
        # back out a folder for the working directory so that the working folder can be renamed
        os.chdir("..")
    
    
    def archive(self, params: dict, model_results_path: str, experiment_id:int=0):
        """
        Re-name Scenario folder to 

        Args:
            params (dict): Dictionary of experiment variables
            model_results_path (str): archive path
            experiment_id (int, optional): The id number for this experiment.

        """

        # rename the model / sandbox directory to the experiment number / id
        os.rename(self.model_path,self.model_path + '_' + str(experiment_id))


    # final list to push all csv results to TMIP-EMAT

    __MEASURE_PARSERS = [

        TableParser(
            "Access.csv",
            {
                'Percentage of Population with Access to 50k Jobs by Car within 5mins in PM': loc['Per_Pop_w_50K_Jobs_in_5_mins', 'x'],
                'Percentage of Population with Access to 50k Jobs by Car within 10mins in PM': loc['Per_Pop_w_50K_Jobs_in_10_mins', 'x'],
                'Percentage of Population with Access to 50k Jobs by Car within 20mins in PM': loc['Per_Pop_w_50K_Jobs_in_20_mins', 'x'],
                'Percentage of Population with Access to 50k Jobs by Car within 30mins in PM': loc['Per_Pop_w_50K_Jobs_in_30_mins', 'x'],
            }
        ),

        TableParser(
            "ModeSplit.csv",
            {
                'Auto SOV Mode Share': loc['AutoSOV', 'x'],
                'Auto 2 Passengers Mode Share': loc['Auto2Per', 'x'],
                'Auto 3 or More Passengers Mode Share': loc['Auto3Plus', 'x'],
                'Walk Mode Share': loc['Walk', 'x'],
                'Bike Mode Share': loc['Bike', 'x'],
                'Transit Mode Share': loc['Transit', 'x'],
                'PNR Mode Share': loc['PNR', 'x'],
                'KNR Mode Share': loc['KNR', 'x'],
                'School Bus Mode Share': loc['SchoolBus', 'x'],
                'Bike and Walk Mode Share': loc['Active', 'x'],
                'Transit with PNR and KNR Mode Share': loc['Transit_PNR_KNR', 'x'],
            }
        ),

        TableParser(
            "million_PMT.csv",
            {
                'Millions of Miles Traveled - Drive Alone Free': loc['DriveAloneFree', 'x'],
                'Millions of Miles Traveled - Shared 2 Person General Purpose': loc['Shared2GP', 'x'],
                'Millions of Miles Traveled - Shared 3 or more Person General Purpose': loc['Shared3GP', 'x'],
                'Millions of Miles Traveled - Walk': loc['Walk', 'x'],
                'Millions of Miles Traveled - Bike': loc['Bike', 'x'],
                'Millions of Miles Traveled - Transit': loc['Transit', 'x'],
                'Millions of Miles Traveled - PNR': loc['PNR', 'x'],
                'Millions of Miles Traveled - KNR': loc['KNR', 'x'],
                'Millions of Miles Traveled - School Bus': loc['SchoolBus', 'x'],
                'Millions of Person Miles Traveled': loc['Total', 'x'],
            },
        ),

        TableParser(
            "million_VMT.csv",
            {
                'Millions of Auto Miles Traveled in EA': loc['EA_VOL_AUTO', 'x'],
                'Millions of Truck Miles Traveled in EA': loc['EA_VOL_TRUCK', 'x'],
                'Millions of Vehicle Miles Traveled in EA': loc['EA_VOL_TOTAL', 'x'],
                'Millions of Auto Miles Traveled in AM': loc['AM_VOL_AUTO', 'x'],
                'Millions of Truck Miles Traveled in AM': loc['AM_VOL_TRUCK', 'x'],
                'Millions of Vehicle Miles Traveled in AM': loc['AM_VOL_TOTAL', 'x'],
                'Millions of Auto Miles Traveled in MD': loc['MD_VOL_AUTO', 'x'],
                'Millions of Truck Miles Traveled in MD': loc['MD_VOL_TRUCK', 'x'],
                'Millions of Vehicle Miles Traveled in MD': loc['MD_VOL_TOTAL', 'x'],
                'Millions of Auto Miles Traveled in PM': loc['PM_VOL_AUTO', 'x'],
                'Millions of Truck Miles Traveled in PM': loc['PM_VOL_TRUCK', 'x'],
                'Millions of Vehicle Miles Traveled in PM': loc['PM_VOL_TOTAL', 'x'],
                'Millions of Auto Miles Traveled in EV': loc['EV_VOL_AUTO', 'x'],
                'Millions of Truck Miles Traveled in EV': loc['EV_VOL_TRUCK', 'x'],
                'Millions of Vehicle Miles Traveled in EV': loc['EV_VOL_TOTAL', 'x'],
                'Millions of Auto Miles Traveled': loc['DAILY_VOL_AUTO', 'x'],
                'Millions of Truck Miles Traveled': loc['DAILY_VOL_TRUCK', 'x'],
                'Millions of Vehicle Miles Traveled': loc['DAILY_VOL_TOTAL', 'x'],
            }
        ),

        TableParser(
            "thousand_VHT.csv",
            {
                'Thousands of Auto Hours Traveled in EA': loc['EA_AUTO_VHT', 'x'],
                'Thousands of Truck Hours Traveled in EA': loc['EA_TRUCK_VHT', 'x'],
                'Thousands of Vehicle Hours Traveled in EA': loc['EA_TOTAL_VHT', 'x'],
                'Thousands of Auto Hours Traveled in AM': loc['AM_AUTO_VHT', 'x'],
                'Thousands of Truck Hours Traveled in AM': loc['AM_TRUCK_VHT', 'x'],
                'Thousands of Vehicle Hours Traveled in AM': loc['AM_TOTAL_VHT', 'x'],
                'Thousands of Auto Hours Traveled in MD': loc['MD_AUTO_VHT', 'x'],
                'Thousands of Truck Hours Traveled in MD': loc['MD_TRUCK_VHT', 'x'],
                'Thousands of Vehicle Hours Traveled in MD': loc['MD_TOTAL_VHT', 'x'],
                'Thousands of Auto Hours Traveled in PM': loc['PM_AUTO_VHT', 'x'],
                'Thousands of Truck Hours Traveled in PM': loc['PM_TRUCK_VHT', 'x'],
                'Thousands of Vehicle Hours Traveled in PM': loc['PM_TOTAL_VHT', 'x'],
                'Thousands of Auto Hours Traveled in EV': loc['EV_AUTO_VHT', 'x'],
                'Thousands of Truck Hours Traveled in EV': loc['EV_TRUCK_VHT', 'x'],
                'Thousands of Vehicle Hours Traveled in EV': loc['EV_TOTAL_VHT', 'x'],
                'Thousands of Auto Hours Traveled': loc['DAILY_AUTO_VHT', 'x'],
                'Thousands of Truck Hours Traveled': loc['DAILY_TRUCK_VHT', 'x'],
                'Thousands of Vehicle Hours Traveled': loc['DAILY_TOTAL_VHT', 'x'],
            }
        ),

        TableParser(
            "percentVC_byFC_above90.csv",
            {
                'Percent of Interstate Miles over 90% V/C Ratio During the PM Peak': loc['1', 'x'],
                'Percent of Principal Arterial Miles over 90% V/C Ratio During the PM Peak': loc['3', 'x'],
                'Percent of Minor Arterial Miles over 90% V/C Ratio During the PM Peak': loc['4', 'x'],
                'Percent of Major Collector Miles over 90% V/C Ratio During the PM Peak': loc['5', 'x'],
                'Percent of Minor Collector Miles over 90% V/C Ratio During the PM Peak': loc['6', 'x'],
                'Percent of Local Road Miles over 90% V/C Ratio During the PM Peak': loc['7', 'x'],
                'Percent of Ramp Miles over 90% V/C Ratio During the PM Peak': loc['30', 'x'],
            },
        ),

        TableParser(
            "hhMeasures.csv",
            {
                'Number of Autos Owned Per Household': loc['AutosOwned', 'x'],
                'Percent of Non-Mandatory Tours': loc['PerNonMand', 'x'],
            }
        ),

    ]

     
