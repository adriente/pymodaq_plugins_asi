from pymodaq_plugins_point_electronic.hardware.revolon import Revolon
from pymodaq_plugins_asi.hardware.cheetah3 import Cheetah3, config

import json
import socket
import time
from numba import njit
import numpy as np
from typing import Any
from pymodaq_utils.logger import set_logger, get_module_name

# BUFFER_SIZE = 64000

logger = set_logger(get_module_name(__file__))

@njit
def fill_array(array,event_list) : 
    for value in event_list : 
        array[value] +=1

class Tp3toolsConfig:
    """
    Attributes
    ----------
    bin : bool
        Only for frame-based acquisition
        True : Full y binning (only the x coordinate is used)
        False : no binning
    bytedepth : int
        bit depth of the data output from tp3_tools server
        1 : uint8
        2 : uint16
        4 : uint32 default for spim (max spim size above 2896*2896*512)
        8 : uint64
    cumul : bool
        Are the data summed over time. Not relevant to Iumi
        True : Yes
        False : No
    mode : int
        0 : live 1D spectrum
        2 : live spectrum image (that's what we will use most at Iumi)
        3 : live 4D in frame-based mode
        6 : Orsay chrono acquisition
        7 : Yves' coincidence2D acquisition
        8 : Orsay chrono acquisition for frame-based
        10 : Live frame-based acquisition
        11 : Live spim frame-based acquisition (unclear : Live1DFrameHyperspec)
        12 : Live coincidence acquisition (Yves)
        13 : Live spim 4D acquisition
        14 : another spim live mode
        12 : Live 2D spim frame-based (unclear : Live2DFrameHyperspec)
    xspim_size : int
        x size of the spectrum image (different than xscan_size if Orsay subscan)
    yspim_size : int
        y size of the spectrum image (different than yscan_size if Orsay subscan)
    xscan_size : int
        x size of the scan (different than xspim_size if Orsay subscan)
    yscan_size : int
        y size of the scan (different than yspim_size if Orsay subscan)
    pixel_time : int
        pixel dwell time in us. 
        I am not sure it is useful for raster scan. I believe it is only used for custom-list scan.
    time_delay : int
        time delay used for coincidence experiments (ns unit ?)
    time_width : int
        time width used for coincidence experiments (ns unit ?)
    time_resolved : bool
        Only used for coincidence experiments. Checks if the electron is in the time window (see just above)
        True : yes
        False : no
    save_locally : bool
        Does it save the data in tpx3 on the ASI PC ?
        True : yes
        False : no
    pixel_mask : int
        Determines which pixel mask bpc file to use (Orsay). Useless at Iumi.
    video_time : int
        Corresponds to video_delay in tp3_tools. I don't know what it is for.
        Value in 1.56625 ns unit. 
    threshold : int 
        Determins which dacs file to use (Orsay). Useless at Iumi.
    bias_voltage : int
        Sets the Cheetah3 bias voltage (Orsay). Useless at Iumi.
    destination_port : int 
        Sets the ASI destination (Orsay). Useless at Iumi.
    acquisition_us : int
        frame-based acquisition time in us.
    sup0 : float
        supplementary paramter, used as metadata in the json saved with the .tpx3 data if save_locally is True.
    sup1 : float
        Same as sup0.
    """

    def __init__(self):
        self.bin = False
        self.bytedepth = 4
        self.cumul = False
        self.mode = 2
        self.xspim_size = 0
        self.yspim_size = 0
        self.xscan_size = 0
        self.yscan_size = 0
        self.pixel_time = 0
        self.time_delay = 0
        self.time_width = 0
        self.time_resolved = False
        self.save_locally = False
        self.pixel_mask = 0
        self.video_time = 0
        self.threshold = 0
        self.bias_voltage = 0
        self.destination_port = 0
        self.acquisition_us = 1000
        self.sup0 = 0.0
        self.sup1 = 0.0

    def create_configuration_bytes(self):
        return json.dumps(self.__dict__).encode()

class ScanCheetah3(Cheetah3) :
    def __init__(self):
        super().__init__()
        self.tp3tools_config = Tp3toolsConfig()
        self._init_client()
        self._xspim_size = 64
        self._yspim_size = 64
        self._data = np.zeros((self._xspim_size*self._yspim_size*(self.x_size+1),))
        self._cumul_num = 1
    
    def _init_client(self) -> None :
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(5)
        self.address = (config("CHEETAH3","scan","server_address"),
                        config("CHEETAH3","scan","server_port"))
        
    def send_config_bytes(self) -> None :
        self.client.connect(self.address)
        config_bytes = self.tp3tools_config.create_configuration_bytes()
        self.client.send(config_bytes)
        
    def reset_data(self) :
        self._data = np.zeros((self._xspim_size*self._yspim_size*(self.x_size+1),))
        
    def estimate_scan_time(self, pixel_dwell_time : float) -> float :
        return self.xspim_size*self.yspim_size*pixel_dwell_time 

    @property
    def xspim_size(self) -> int :
        return self._xspim_size

    @xspim_size.setter
    def xspim_size(self,value : int) -> None :
        self.tp3tools_config.xspim_size = value
        self.tp3tools_config.xscan_size = value
        self._data = np.zeros((self._xspim_size*self._yspim_size*(self._x_size+1),))
        self._xspim_size = value

    @property
    def yspim_size(self) -> int :
        return self._yspim_size

    @yspim_size.setter
    def yspim_size(self,value : int) -> None :
        self.tp3tools_config.yspim_size = value
        self.tp3tools_config.yscan_size = value
        self._data = np.zeros((self._xspim_size*self._yspim_size*(self._x_size+1),))
        self._yspim_size = value
        
    @property
    def cumul_num(self) -> int : 
        return self._cumul_num
    
    @cumul_num.setter
    def cumul_num(self, value : int) -> None :
        self._cumul_num = value 

    # def start(self,timeout = 0.0) :
    #     # diffrent ways depending on destination name : if tp3tools go to scan, else use the super().
    #     if 'scan' in self.destination_profiles : 
            
    #     else : 
    #         super().start(timeout = timeout)
    
    def start(self, mode = 'continuous') -> None:
        """Perform acquisition

        Keyword arguments:
        serverurl -- the URL of the running SERVAL (string)
        
        Parameters
        ----------
        timeout : float
            time until camera stop is automatically called
        """
        if 'scan' in self.destination_profiles :
            self.set_detector_config(ntriggers=self.ntriggers, trigger_mode=mode)
            self.set_destination(profile_list=self.destination_profiles)
            self.send_config_bytes()
            response = self.get_request(url=self.serverurl + '/measurement/start')
            logger.info('Response of acquisition start: %s', response.text)
        else :
            super().start(mode=mode)
            
    def stop(self):
        super().stop()
        self.client.shutdown(socket.SHUT_RDWR)
        self.client.close()
        
        
        
        
    
        
    
    
        
    