import numpy as np

from pymodaq_utils.utils import ThreadCommand
from pymodaq_data.data import DataToExport, Axis
from pymodaq_gui.parameter import Parameter
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import QThread
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins
from pymodaq_utils.logger import set_logger, get_module_name
import collections

from pymodaq_plugins_asi.hardware.cheetah3 import config
from pymodaq_plugins_asi.hardware.scan_cheetah3 import ScanCheetah3, fill_array
from pymodaq_plugins_asi.hardware.camera_utils import bin2d, get_bin_list

logger = set_logger(get_module_name(__file__))

################
# Code Outline #
################

# I. DAQ_2DViewer_Cheetah3
# I. 1. Parameters
# I. 2. Initialisation
# I. 3. Data acquisition and axes
# I. 4. Properties
# II. Callback class
# III. Local testing code

############################
# I. DAQ_2DViewer_Cheetah3 #
############################

class DAQ_NDViewer_ScanCheetah3(DAQ_Viewer_base):
    """ Instrument plugin class for the Cheetah3 camera. It is a frame-based implementation of the camera.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    * This plugin is compatible with the Cheetah3 (2025) camera from Amsterdam Scientific instruments.
    * It has been tested with the Cheetah3 (2025).
    * This plugin was tested with PyMoDAQ 5.1.x on a windows 10 system.
    * To run this plugin you need another computer that controls the camera through ASI's Serval software.

    Attributes:
    -----------
    controller: Cheetah3
        The particular object that allows the communication with the camera.
    x_axis: Axis
        The horizontal axis of the camera #TODO Implement the dispersive scale
    y_axis: Axis
        The vertical axis of the camera
    binning : str
        The full binning status, either None (2D data), Vertical or Horizontal (1D data)
  
    Notes
    -----
    Additional attributes for the asynchronous acquistion

    callback: Cheetah3Callback
        The callback object for asynchronous acquistion
    callback_thread : QThread
        The callback lives on a different thread.
    startup_callback_signal
        The callback emits a signal when data are ready.

    Comments were made on how it works and can be found by search CT{0-99}.
    """

    ####################
    # I. 1. Parameters #
    ####################

    params = comon_parameters + [
        {'title' : "Frame-based camera settings", 'name' : 'camera_settings', 'type' : 'group', 'expanded' : True, 'children' : [
            {'title' : 'Exposure time', 'name' : 'exposure_time', 'type' : 'float', 'value' : 0.5},
            {'title' : 'x binning', 'name' : 'x_binning', 'type' : 'list', 'value' : 1, 'limits' : [1]},
            {'title' : 'y binning', 'name' : 'y_binning', 'type' : 'list', 'value' : 1, 'limits' : [1]},
        ]},
        {'title' : 'Spim parameters', 'name' : 'scan_params', 'type' : 'group', 'children' : [
            {'title': 'Scan width', 'name': 'image_width', 'type': 'int', 'value': 512},
            {'title': 'Scan height', 'name': 'image_height', 'type': 'int', 'value': 512},
            {'title' : 'Dwell time (us)', 'name' : 'dwell_time', 'type' : 'int', 'value' : 10},
        ]},
        {'title' : 'File paths input', 'name' : 'file_paths', 'type' : 'group', 'expanded' : False, 'children' : [
            {'title' : 'Add bpc file path', 'name' : 'bpc_file_path', 'type' : 'str', 'value' : '/home/asi/'},
            {'title' : 'Add dacs file path', 'name' : 'dacs_file_path', 'type' : 'str', 'value' : '/home/asi/'},
            {'title' : 'Add save folder path', 'name' : 'save_folder_path', 'type' : 'str', 'value' : '/home/asi/'},
        ]},
        {'title' : 'Current file paths', 'name' : 'file_paths_lists', 'type' : 'group', 'expanded' : True, 'children' : [
            {'title' : 'bpc file paths', 'name' : 'bpc_file_paths_list', 'type' : 'list', 'value' : '', 'limits' : ['']},
            {'title' : 'dacs file paths', 'name' : 'dacs_file_paths_list', 'type' : 'list', 'value' : '', 'limits' : ['']},
            {'title' : 'save folder paths', 'name' : 'save_folder_paths_list', 'type' : 'list', 'value' : '', 'limits' : ['']},
        ]}            
    ]

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        # There is probably a better way to select params, e.g. based on groups. But the current version is simpler even though quite verbose.
        if param.name() == "exposure_time":
            self.controller.camera_controller.exposure_time = param.value()
        elif param.name() == 'x_binning' :
            self.x_binning = param.value()
            self.set_axes()
        elif param.name() == 'y_binning' :
            self.y_binning = param.value()
            self.set_axes()
        elif param.name() == 'bpc_file_path' :
            self.controller.camera_controller.cheetah3_config.add_bpc_file(param.value())
            # self.controller.cheetah3_config.refresh()
            self.settings.child('file_paths_lists','bpc_file_paths_list').setLimits(config("CHEETAH3","file_paths",'bpc'))
        elif param.name() == 'dacs_file_path' :
            self.controller.camera_controller.cheetah3_config.add_dacs_file(param.value())
            # self.controller.cheetah3_config.refresh()
            self.settings.child('file_paths_lists','dacs_file_paths_list').setLimits(config("CHEETAH3","file_paths",'dacs'))
        elif param.name() == 'save_folder_path' :
            self.controller.camera_controller.cheetah3_config.add_save_folder(param.value())
            # self.controller.cheetah3_config.refresh()
            self.settings.child('file_paths_lists','save_folder_paths_list').setLimits(config("CHEETAH3","file_paths",'data'))
        elif param.name() == 'destination' :
            self.controller.camera_controller.cheetah3_config.build_destination(param.value()["selected"])
        elif param.name() == 'bpc_file_paths_list' :
            self.controller.camera_controller.bpc_file = param.value()
        elif param.name() == 'dacs_file_paths_list' :
            self.controller.camera_controller.dacs_file = param.value()
        elif param.name() == 'save_folder_paths_list' :
            self.controller.camera_controller.save_folder = param.value()
        if param.name() == "image_width":
            self.controller.image_width = param.value()
            self.set_axes()
        if param.name() == "image_height" :
            self.controller.image_height = param.value()
            self.set_axes()
        if param.name() == "dwell_time" :
            self.controller.dwell_time = param.value()

    ########################
    # I. 1. Initialisation #
    ########################

    live_mode_available = True
    # CT01. We create a signal object to start the execution of the callback thread.
    callback_signal = QtCore.Signal(int)

    def ini_attributes(self):       
        # if config('CHEETAH3','scan','scan_engine') == "Revolon" :
        #     from pymodaq_plugins_point_electronic.hardware.revolon import Revolon
        #     # self.controller = Revolon(camera_controller = ScanCheetah3())
        #     self.controller : Revolon = None
        # else :
        #     raise NotImplementedError("No scan valid scan engine was provided.")
        self.controller = None
        self.x_axis = None
        self.y_axis = None
        self.x_binning = 1
        self.y_binning = 1
        self._x_size = 1
        self._y_size = 1

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        info = ""
        if self.is_master:
            if config('CHEETAH3','scan','scan_engine') == "Revolon" :
                from pymodaq_plugins_point_electronic.hardware.revolon import Revolon
                self.controller = Revolon(camera_controller = ScanCheetah3())
            else : 
                raise NotImplementedError("No scan valid scan engine was provided.")
            camera_initialized = self.controller.camera_controller.check_connection()
            connect_rc = self.controller.connect()
            if connect_rc == 0x0000000 #SUCCESS :
                scan_initialized = True
            else :
                info = f"Init failed (return code {connect_rc:08X})!"
                scan_initialized = False
            initialized = camera_initialized and scan_initialized
            info = "The DAQ_viewer ScanCheetah3 has successfully started"
            # CT02. An object (called callback), is instanciated.
            self.callback = ScanCheetah3Callback(self.controller.camera_controller,self.controller)
            # CT03. A thread object is created (callback_thread)
            self.callback_thread = QtCore.QThread()
            # CT04. The thread object is made ready to be executed parallel to the main thread
            self.callback.moveToThread(self.callback_thread)
            # CT05. The function to be called by the thread is the callback object
            self.callback_thread.callback = self.callback
            # CT06. We make the thread ready to execute
            self.callback_thread.start()
            # CT07. We connect the signal to the execution of data read-out from the detector
            self.callback_signal.connect(self.callback.readout)
            # CT08. We connect the signal of the callback to the execution of PyMoDAQ GUI to display data.
            self.callback.data_sig.connect(self.emit_data)
        else:
            initialized = False
            info = ''
            raise NotImplementedError
            

        profile_names = self.controller.camera_controller.cheetah3_config.destination_names_list()
        self.settings.addChild({'title' : 'Data destination', 'name' : 'destination', 'type' : 'itemselect', 'value' : dict(
            all_items = profile_names, selected =['live_preview']
        ), 'checkbox' : True})
        self._x_size = self.controller.camera_controller.x_size
        self._y_size = self.controller.camera_controller.y_size
        self.settings.child('camera_settings','x_binning').setLimits(get_bin_list(self.x_size))
        self.settings.child('camera_settings','y_binning').setLimits(get_bin_list(self.y_size))
        self.settings.child('file_paths_lists','bpc_file_paths_list').setLimits(config("CHEETAH3","file_paths",'bpc'))
        self.settings.child('file_paths_lists','dacs_file_paths_list').setLimits(config("CHEETAH3","file_paths",'dacs'))
        self.settings.child('file_paths_lists','save_folder_paths_list').setLimits(config("CHEETAH3","file_paths",'data'))

        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        self.controller.stop()
        self.controller.camera_controller.stop()

    ###################################
    # I. 3. Data acquisition and axes #
    ###################################
    
    def set_axes(self) :
        """
        Set the axes for display depending on binning values. Can change the representation from 2D to 1D or 0D.
        """
        data_x_axis = np.linspace(start= 0, stop = self.x_size//self.x_binning, num = self.x_size//self.x_binning)
        data_y_axis = np.linspace(start= 0, stop = self.y_size//self.y_binning, num = self.y_size//self.y_binning)
        data_xscan_axis = np.linspace(start = 0,
                                      stop = self.controller.image_width,
                                      num = self.controller.image_width)
        data_yscan_axis = np.linspace(start = 0,
                                      stop = self.controller.image_height,
                                      num = self.controller.image_height)
        data_escan_axis = np.linspace(start = 0,
                                      stop = 513,
                                      num = 513)
        self.xscan_axis = Axis(data=data_xscan_axis, label='scan pixels', units='', index=0)
        self.yscan_axis = Axis(data=data_yscan_axis, label='scan pixels', units='', index=1)
        self.escan_axis = Axis(data=data_escan_axis, label='camera pixels', units='', index=2)
        # Case 1 : full binning both directions -> 0D data
        if self.x_binning == self.x_size and self.y_binning == self.y_size :
            dummy_data = np.array([0.0])
        # Case 2 : full binning in the x direction -> 1D data
        elif self.x_binning == self.x_size and self.y_binning != self.y_size : 
            dummy_data = np.zeros((self.y_size//self.y_binning,))
            self.y_axis = Axis(data=data_y_axis, label='', units='', index=0)
        # Case 3 : full binning in the y direction -> 1D data
        elif self.y_binning == self.y_size and self.x_binning != self.x_size :
            dummy_data = np.zeros((self.x_size//self.x_binning,))
            self.x_axis = Axis(data=data_x_axis, label='', units='', index=0)
        # Case 4 : No full binning -> 2D data
        else :
            dummy_data = np.zeros((self.y_size//self.y_binning,self.x_size//self.x_binning))
            self.y_axis = Axis(data=data_y_axis, label='', units='', index=0)
            self.x_axis = Axis(data=data_x_axis, label='', units='', index=1)

        dfp = self.prepare_dfp(dummy_data)
        # Prepares the viewer
        self.dte_signal_temp.emit(DataToExport('Cheetah3',
                                               data=dfp))
        
    def prepare_dfp (self, data : np.ndarray) -> DataFromPlugins :
        """
        Prepares DataFromPlugins for display. Chooses the right axes and data size based on the data shape.
        
        Parameters
        ----------
        data : np.ndarray
            input data. It can be any shape up to 2D.
        """ 
        dfp = []
        
        # if 'scan' in self.controller.destination_profiles :
        xscan_size = self.controller.image_width
        yscan_size = self.controller.image_height
        scan_data = np.reshape(self.controller.camera_controller._data, shape=(xscan_size,yscan_size,513))
        dfp.append(DataFromPlugins('Spectrum image', data=[scan_data],
                                    axes=[self.xscan_axis,
                                        self.yscan_axis,
                                        self.escan_axis],
                                    nav_indexes=(0, 1),
                                    do_save=False, do_plot=True))
        if "live_preview" in self.controller.camera_controller.destination_profiles : 
            if self.x_binning == self.x_size and self.y_binning == self.y_size :
                dfp.append(DataFromPlugins(name = 'Cheetah3 full sum',
                                    data = data,
                                    dim = 'Data0D'))
            elif self.x_binning == self.x_size and self.y_binning != self.y_size :
                dfp.append(DataFromPlugins(name = 'Cheetah3 sum x',
                                    data = [np.atleast_1d(data)],
                                    dim = 'Data1D',axes=[self.y_axis]))
            elif self.y_binning == self.y_size and self.x_binning != self.x_size :
                dfp.append(DataFromPlugins(name = 'Cheetah3 sum y',
                                    data = [np.atleast_1d(data)],
                                    dim = 'Data1D',axes=[self.x_axis]))
            else :
                dfp.append(DataFromPlugins(name = 'Cheetah3',
                                    data = [np.atleast_1d(data)],
                                    dim = 'Data2D',axes=[self.x_axis, self.y_axis]))
        return dfp

    def emit_data(self,data : np.ndarray, end : bool) -> None :
        # Add a bool as arg so that I can pick finishing acquisition or current
        # Avant de broadcaster les données, il vaut mieux créer une copie pour éviter d'avoir des soucis de pointeur. Le reshape doit faire une copie à priori.
        """
            Fonction used to emit data obtained by callback.

            See Also
            --------
            daq_utils.ThreadCommand
        """
        # CT13. The callback emitted a signal to display data
        try:
            binned_data = bin2d(data,self.x_binning,self.y_binning)
            dfp = self.prepare_dfp(data=binned_data) 
            if end :                   
                self.dte_signal.emit(DataToExport('Cheetah3',
                                                data=dfp))
            else :                    
                self.dte_signal.emit(DataToExport('Cheetah3',
                                                data=dfp))
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), 'log']))

    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        self.controller.camera_controller.ntriggers = int(2e9)
        self.controller.camera_controller.xspim_size = self.controller.image_width
        self.controller.camera_controller.yspim_size = self.controller.image_height
        try:
            if 'scan' in self.controller.camera_controller.destination_profiles :
                if kwargs.get('live',False) :
                    
                    # self.scan_viewer.grab_data(**kwargs)
                    self.controller.start(num_frame=0)
                    self.controller.camera_controller.start(timeout=0.0)
                    self.callback_signal.emit(0)
                else :
                    self.controller.start(num_frame=1)
                    self.controller.camera_controller.start(timeout=0.0)
                    pixel_time_s = self.dwell_time/1e6
                    frame_number = round(self.controller.camera_controller.estimate_scan_time(pixel_time_s)/self.controller.camera_controller.exposure_time)
                    self.callback_signal.emit(frame_number)
            else : 
                if kwargs.get('live',False) :
                    self.controller.start(timeout = 0.0)
                    # CT9. We trigger the execution of the callback thread start_readout function. 
                    self.callback_signal.emit(0)

                else:
                    self.controller.start(timeout = 5.0)
                    self.callback_signal.emit(1)


        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [str(e), "log"]))
        #########################################################

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        self.controller.stop()
        self.controller.camera_controller.stop()
        
    ####################
    # I. 4. Properties #
    ####################
    
    @property
    def x_size(self) : 
        return self._x_size
    
    @property
    def y_size(self) : 
        return self._y_size

######################
# II. Callback class #
######################

class ScanCheetah3Callback(QtCore.QObject):
    """

    """
    data_sig = QtCore.Signal(np.ndarray,bool)

    def __init__(self, controller, scan_controller):
        self.controller = controller
        self.scan_controller = scan_controller
        self.buffer_size = config('CHEETAH3', 'scan', 'buffer_size')
        super().__init__()
 
    def readout(self,num_frames : int) :
        if 'scan' in self.controller.destination_profiles :
            self.scan_readout(num_frames)
        elif 'live_preview' in self.controller.destination_profiles :
            self.preview_readout(num_frames)
        else :
            raise NotImplementedError('The selected destination profile is not supported.')
                
    def preview_readout(self,num_frames : int) :
        if num_frames == 0 :
            while True :
                try : 
                # CT10. We start a blocking function. It waits until data are avaible.
                    current_image = self.controller.preview()
                    self.data_sig.emit(current_image,True)
                    if self.controller.get_status() == "DA_IDLE" :
                        logger.info("Acquisition finished")
                        break
                except BrokenPipeError :
                    logger.info('Acquistion stopped.')
                    break
        else :
            for i in range(num_frames) : 
                try : 
                # CT10. We start a blocking function. It waits until data are avaible.
                    current_image = self.controller.preview() 
                    self.data_sig.emit(current_image,True)
                    if self.controller.get_status() == "DA_IDLE" : 
                        logger.info("Acquisition finished")
                        break
                except BrokenPipeError : 
                    logger.info('Acquistion stopped.')
                    break
                
    def scan_readout(self,num_frames : int) :
        current_image = np.array(0)
        if num_frames == 0 :
            if 'preview' in self.controller.destination_profiles :
                while True :
                    try :
                    # CT10. We start a blocking function. It waits until data are avaible.
                        data_read = self.controller.client.recv(self.buffer_size)
                        # if len(data_read) == 0 :
                        #     self.revolon.close()
                        #     self.get_request(url=self.serverurl + '/measurement/stop')
                        #     break
                        q = len(data_read) % 8
                        if q:
                            data_read += self.controller.client.recv(8 - q)
                        event_list = np.frombuffer(data_read, dtype=np.uint32)
                        fill_array(self.controller._data,event_list)
                        current_image = self.controller.preview()
                        if self.scan_controller.frame_count % self.controller.cumul_num == 0 :
                            self.data_sig.emit(current_image,True)
                        else :
                            self.data_sig.emit(current_image,False)
                        if self.controller.get_status() == "DA_IDLE" :
                            logger.info("Acquisition finished")
                            break
                        if not self.scan_controller.wait_for_acq() : 
                            logger.info("Scan finished")
                            break
                    except BrokenPipeError :
                        logger.info('Acquistion stopped.')
                        break
            else :
                while True :
                    try :
                    # CT10. We start a blocking function. It waits until data are avaible.
                        data_read = self.controller.client.recv(self.buffer_size)
                        # if len(data_read) == 0 :
                        #     self.revolon.close()
                        #     self.get_request(url=self.serverurl + '/measurement/stop')
                        #     break
                        q = len(data_read) % 8
                        if q:
                            data_read += self.controller.client.recv(8 - q)
                        event_list = np.frombuffer(data_read, dtype=np.uint32)
                        fill_array(self.controller._data,event_list)
                        
                        if self.scan_controller.frame_count % self.controller.cumul_num == 0 :
                            self.data_sig.emit(current_image,True)
                        else :
                            self.data_sig.emit(current_image,False)
                        if self.controller.get_status() == "DA_IDLE" :
                            logger.info("Acquisition finished")
                            break
                        if not self.scan_controller.wait_for_acq() :
                            logger.info("Scan finished")
                            break
                    except BrokenPipeError :
                        logger.info('Acquistion stopped.')
                        break
        else :
            if 'preview' in self.controller.destination_profiles :
                for i in range(num_frames) :
                    try :
                    # CT10. We start a blocking function. It waits until data are avaible.
                        data_read = self.controller.client.recv(self.buffer_size)
                        # if len(data_read) == 0 :
                        #     self.revolon.close()
                        #     self.get_request(url=self.serverurl + '/measurement/stop')
                        #     break
                        q = len(data_read) % 8
                        if q:
                            data_read += self.controller.client.recv(8 - q)
                        event_list = np.frombuffer(data_read, dtype=np.uint32)
                        fill_array(self.controller._data,event_list)
                        current_image = self.controller.preview()
                        if self.scan_controller.frame_count % self.controller.cumul_num == 0 :
                            self.data_sig.emit(current_image,True)
                        else :
                            self.data_sig.emit(current_image,False)
                        if self.controller.get_status() == "DA_IDLE" :
                            logger.info("Acquisition finished")
                            break
                        if not self.scan_controller.wait_for_acq() : 
                            logger.info("Scan finished")
                            break
                    except BrokenPipeError :
                        logger.info('Acquistion stopped.')
                        break
            else : 
                for i in range(num_frames) :
                    try :
                    # CT10. We start a blocking function. It waits until data are avaible.
                        data_read = self.controller.client.recv(self.buffer_size)
                        # if len(data_read) == 0 :
                        #     self.revolon.close()
                        #     self.get_request(url=self.serverurl + '/measurement/stop')
                        #     break
                        q = len(data_read) % 8
                        if q:
                            data_read += self.controller.client.recv(8 - q)
                        event_list = np.frombuffer(data_read, dtype=np.uint32)
                        fill_array(self.controller._data,event_list)
                        
                        if self.scan_controller.frame_count % self.controller.cumul_num == 0 :
                            self.data_sig.emit(current_image,True)
                        else :
                            self.data_sig.emit(current_image,False)
                        if self.controller.get_status() == "DA_IDLE" :
                            logger.info("Acquisition finished")
                            break
                        if not self.scan_controller.wait_for_acq() :
                            logger.info("Scan finished")
                            break
                    except BrokenPipeError :
                        logger.info('Acquistion stopped.')
                        break

###########################            
# III. Local testing code #
###########################

if __name__ == '__main__':
    main(__file__)
