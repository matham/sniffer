''' Devices used in the experiment.
'''


__all__ = ('DeviceStageInterface', 'Server', 'FTDIDevChannel', 'FTDIOdorsBase',
           'FTDIOdorsSim', 'FTDIOdors', 'FTDIADCBase', 'FTDIADCSim', 'FTDIADC')

from functools import partial
import traceback
from math import cos, pi

from moa.compat import bytes_type, unicode_type
from moa.threads import ScheduledEventLoop
from moa.device import Device
from moa.device.digital import ButtonPort, ButtonChannel
from moa.device.analog import NumericPropertyChannel
from moa.device.adc import VirtualADCPort
from moa.tools import ConfigPropertyList, to_bool

from pybarst.core.server import BarstServer
from pybarst.ftdi import FTDIChannel
from pybarst.ftdi.switch import SerializerSettings
from pybarst.ftdi.adc import ADCSettings

from moadevs.ftdi import FTDISerializerDevice, FTDIADCDevice

from kivy.properties import (ConfigParserProperty, BooleanProperty,
                             ListProperty, ObjectProperty, NumericProperty)
from kivy.app import App
from kivy.clock import Clock

from sniffer import device_config_name


class DeviceStageInterface(object):
    ''' Base class for devices used in this project. It provides the callback
    on exception functionality which calls
    :meth:`ExperimentApp.device_exception` when an exception occurs.
    '''

    exception_callback = None
    '''The partial function that has been scheduled to be called by the kivy
    thread when an exception occurs. This function must be unscheduled when
    stopping, in case there are waiting to be called after it already has been
    stopped.
    '''

    def handle_exception(self, exception, event):
        '''The overwritten method called by the devices when they encounter
        an exception.
        '''
        callback = self.exception_callback = partial(
            App.get_running_app().device_exception, exception, event)
        Clock.schedule_once(callback)

    def cancel_exception(self):
        '''Called to cancel the potentially scheduled exception, scheduled with
        :meth:`handle_exception`.
        '''
        Clock.unschedule(self.exception_callback)
        self.exception_callback = None

    def create_device(self):
        '''Called from the kivy thread to create the internal target of this
        device.
        '''
        pass

    def start_channel(self):
        '''Called from secondary thread to initialize the target device. This
        is typically called after :meth:`create_device` is called.
        This method typically opens e.g. the Barst channels on the server and
        sets them to their initial values.
        '''
        pass


class Server(DeviceStageInterface, ScheduledEventLoop, Device):
    '''Server device which creates and opens the Barst server.
    '''

    def create_device(self):
        # create actual server
        self.target = BarstServer(
            barst_path=(self.server_path if self.server_path else None),
            pipe_name=self.server_pipe)

    def start_channel(self):
        server = self.target
        server.open_server()

    server_path = ConfigParserProperty(
        '', 'Server', 'barst_path', device_config_name, val_type=unicode_type)
    '''The full path to the Barst executable. Could be empty if the server
    is already started, on remote computer, or if it's in the typical
    `Program Files` path. If the server is not running, this path is needed
    to launch the server.

    Defaults to `''`.
    '''

    server_pipe = ConfigParserProperty(b'', 'Server', 'pipe',
                                       device_config_name, val_type=bytes_type)
    '''The full path to the pipe name (to be) used by the server. Examples are
    ``\\\\remote_name\pipe\pipe_name``, where ``remote_name`` is the name of
    the remote computer, or a period (`.`) if the server is local, and
    ``pipe_name`` is the name of the pipe used to create the server.

    Defaults to `''`.
    '''


class FTDIDevChannel(DeviceStageInterface, ScheduledEventLoop, Device):
    '''FTDI channel device. This controls internally both the odor
    and ftdi pin devices.
    '''

    def create_device(self, dev_settings, server):
        '''See :meth:`DeviceStageInterface.create_device`.

        `dev_settings` is the list of device setting to be passed to the
        Barst ftdi channel. `server` is the Barst server.
        '''
        self.target = FTDIChannel(
            channels=dev_settings, server=server,
            desc=self.ftdi_desc[self.dev_idx],
            serial=self.ftdi_serial[self.dev_idx])

    def start_channel(self):
        self.target.open_channel(alloc=True)
        self.target.close_channel_server()
        return self.target.open_channel(alloc=True)

    ftdi_serial = ConfigPropertyList(b'', 'FTDI_chan', 'serial_number',
                                     device_config_name, val_type=bytes_type)
    '''The serial number if the FTDI hardware board. Can be empty.
    '''

    ftdi_desc = ConfigPropertyList(b'', 'FTDI_chan', 'description_id',
                                   device_config_name, val_type=bytes_type)
    '''The description of the FTDI hardware board.

    :attr:`ftdi_serial` or :attr:`ftdi_desc` are used to locate the correct
    board to open. An example is `'Alder Board'` for the Alder board.
    '''

    dev_idx = NumericProperty(0)


class FTDIOdorsBase(object):
    '''Base class for the FTDI odor devices.
    '''

    p0 = BooleanProperty(False, allownone=True)
    '''Controls valve 0. '''

    p1 = BooleanProperty(False, allownone=True)
    '''Controls valve 1. '''

    p2 = BooleanProperty(False, allownone=True)
    '''Controls valve 2. '''

    p3 = BooleanProperty(False, allownone=True)
    '''Controls valve 3. '''

    p4 = BooleanProperty(False, allownone=True)
    '''Controls valve 4. '''

    p5 = BooleanProperty(False, allownone=True)
    '''Controls valve 5. '''

    p6 = BooleanProperty(False, allownone=True)
    '''Controls valve 6. '''

    p7 = BooleanProperty(False, allownone=True)
    '''Controls valve 7. '''

    p8 = BooleanProperty(False, allownone=True)
    '''Controls valve 8. '''

    p9 = BooleanProperty(False, allownone=True)
    '''Controls valve 9. '''

    p10 = BooleanProperty(False, allownone=True)
    '''Controls valve 10. '''

    p11 = BooleanProperty(False, allownone=True)
    '''Controls valve 11. '''

    p12 = BooleanProperty(False, allownone=True)
    '''Controls valve 12. '''

    p13 = BooleanProperty(False, allownone=True)
    '''Controls valve 13. '''

    p14 = BooleanProperty(False, allownone=True)
    '''Controls valve 14. '''

    p15 = BooleanProperty(False, allownone=True)
    '''Controls valve 15. '''

    num_boards = ConfigPropertyList(2, 'FTDI_odor', 'num_boards',
                                    device_config_name, val_type=int)
    '''The number of valve boards connected to the FTDI device.

    Each board controls 8 valves. Defaults to 2.
    '''

    dev_idx = NumericProperty(0)


class FTDIOdorsSim(FTDIOdorsBase, ButtonPort):
    '''Device used when simulating the odor devices.
    '''
    pass


class FTDIOdors(FTDIOdorsBase, DeviceStageInterface, FTDISerializerDevice):
    '''Device used when using the barst ftdi odor devices.
    '''

    def __init__(self, **kwargs):
        mapping = {'p{}'.format(i): i for i in range(8 * self.num_boards[kwargs['dev_idx']])}
        super(FTDIOdors, self).__init__(mapping=mapping, **kwargs)

    def get_settings(self):
        '''Returns the :class:`SerializerSettings` instance used to create the
        Barst FTDI odor device.
        '''
        i = self.dev_idx
        return SerializerSettings(
            clock_bit=self.clock_bit[i], data_bit=self.data_bit[i],
            latch_bit=self.latch_bit[i], num_boards=self.num_boards[i],
            output=True)

    def start_channel(self):
        odors = self.target
        odors.open_channel()
        odors.set_state(True)
        odors.write(set_low=range(8 * self.num_boards[self.dev_idx]))

    ftdi_dev = ConfigPropertyList(0, 'FTDI_odor', 'ftdi_dev',
                                  device_config_name, val_type=int)

    clock_bit = ConfigPropertyList(0, 'FTDI_odor', 'clock_bit',
                                   device_config_name, val_type=int)
    '''The pin on the FTDI board to which the valve's clock bit is connected.

    Defaults to zero.
    '''

    data_bit = ConfigPropertyList(0, 'FTDI_odor', 'data_bit',
                                  device_config_name, val_type=int)
    '''The pin on the FTDI board to which the valve's data bit is connected.

    Defaults to zero.
    '''

    latch_bit = ConfigPropertyList(0, 'FTDI_odor', 'latch_bit',
                                   device_config_name, val_type=int)
    '''The pin on the FTDI board to which the valve's latch bit is connected.

    Defaults to zero.
    '''


class FTDIADCBase(object):

    def __init__(self, **kwargs):
        super(FTDIADCBase, self).__init__(**kwargs)
        i = self.dev_idx
        self.active_channels = [self.chan1_active[i], self.chan2_active[i]]
        self.bit_depth = self.data_width
        self.frequency = self.sampling_rate
        self.num_channels = 2

    dev_idx = NumericProperty(0)

    sampling_rate = ConfigParserProperty(1000, 'FTDI_ADC', 'sampling_rate',
                                         device_config_name, val_type=float)

    data_width = ConfigParserProperty(24, 'FTDI_ADC', 'data_width',
                                      device_config_name, val_type=int)

    chan1_active = ConfigPropertyList(True, 'FTDI_ADC', 'chan1_active',
                                      device_config_name, val_type=to_bool)

    chan2_active = ConfigPropertyList(False, 'FTDI_ADC', 'chan2_active',
                                      device_config_name, val_type=to_bool)

    transfer_size = ConfigParserProperty(1000, 'FTDI_ADC', 'transfer_size',
                                         device_config_name, val_type=int)


class FTDIADCSim(FTDIADCBase, VirtualADCPort):

    def __init__(self, **kwargs):
        super(FTDIADCSim, self).__init__(**kwargs)
        self.data_size = self.transfer_size

        def next_point(i):
            return .2 * cos(2 * pi * i / float(self.frequency))
        self.data_func = next_point


class FTDIADC(FTDIADCBase, DeviceStageInterface, FTDIADCDevice):

    def get_settings(self):
        i = self.dev_idx
        return ADCSettings(
            clock_bit=self.clock_bit[i], lowest_bit=self.lowest_bit[i],
            num_bits=self.num_bits[i], sampling_rate=self.sampling_rate,
            chan1=self.chan1_active[i], chan2=self.chan2_active[i],
            transfer_size=self.transfer_size, data_width=self.data_width)

    def start_channel(self):
        adc = self.target
        adc.open_channel()
#         adc.set_state(True)
#         adc.read()
#         adc.set_state(False)
#         try:
#             adc.read()
#         except:
#             pass

    ftdi_dev = ConfigPropertyList(0, 'FTDI_ADC', 'ftdi_dev',
                                  device_config_name, val_type=int)

    clock_bit = ConfigPropertyList(0, 'FTDI_ADC', 'clock_bit',
                                   device_config_name, val_type=int)

    lowest_bit = ConfigPropertyList(0, 'FTDI_ADC', 'lowest_bit',
                                    device_config_name, val_type=int)

    num_bits = ConfigPropertyList(0, 'FTDI_ADC', 'num_bits',
                                  device_config_name, val_type=int)
