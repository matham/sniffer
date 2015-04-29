''' Devices used in the experiment.
'''


__all__ = ('DeviceStageInterface', 'Server', 'FTDIDevChannel', 'FTDIOdorsBase',
           'FTDIOdorsSim', 'FTDIOdors', 'FTDIADCBase', 'FTDIADCSim', 'FTDIADC')

from math import cos, pi

from moa.device.digital import ButtonPort
from moa.device.adc import VirtualADCPort
from moa.utils import ConfigPropertyList, to_bool

from kivy.properties import NumericProperty

from cplcom import device_config_name
from cplcom.device.ftdi import FTDISerializerDevice


class FTDIOdorsBase(object):
    '''Base class for the FTDI odor devices.
    '''

    def __init__(self, odor_btns=None, N=8, **kwargs):
        Nb = len(odor_btns)
        for i in range(N):
            self.create_property('p{}'.format(i), value=False, allownone=True)
        attr_map = {
            'p{}'.format(i): odor_btns[i].__self__ for i in range(Nb)}
        super(FTDIOdorsBase, self).__init__(
            attr_map=attr_map, direction='o', **kwargs)


class FTDIOdorsSim(FTDIOdorsBase, ButtonPort):
    '''Device used when simulating the odor devices.
    '''
    pass


class FTDIOdors(FTDIOdorsBase, FTDISerializerDevice):
    '''Device used when using the barst ftdi odor devices.
    '''

    def __init__(self, N=8, **kwargs):
        dev_map = {'p{}'.format(i): i for i in range(N)}
        super(FTDIOdors, self).__init__(dev_map=dev_map, N=N, **kwargs)


class FTDIADCSim(VirtualADCPort):

    def __init__(self, **kwargs):
        super(FTDIADCSim, self).__init__(**kwargs)
        i = self.idx
        self.bit_depth = self.data_width[i]
        self.frequency = self.sampling_rate[i]
        self.num_channels = 2
        self.active_channels = [self.chan1_active[i], self.chan2_active[i]]
        self.raw_data = [None, None]
        self.data = [None, None]
        self.ts_idx = [0, 0]

        self.data_size = self.transfer_size[i]
        self.scale = 10
        self.offset = 5

        def next_point(i, *largs):
            return .2 * cos(2 * pi * 10 * i / float(self.frequency))
        self.data_func = next_point

    sampling_rate = ConfigPropertyList(
        1000, 'FTDI_ADC', 'sampling_rate', device_config_name, val_type=float)

    data_width = ConfigPropertyList(
        24, 'FTDI_ADC', 'data_width', device_config_name, val_type=int)

    chan1_active = ConfigPropertyList(
        True, 'FTDI_ADC', 'chan1_active', device_config_name, val_type=to_bool)

    chan2_active = ConfigPropertyList(
        False, 'FTDI_ADC', 'chan2_active', device_config_name,
        val_type=to_bool)

    transfer_size = ConfigPropertyList(
        1000, 'FTDI_ADC', 'transfer_size', device_config_name, val_type=int)

    idx = NumericProperty(0)
