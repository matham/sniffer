# -*- coding: utf-8 -*-
'''The stages of the experiment.
'''


from functools import partial
from time import strftime, sleep
from re import match, compile
import csv
from os.path import exists, splitext
import tables as tb
from random import randint, shuffle
from fractions import Fraction

from moa.stage import MoaStage
from moa.stage.delay import Delay
from moa.utils import (
    ConfigPropertyList, ConfigPropertyDict, to_string_list, to_bool)
from moa.compat import unicode_type
from cplcom.moa.data_logger import DataLogger
from moa.base import named_moas as moas
from moa.device.digital import ButtonChannel

from kivy.app import App
from kivy.properties import (
    ObjectProperty, ListProperty, ConfigParserProperty, NumericProperty,
    BooleanProperty, StringProperty, OptionProperty)
from kivy import resources
from kivy.lang import Factory
from kivy.uix.button import Button

from cplcom.device.barst_server import Server
from cplcom.device.ftdi import FTDIDevChannel, FTDIADCDevice
from cplcom.moa.stages import InitStage
from cplcom.device.ffplayer import FFPyPlayerDevice, FFPyWriterDevice
from cplcom.graphics import FFImage

from sniffer.devices import FTDIOdors,\
    FTDIOdorsSim, FTDIADCSim
from sniffer import exp_config_name, device_config_name
from sniffer.graphics import BoxDisplay


odor_name_pat = compile('([0-9])\.p([0-9])+')
to_list_pat = compile('(?:, *)?\\n?')


class OdorTuple(tuple):
    def __str__(self):
        return '{}.p{}'.format(self[0], self[1])

    def __repr__(self):
        return self.__str__()


def verify_valve_name(val):
    if isinstance(val, OdorTuple):
        return val
    m = match(odor_name_pat, val)
    if m is None:
        raise Exception('{} does not match the valve name pattern'.format(val))
    return OdorTuple((int(m.group(1)), int(m.group(2))))


def verify_out_fmt(fmt):
    if fmt not in ('rgb24', 'gray', 'yuv420p'):
        raise Exception('{} is not a valid output format'.format(fmt))
    return fmt


def verify_fraction(val):
    return Fraction(*val.split('/'))


class InitBarstStage(InitStage):

    server = ObjectProperty(None, allownone=True)

    ftdi_chans = ObjectProperty(None, allownone=True)

    odor_devs = ObjectProperty(None, allownone=True, rebind=True)

    adc_devs = ObjectProperty(None, allownone=True, rebind=True)

    num_ftdi_chans = ConfigParserProperty(
        1, 'FTDI_chan', 'num_ftdi_chans', device_config_name, val_type=int)

    num_adc_chans = ConfigParserProperty(
        1, 'FTDI_ADC', 'num_adc_chans', device_config_name, val_type=int)

    adc_ftdi_dev = ConfigPropertyList(
        0, 'FTDI_ADC', 'adc_ftdi_dev', device_config_name, val_type=int)

    num_odor_chans = ConfigParserProperty(
        1, 'FTDI_odor', 'num_odor_chans', device_config_name, val_type=int)

    odor_ftdi_dev = ConfigPropertyList(
        0, 'FTDI_odor', 'odor_ftdi_dev', device_config_name, val_type=int)

    num_boards = ConfigPropertyList(
        1, 'FTDI_odor', 'num_boards', device_config_name, val_type=int)

    odor_clock_size = ConfigParserProperty(
        10, 'FTDI_odor', 'odor_clock_size', device_config_name, val_type=int)

    players = ListProperty([])

    src_names = ConfigPropertyList(
        '', 'Video', 'src_names', exp_config_name, val_type=unicode_type,
        autofill=False)

    src_names_sim = ConfigPropertyList(
        'Wildlife.mp4', 'Video', 'src_names_sim', exp_config_name,
        val_type=unicode_type, autofill=False)

    img_fmt = ConfigPropertyList(
        'yuv420p', 'Video', 'img_fmt', exp_config_name,
        val_type=verify_out_fmt)

    video_rate = ConfigPropertyList(
        '30.', 'Video', 'video_rate', exp_config_name,
        val_type=verify_fraction)

    def start_init(self, sim=True):
        odor_btns = App.get_running_app().root.ids.odors

        if sim:
            odorcls = FTDIOdorsSim
            adccls = FTDIADCSim
        else:
            odorcls = FTDIOdors
            adccls = FTDIADCDevice

        dev_cls = [Factory.get('ToggleDevice'), Factory.get('DarkDevice')]
        odor_btns.clear_widgets()
        num_boards = self.num_boards
        odors = []
        for i in range(self.num_odor_chans):
            btns = [dev_cls[j % 2](text='{}.p{}'.format(i, j))
                    for j in range(num_boards[i] * 8)]
            for btn in btns:
                odor_btns.add_widget(btn)
            odors.append(odorcls(
                name='odors{}'.format(i), odor_btns=btns, N=num_boards[i] * 8,
                idx=i))
        self.odor_devs = odors

        adcs = [adccls(name='adc{}'.format(i), idx=i)
                for i in range(self.num_adc_chans)]
        self.adc_devs = adcs

        players = [None, ] * moas.verify.num_boxes
        fmts = self.img_fmt
        rate = self.video_rate
        for i, name in enumerate(
                self.src_names_sim if sim else self.src_names):
            if i >= len(players):
                break
            if not name:
                continue
            players[i] = FFPyPlayerDevice(
                filename=name, output_img_fmt=fmts[i],
                input_rate=float(rate[i]))
        self.players = players

        if not sim:
            for o in self.odor_devs:
                o.clock_size = self.odor_clock_size
            server = self.server = Server(restart=False)
            server.create_device()

            ftdis = [FTDIDevChannel(idx=i) for i in range(self.num_ftdi_chans)]
            self.ftdi_chans = ftdis
            adc_chans = self.adc_ftdi_dev
            odor_chans = self.odor_ftdi_dev
            for i, ftdev in enumerate(ftdis):
                ftdev.create_device(
                    [o for (j, o) in enumerate(odors) if odor_chans[j] == i] +
                    [a for (j, a) in enumerate(adcs) if adc_chans[j] == i],
                    server)
        return super(InitBarstStage, self).start_init(
            sim=sim, devs=self.adc_devs + self.odor_devs)

    def init_threaded(self):
        return super(InitBarstStage, self).init_threaded(
            devs=[self.server] + self.ftdi_chans + self.adc_devs +
            self.odor_devs)

    def finish_init(self, *largs):
        return super(InitBarstStage, self).finish_init(
            self.adc_devs + self.odor_devs, *largs)

    def stop_devices(self):
        boxes = App.get_running_app().root.ids.boxes
        if boxes is not None:
            for box in boxes.children:
                box.ids.acquire.state = 'normal'
        for player in self.players:
            if player is not None:
                player.set_state(False)
                player.deactivate(self)

        boxes = moas.boxes
        if boxes is not None:
            for box in boxes.stages:
                box.deinitialize_box()

        return super(InitBarstStage, self).stop_devices(
            self.odor_devs + self.adc_devs + self.ftdi_chans)


class VerifyConfigStage(MoaStage):
    '''Stage that is run before the first block of each animal.

    The stage verifies that all the experimental parameters are correct and
    computes all the values, e.g. odors needed for the trials.

    If the values are incorrect, it calls
    :meth:`ExperimentApp.device_exception` with the exception.
    '''

    _cum_boards = []

    def __init__(self, **kw):
        super(VerifyConfigStage, self).__init__(**kw)
        self.exclude_attrs = ['finished']

    def step_stage(self, *largs, **kwargs):
        if not super(VerifyConfigStage, self).step_stage(*largs, **kwargs):
            return False

        app = App.get_running_app()
        try:
            self._cum_boards = cum_boards = []
            brds = moas.barst.num_boards
            last = 0
            for n in range(moas.barst.num_odor_chans):
                cum_boards.append(last)
                last += brds[n]
            N = last * 8

            self.read_odors()
            self.parse_odors()

            btns = App.get_running_app().root.ids.odors.children
            for board, idx in self.NO_valves:
                valve = btns[N - 1 - (cum_boards[board] * 8 + idx)]
                valve.background_down = 'dark-blue-led-on-th.png'
                valve.background_normal = 'dark-blue-led-off-th.png'
            for board, idx in self.rand_valves:
                valve = btns[N - 1 - (cum_boards[board] * 8 + idx)]
                valve.background_down = 'brown-led-on-th.png'
                valve.background_normal = 'brown-led-off-th.png'

            boxes = moas.boxes
            gui_boxes = App.get_running_app().root.ids.boxes
            video_root = App.get_running_app().root.ids.video
            gui_boxes.clear_widgets()
            video_root.clear_widgets()
            num_boxes = self.num_boxes
            adcs = moas.barst.adc_devs
            stages = [BoxStage(box=i, moas=moas.new_moas())
                      for i in range(num_boxes)]
            displays = [BoxDisplay(moas=stage.moas) for stage in stages]
            for i, (stage, display, player) in enumerate(
                    zip(stages, displays, moas.barst.players)):
                stage.init_display(display, player)
                dev, chan = self.adc_dev[i], self.adc_dev_chan[i]
                if not adcs[dev].active_channels[chan]:
                    raise Exception('ADC device {}, inactive channel {} used'.
                                    format(dev, chan))
                display.init_adc(adcs[dev], chan)

                boxes.add_stage(stage)
                gui_boxes.add_widget(display)
        except Exception as e:
            app.device_exception(e)
            return
        self.step_stage()
        return True

    def read_odors(self):
        N = moas.barst.num_odor_chans
        brds = moas.barst.num_boards
        cum_boards = self._cum_boards
        odor_names = [
            'p{}'.format(j) for n in range(N) for j in range(8 * brds[n])]

        # now read the odor list
        odor_path = resources.resource_find(self.odor_path)
        with open(odor_path, 'rb') as fh:
            for row in csv.reader(fh):
                row = [elem.strip() for elem in row]
                if not row:
                    continue
                valve, name = row[:2]
                board, idx = verify_valve_name(valve)
                if board >= N:
                    raise Exception('Board number of {} is too large'.
                                    format(valve))
                if idx >= 8 * brds[board]:
                    raise Exception('Index of {} is too large'.format(valve))
                odor_names[cum_boards[board] * 8 + idx] = name
        self.odor_names = odor_names

    def parse_odors(self):
        num_boxes = self.num_boxes
        n_odors = moas.barst.num_odor_chans
        n_adcs = moas.barst.num_adc_chans
        num_boards = moas.barst.num_boards
        if num_boxes <= 0:
            raise Exception('Number of boxes is not positive')
        if any([i > n_adcs for i in self.adc_dev]):
            raise Exception('ADC dev index out of range')
        if any([i >= 2 for i in self.adc_dev_chan]):
            raise Exception('ADC dev channel larger or equal to 2')

        if any([not len(box_odors)
                for box_odors in self.odor_protocols.values()]):
            raise Exception('No odor specified for every box')
        for odors in self.odor_protocols.values():
            for board, idx in odors:
                if board >= n_odors:
                    raise Exception('Valve board number {} is too large'.
                                    format(board))
                if idx >= 8 * num_boards[board]:
                    raise Exception('Valve index {} too large'.format(idx))
        for board, idx in self.NO_valves:
            if board >= n_odors:
                raise Exception('Board number {} is too large for NO valve'.
                                format(board))
            if idx >= 8 * num_boards[board]:
                raise Exception('Index {} too large for NO valve'.format(idx))
        for board, idx in self.rand_valves:
            if board >= n_odors:
                raise Exception('Board number {} is too large for rand valve'.
                                format(board))
            if idx >= 8 * num_boards[board]:
                raise Exception('Index {} too large for random valve'.
                                format(idx))

    num_boxes = ConfigParserProperty(
        1, 'Experiment', 'num_boxes', device_config_name, val_type=int)

    hab_dur = ConfigParserProperty(1, 'Experiment', 'hab_dur',
                                   exp_config_name, val_type=float)

    num_trials = ConfigParserProperty(1, 'Experiment', 'num_trials',
                                      exp_config_name, val_type=int)

    trial_dur = ConfigParserProperty(1, 'Experiment', 'trial_dur',
                                     exp_config_name, val_type=float)

    iti_min = ConfigParserProperty(1, 'Experiment', 'iti_min',
                                   exp_config_name, val_type=float)

    iti_max = ConfigParserProperty(1, 'Experiment', 'iti_max',
                                   exp_config_name, val_type=float)

    post_dur = ConfigParserProperty(1, 'Experiment', 'post_dur',
                                    exp_config_name, val_type=float)

    odor_protocols = ConfigPropertyDict(
        'prot1: 0.p1\nprot2: 0.p1', 'Odor', 'odor_protocols',
        exp_config_name, val_type=partial(to_string_list, verify_valve_name),
        key_type=str)

    NO_valves = ConfigPropertyList('0.p0', 'Odor', 'NO_valves',
                                   exp_config_name, val_type=verify_valve_name)

    rand_valves = ConfigPropertyList(
        '0.p0', 'Odor', 'rand_valves', exp_config_name,
        val_type=verify_valve_name)

    odor_path = ConfigParserProperty(
        u'odor_list.txt', 'Odor', 'Odor_list_path', exp_config_name,
        val_type=unicode_type)

    adc_dev = ConfigPropertyList(0, 'ADC', 'adc_dev', exp_config_name,
                                 val_type=int)

    adc_dev_chan = ConfigPropertyList(0, 'ADC', 'adc_dev_chan',
                                      exp_config_name, val_type=int)

    odor_names = ObjectProperty(None)

    def on_odor_names(self, *largs):
        for name, btn in zip(
            self.odor_names, reversed(App.get_running_app().
                                      root.ids.odors.children)):
            btn.text = name


class BoxStage(MoaStage):
    '''In this stage, each loop runs another animal and its blocks and trials.
    '''

    next_animal_dev = ObjectProperty(None)

    tb_file = None

    box = NumericProperty(0)

    display = ObjectProperty(None)

    animal_id = StringProperty('')

    odors = ObjectProperty(None, allownone=True)

    last_adc_data = None

    bound_callbacks = []

    log_filename = ConfigParserProperty(
        '%m-%d-%y_{animal}.h5', 'Experiment', 'log_filename', exp_config_name,
        val_type=unicode_type)

    record_video = ConfigPropertyList(
        True, 'Video', 'record_video', exp_config_name, val_type=to_bool)

    play_video = ConfigPropertyList(
        True, 'Video', 'play_video', exp_config_name, val_type=to_bool)

    video_display = None

    video_writer = None

    video_player = None

    base_pts = None

    def __init__(self, **kw):
        super(BoxStage, self).__init__(**kw)
        self.exclude_attrs = ['finished']

    def init_display(self, display, player):
        self.display = display
        btn = display.ids.start_btn
        btn.state = 'normal'
        self.next_animal_dev = ButtonChannel(
            button=btn.__self__, name='start_btn', moas=self.moas)
        self.next_animal_dev.activate(self)

        timer = self.display.ids.timer
        timer.add_slice('Pre', duration=moas.verify.hab_dur)
        timer.add_slice(
            'Trial', duration=moas.verify.trial_dur,
            text='Trial ({}/{})'.format(1, moas.verify.num_trials))
        timer.add_slice('ITI', duration=moas.verify.iti_max)
        timer.add_slice('Post', duration=moas.verify.post_dur)
        timer.add_slice('Done',)
        timer.smear_slices()
        timer.set_active_slice('Pre')

        play = self.play_video[self.box]
        self.video_player = player
        video_root = App.get_running_app().root.ids.video
        self.video_display = video_display = FFImage()
        video_root.add_widget(video_display)
        if player is None or not play:
            return
        player.callback = self.video_callback
        player.button = Button()
        player.activate(self)
        player.set_state(True)

    def video_callback(self, frame, pts):
        writer = self.video_writer
        if writer is not None:
            base_pts = self.base_pts
            if base_pts is None:
                base_pts = self.base_pts = pts
            writer.add_frame(frame, pts - base_pts)
        display = self.video_display
        if display is not None:
            display.display(frame)

    def set_adc_state(self, activate=True, source=None):
        adc = moas.barst.adc_devs[moas.verify.adc_dev[self.box]]
        if activate:
            adc.activate(self if source is None else source)
        else:
            adc.deactivate(self if source is None else source)

    def initialize_box(self):
        self.set_adc_state(True)
        odors = moas.verify.odor_protocols[self.display.ids.protocols.text]
        timer = self.display.ids.timer
        self.odors = odors
        timer.set_active_slice('Pre')

        player = self.video_player
        box = self.box
        record = player and self.play_video[box] and self.record_video[box]

        fname = strftime(self.log_filename.format(
            **{'animal': self.display.ids.animal_name.text}))
        video_fname = splitext(fname)[0] + '.avi'
        while exists(fname) or (record and exists(video_fname)):
            n, ext = splitext(fname)
            m = match('(.+)_r([0-9]+)', n)
            if m is not None:
                name, count = m.groups()
                count = int(count) + 1
            else:
                name = n
                count = 2
            fname = '{}_r{}{}'.format(name, count, ext)
            video_fname = splitext(fname)[0] + '.avi'

        if record:
            while player.size is None or player.rate is None:
                sleep(0.005)
            self.video_writer = self.video_writer = FFPyWriterDevice(
                filename=video_fname, size=player.size,
                rate=moas.barst.video_rate[box], ifmt=player.output_img_fmt)

        self.tb_file = DataLogger(filename=fname)
        f = self.tb_file.tb_file
        raw = f.root.raw_data
        odors = f.create_group(
            raw, 'odors', 'States of the odor valves (powered/unpowered)')

        bound = self.bound_callbacks = []
        box = self.box
        box_odors = list(set(self.odors))
        if box_odors:
            odor_devs = moas.barst.odor_devs
            names = moas.verify.odor_names
            for board, idx in box_odors:
                dev = odor_devs[board]
                name = '{}_{}.p{}'.format(names[board][idx], board, idx)
                attr = 'p{}'.format(idx)

                group = f.create_group(odors, name, 'State of {}'.format(name))
                s = f.create_earray(
                    group, 'state', tb.BoolAtom(), (0, ), 'Valve state')
                ts = f.create_earray(group, 'ts', tb.Float64Atom(), (0, ),
                                     'Timestamps of state values.')
                bound.append(
                    (dev, attr,
                     dev.fast_bind(attr, self.add_odor_point, dev, attr, s,
                                   ts)))

        pressure = f.create_group(
            raw, 'pressure', 'Pressure data of the chamber')
        data = f.create_earray(
            pressure, 'data', tb.UInt32Atom(), (0, ), 'The raw pressure data.')
        ts = f.create_earray(
            pressure, 'ts', tb.Float64Atom(), (0, ),
            'The timestamps of the data.')
        ts_idx = f.create_earray(
            pressure, 'ts_idx', tb.UInt32Atom(), (0, ),
            'The indices in data of the timestamps.')
        adc_dev = moas.barst.adc_devs[moas.verify.adc_dev[box]]
        adc_chan = moas.verify.adc_dev_chan[box]
        attrs = pressure._v_attrs
        for k in ['bit_depth', 'scale', 'offset', 'frequency']:
            attrs[k] = getattr(adc_dev, k)
        bound.append(
            (adc_dev, 'data',
             adc_dev.fast_bind('data', self.add_adc_points, adc_dev, adc_chan,
                               data, ts, ts_idx)))

    def add_odor_point(self, dev, attr, tb_state, tb_ts, *largs):
        tb_state.append((getattr(dev, attr), ))
        tb_ts.append((dev.timestamp, ))

    def add_adc_points(self, dev, chan, data, ts, ts_idx, *l):
        new_data = dev.raw_data[chan]
        if new_data is self.last_adc_data or new_data is None:
            return
        self.last_adc_data = new_data
        ts.append((dev.timestamp, ))
        ts_idx.append((dev.ts_idx[chan] + data.nrows, ))
        data.append(list(new_data))

    def do_odor(self, trial, start=True):
        (d1, i1), (d2, i2) = moas.verify.NO_valves[self.box], self.odors[trial]
        devs = moas.barst.odor_devs

        state = 'high' if start else 'low'
        if d1 == d2:
            devs[d1].set_state(**{state: ['p{}'.format(i1), 'p{}'.format(i2)]})
        else:
            devs[d1].set_state(**{state: ['p{}'.format(i1)]})
            devs[d2].set_state(**{state: ['p{}'.format(i2)]})

    def deinitialize_box(self):
        self.set_adc_state(False)
        for dev, name, uid in self.bound_callbacks:
            dev.unbind_uid(name, uid)
        self.bound_callbacks = []

        f = self.tb_file
        if f is not None:
            f.tb_file.close()
            self.tb_file = None

        writer = self.video_writer
        self.video_writer = None
        if writer is not None:
            writer.add_frame()

        self.display.ids.timer.set_active_slice('Done')


class RandValves(Delay):

    def __init__(self, **kwargs):
        super(RandValves, self).__init__(**kwargs)
        self.high = []
        self.delay_type = 'random'
        self.max = self.valve_rand_max
        self.min = self.valve_rand_min

    def step_stage(self, *largs, **kwargs):
        if not super(RandValves, self).step_stage(*largs, **kwargs):
            return False
        if self.low is None:
            self.low = list(set(moas.verify.rand_valves))

        h = self.high
        l = self.low
        shuffle(h)
        shuffle(l)
        hnew = l[:randint(0, len(l))]
        lnew = h[:randint(0, len(h))]
        self.low = lnew + l[len(hnew):]
        self.high = hnew + h[len(lnew):]

        devs = moas.barst.odor_devs
        h = [[] for _ in devs]
        l = [[] for _ in devs]
        for board, idx in self.low:
            l[board].append('p{}'.format(idx))
        for board, idx in self.high:
            h[board].append('p{}'.format(idx))
        for i, dev in enumerate(devs):
            if l[i] or h[i]:
                dev.set_state(low=l[i], high=h[i])
        return True

    high = []
    low = None

    valve_rand_min = ConfigParserProperty(
        .4, 'Experiment', 'valve_rand_min', exp_config_name, val_type=float)

    valve_rand_max = ConfigParserProperty(
        .8, 'Experiment', 'valve_rand_max', exp_config_name, val_type=float)
