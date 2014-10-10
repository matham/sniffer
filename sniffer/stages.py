# -*- coding: utf-8 -*-
'''The stages of the experiment.
'''


from functools import partial
import traceback
from time import strftime
from re import match, compile, split
import csv

from moa.stage import MoaStage
from moa.threads import ScheduledEventLoop
from moa.tools import ConfigPropertyList, ConfigPropertyDict, StringList
from moa.compat import unicode_type

from kivy.app import App
from kivy.properties import (
    ObjectProperty, ListProperty, ConfigParserProperty, NumericProperty,
    BooleanProperty, StringProperty, OptionProperty)
from kivy.clock import Clock
from kivy.factory import Factory
from kivy import resources

from sniffer.devices import Server, FTDIDevChannel, FTDIOdors,\
    FTDIOdorsSim, FTDIADCSim, FTDIADC
from sniffer import exp_config_name, device_config_name
from sniffer.graphics import OdorContainer, BoxDisplay


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


def parse_odor_list(val):
    try:
        if isinstance(val, list):
            return StringList([OdorTuple(v) for v in val])
        return [verify_valve_name(v) for v in
                split(to_list_pat, val.strip(' []()'))]
    except:
        raise Exception('Cannot parse odor list "{}"'.format(val))


class RootStage(MoaStage):
    '''The root stage of the experiment. This stage contains all the other
    experiment stages.
    '''

    def on_finished(self, *largs, **kwargs):
        '''Executed after the root stage and all sub-stages finished. It stops
        all the devices.
        '''
        if self.finished:
            def clear_app(*l):
                app = App.get_running_app()
                app.app_state = 'clear'
            barst = self.barst
            barst.clear_events()
            barst.start_thread()
            barst.request_callback('stop_devices', clear_app)
            for child in self.ids.boxes.stages:
                fd = getattr(child, '_fd', None)
                if fd is not None:
                    fd.close()


class InitBarstStage(MoaStage, ScheduledEventLoop):
    '''The stage that creates and initializes all the Barst devices (or
    simulation devices if :attr:`ExperimentApp.simulate`).
    '''

    # if a device is currently being initialized by the secondary thread.
    _finished_init = False
    # if while a device is initialized, stage should stop when finished.
    _should_stop = None

    simulate = BooleanProperty(False)
    '''If True, virtual devices should be used for the experiment. Otherwise
    actual Barst devices will be used. This is set to the same value as
    :attr:`ExperimentApp.simulate`.
    '''

    server = ObjectProperty(None, allownone=True)
    '''The :class:`Server` instance. When :attr:`simulate`, this is None. '''

    ftdi_chans = ObjectProperty(None, allownone=True)
    '''The :class:`FTDIDevChannel` instance. When :attr:`simulate`, this is
    None.
    '''

    odor_devs = ObjectProperty(None, allownone=True, rebind=True)
    '''The :class:`FTDIOdors` instance, or :class:`FTDIOdorsSim` instance when
    :attr:`simulate`.
    '''

    adc_devs = ObjectProperty(None, allownone=True, rebind=True)
    '''The :class:`FTDIOdors` instance, or :class:`FTDIOdorsSim` instance when
    :attr:`simulate`.
    '''

    num_ftdi_chans = ConfigParserProperty(
        1, 'FTDI_chan', 'num_ftdi_chans', device_config_name, val_type=int)

    num_adc_chans = ConfigParserProperty(
        1, 'FTDI_ADC', 'num_adc_chans', device_config_name, val_type=int)

    num_odor_chans = ConfigParserProperty(
        1, 'FTDI_odor', 'num_odor_chans', device_config_name, val_type=int)

    num_boxes = ConfigParserProperty(
        1, 'Experiment', 'num_boxes', device_config_name, val_type=int)

    exception_callback = None
    '''The partial function that has been scheduled to be called by the kivy
    thread when an exception occurs. This function must be unscheduled when
    stopping, in case there are waiting to be called after it already has been
    stopped.
    '''

    def __init__(self, **kw):
        super(InitBarstStage, self).__init__(**kw)
        self.stop_thread()
        self.simulate = App.get_running_app().simulate
        Clock.schedule_once(lambda *x: self.start_thread())

    def recover_state(self, state):
        # When recovering stage, even if finished before, always redo it
        # because we need to init the Barst devices, so skip `finished`.
        state.pop('finished', None)
        return super(InitBarstStage, self).recover_state(state)

    def clear(self, *largs, **kwargs):
        self._finished_init = False
        self._should_stop = None
        return super(InitBarstStage, self).clear(*largs, **kwargs)

    def unpause(self, *largs, **kwargs):
        # if simulating, we cannot be in pause state
        if super(InitBarstStage, self).unpause(*largs, **kwargs):
            if self._finished_init:
                # when unpausing, just continue where we were
                self.finish_start_devices()
            return True
        return False

    def stop(self, *largs, **kwargs):
        if self.started and not self._finished_init and not self.finished:
            self._should_stop = largs, kwargs
            return False
        return super(InitBarstStage, self).stop(*largs, **kwargs)

    def step_stage(self, *largs, **kwargs):
        if not super(InitBarstStage, self).step_stage(*largs, **kwargs):
            return False

        # if we simulate, create the sim devices, otherwise the barst devices
        try:
            s = App.get_running_app().simulation_devices
            s.clear_widgets()
            dummy = FTDIOdorsSim()

            for i, n in enumerate([dummy.num_boards[i]
                                   for i in range(self.num_odor_chans)]):
                s.add_widget(OdorContainer(dev_idx=i, num_boards=n))

            if self.simulate:
                self.create_sim_devices()
                self.step_stage()
                return True
            self.create_devices()
            self.request_callback('start_devices',
                                  callback=self.finish_start_devices)
        except Exception as e:
            App.get_running_app().device_exception((e, traceback.format_exc()))

        return True

    def create_sim_devices(self):
        '''Creates simulated versions of the barst devices.
        '''
        app = App.get_running_app()
        s = app.simulation_devices

        devs = self.odor_devs = [FTDIOdorsSim(mapping={
            'p{}'.format(i): o.__self__
            for i, o in enumerate(reversed(odors.children))}, dev_idx=j)
            for j, odors in enumerate(reversed(s.children))]
        for dev in devs:
            dev.activate(self)

        self.adc_devs = [FTDIADCSim(dev_idx=i)
                         for i in range(self.num_adc_chans)]

    def create_devices(self):
        server = self.server = Server()
        server.create_device()
        barst_server = server.target

        n = self.num_ftdi_chans
        n_odors = self.num_odor_chans
        n_adcs = self.num_adc_chans
        ftdis = self.ftdi_chans = [FTDIDevChannel(dev_idx=i) for i in range(n)]
        odors = self.odor_devs = [FTDIOdors(dev_idx=i) for i in range(n_odors)]
        adcs = self.adc_devs = [FTDIADC(dev_idx=i) for i in range(n_adcs)]
        if any([o.ftdi_dev[o.dev_idx] >= n for o in odors]):
            raise Exception('Odor device index larger than # FTDI devices.')
        if any([a.ftdi_dev[a.dev_idx] >= n for a in adcs]):
            raise Exception('ADC device index larger than # FTDI devices.')

        settings = [[] for _ in range(n)]
        for o in odors:
            settings[o.ftdi_dev[o.dev_idx]].append(o.get_settings())
        for a in adcs:
            settings[a.ftdi_dev[a.dev_idx]].append(a.get_settings())
        for i, s in enumerate(settings):
            ftdis[i].create_device(s, barst_server)

    def start_devices(self):
        self.server.start_channel()
        targets = [ftdi.start_channel() for ftdi in self.ftdi_chans]

        target_idx = [0, ] * self.num_ftdi_chans
        for o in self.odor_devs:
            ft_idx = o.ftdi_dev[o.dev_idx]
            o.target = targets[ft_idx][target_idx[ft_idx]]
            o.start_channel()
            target_idx[ft_idx] += 1
        for a in self.adc_devs:
            ft_idx = a.ftdi_dev[a.dev_idx]
            a.target = targets[ft_idx][target_idx[ft_idx]]
            a.start_channel()
            target_idx[ft_idx] += 1

    def finish_start_devices(self, *largs):
        self._finished_init = True
        should_stop = self._should_stop
        if should_stop is not None:
            super(InitBarstStage, self).stop(*should_stop[0], **should_stop[1])
            return
        if self.paused:
            return

        for adc in self.adc_devs:
            adc.frequency = adc.target.settings.sampling_rate
            adc.bit_depth, adc.scale, adc.offset = \
                adc.target.get_conversion_factors()
        for o in self.odor_devs:
            o.activate(self)
        self.step_stage()

    def handle_exception(self, exception, event):
        '''The overwritten method called by the devices when they encounter
        an exception.
        '''
        callback = self.exception_callback = partial(
            App.get_running_app().device_exception, exception, event)
        Clock.schedule_once(callback)

    def stop_devices(self):
        '''Called from :class:`InitBarstStage` internal thread. It stops
        and clears the states of all the devices.
        '''
        odors = self.odor_devs or []
        adcs = self.adc_devs or []
        ftdis = self.ftdi_chans or []
        unschedule = Clock.unschedule
        for dev in odors + adcs:
            if dev is not None:
                dev.deactivate(self)

        unschedule(self.exception_callback)
        if self.simulate:
            self.stop_thread()
            return

        for dev in [self.server] + odors + adcs + ftdis:
            if dev is not None:
                dev.cancel_exception()
                dev.stop_thread(True)
                dev.clear_events()

        f = []
        append = f.append
        for o in odors:
            if o.target is not None:
                append(partial(o.target.write,
                               set_low=range(8 * o.num_boards[o.dev_idx])))
                append(partial(o.target.set_state, False))
        for a in adcs:
            if a.target is not None:
                append(partial(a.target.set_state, False))
                append(a.target.read)
        for ft in ftdis:
            if ft.target is not None:
                append(ft.target.close_channel_server)

        for fun in f:
            try:
                fun()
            except:
                pass
        self.stop_thread()


class VerifyConfigStage(MoaStage):
    '''Stage that is run before the first block of each animal.

    The stage verifies that all the experimental parameters are correct and
    computes all the values, e.g. odors needed for the trials.

    If the values are incorrect, it calls
    :meth:`ExperimentApp.device_exception` with the exception.
    '''

    def recover_state(self, state):
        state.pop('finished', None)
        return super(InitBarstStage, self).recover_state(state)

    def step_stage(self, *largs, **kwargs):
        if not super(VerifyConfigStage, self).step_stage(*largs, **kwargs):
            return False

        app = App.get_running_app()
        try:
            self.read_odors()
            self.parse_odors()
            boards = App.get_running_app().simulation_devices.children
            for board, idx in self.NO_valves:
                children = boards[len(boards) - 1 - board].children
                valve = children[len(children) - 1 - idx]
                valve.background_down = 'dark-blue-led-on-th.png'
                valve.background_normal = 'dark-blue-led-off-th.png'
            for board, idx in self.rand_valves:
                children = boards[len(boards) - 1 - board].children
                valve = children[len(children) - 1 - idx]
                valve.background_down = 'brown-led-on-th.png'
                valve.background_normal = 'brown-led-off-th.png'

            boxes = app.base_stage.ids.boxes
            gui_boxes = app.boxes
            gui_boxes.clear_widgets()
            barst = self.barst
            adcs = barst.adc_devs
            displays = [BoxDisplay(box=i) for i in range(barst.num_boxes)]
            for i, display in enumerate(displays):
                display.stage = BoxStage(box=i, display=display, barst=barst,
                                         verify=self)
                dev, chan = self.adc_dev[i], self.adc_dev_chan[i]
                if not adcs[dev].active_channels[chan]:
                    raise Exception('ADC device {}, inactive channel {} used'.
                                    format(dev, chan))
                display.adc = adcs[dev]
                display.adc_channel = chan
                boxes.add_stage(display.stage)
                gui_boxes.add_widget(display)
        except Exception as e:
            app.device_exception((e, traceback.format_exc()))
            return
        self.step_stage()
        return True

    def read_odors(self):
        '''Reads odors from a csv file. Each line is 3, or 4 cols with
        valve index, odor name, and the side of the odor (r, l, rl, lr, or -).
        If using an mfc, the 4th column is either a, or b indicating the mfc
        to use of that valve.
        '''
        devs = self.barst.odor_devs
        odor_names = [['p{}'.format(i) for i in range(8 * dev.num_boards[j])]
                      for j, dev in enumerate(devs)]

        # now read the odor list
        odor_path = resources.resource_find(self.odor_path)
        with open(odor_path, 'rb') as fh:
            for row in csv.reader(fh):
                row = [elem.strip() for elem in row]
                valve, name = row[:2]
                board, idx = verify_valve_name(valve)
                if board >= len(devs):
                    raise Exception('Board number of {} is too large'.
                                    format(valve))
                if idx >= 8 * devs[board].num_boards:
                    raise Exception('Index of {} is too large'.format(valve))
                odor_names[board][idx] = name
        self.odor_names = odor_names

    def parse_odors(self):
        num_boxes = self.barst.num_boxes
        num_trials = self.num_trials
        devs = self.barst.odor_devs
        adcs = self.barst.adc_devs
        if num_boxes <= 0:
            raise Exception('Number of boxes is not positive')
        if num_boxes != len(self.adc_dev):
            raise Exception('Number of boxes, {}, does not match number of '
                            'ADC devs'.format(num_boxes))
        if num_boxes != len(self.adc_dev_chan):
            raise Exception('Number of boxes, {}, does not match number of '
                            'ADC channels'.format(num_boxes))
        if any([i >= len(adcs) for i in self.adc_dev]):
            raise Exception('ADC dev index out of range')
        if any([i >= 2 for i in self.adc_dev_chan]):
            raise Exception('ADC dev channel larger or equal to 2')

        # make sure the number of blocks match, otherwise, fill it up
        for item in (self.NO_valves, ):
            if len(item) > num_boxes:
                raise Exception('The size of {} is larger than the number '
                                'of boxes, {}'.format(item, num_boxes))
            elif len(item) < num_boxes:
                item += [item[-1]] * (num_boxes - len(item))

        if any([len(box_odors) != num_trials
                for box_odors in self.odor_selection.values()]):
            raise Exception('Number of trials does not match the number of '
                            'odors for every box')
        for odors in self.odor_selection.values():
            for board, idx in odors:
                if board >= len(devs):
                    raise Exception('Valve board number {} is too large'.
                                    format(board))
                if idx >= 8 * devs[board].num_boards:
                    raise Exception('Valve index {} too large'.format(idx))
        for board, idx in self.NO_valves:
            if board >= len(devs):
                raise Exception('Board number {} is too large for NO valve'.
                                format(board))
            if idx >= 8 * devs[board].num_boards:
                raise Exception('Index {} too large for NO valve'.format(idx))
        for board, idx in self.rand_valves:
            if board >= len(devs):
                raise Exception('Board number {} is too large for rand valve'.
                                format(board))
            if idx >= 8 * devs[board].num_boards:
                raise Exception('Index {} too large for random valve'.
                                format(idx))

    hab_dur = ConfigParserProperty(1, 'Experiment', 'hab_dur',
                                   exp_config_name, val_type=float)

    num_trials = ConfigParserProperty(1, 'Experiment', 'num_trials',
                                      exp_config_name, val_type=int)

    trial_dur = ConfigParserProperty(1, 'Experiment', 'trial_dur',
                                     exp_config_name, val_type=float)

    iti = ConfigParserProperty(1, 'Experiment', 'iti',
                               exp_config_name, val_type=float)

    post_dur = ConfigParserProperty(1, 'Experiment', 'post_dur',
                                    exp_config_name, val_type=float)

    odor_selection = ConfigPropertyDict(
        'animal: 0.p1\nanimal2: 0.p1', 'Odor', 'odor_selection', exp_config_name,
        val_type=parse_odor_list, key_type=str)

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
        odors = [list(reversed(board.children)) for board in
                 reversed(App.get_running_app().simulation_devices.children)]
        for board, idxs in enumerate(self.odor_names):
            for idx, name in enumerate(idxs):
                odors[board][idx].text = name


class BoxStage(MoaStage):
    '''In this stage, each loop runs another animal and its blocks and trials.
    '''

    _filename = ''
    _fd = None

    verify = ObjectProperty(None)

    barst = ObjectProperty(None)

    box = NumericProperty(0)

    display = ObjectProperty(None, rebind=True)

    animal_id = StringProperty('')

    odors = ObjectProperty(None, allownone=True)

    log_filename = ConfigParserProperty(
        '%m-%d-%y_{animal}.h5', 'Experiment', 'log_filename', exp_config_name,
        val_type=unicode_type)

    def on_animal_id(self, *l):
        names = self.verify.odor_names
        odors = self.verify.odor_selection.get(self.animal_id, None)
        if odors:
            names = [names[board][idx] for board, idx in odors]
            self.display.init_trials(names)
            self.odors = odors
        else:
            self.odors = None
            self.display.init_trials([])

    def recover_state(self, state):
        state.pop('finished', None)
        return super(BoxStage, self).recover_state(state)

    def initialize_box(self):
        pass

    def do_odor(self, trial, start=True):
        (d1, i1), (d2, i2) = self.verify.NO_valves[self.box], self.odors[trial]
        devs = self.barst.odor_devs

        state = 'high' if start else 'low'
        if d1 == d2:
            devs[d1].set_state(**{state: ['p{}'.format(i1), 'p{}'.format(i2)]})
        else:
            devs[d1].set_state(**{state: ['p{}'.format(i1)]})
            devs[d2].set_state(**{state: ['p{}'.format(i2)]})

    def deinitialize_box(self):
        pass

    def post_trial(self):
        '''Executed after each trial. '''
        fname = strftime(self.log_filename.format(
            **{'animal': self.animal_id}))
        filename = self._filename

        if filename != fname:
            if not fname:
                return
            fd = self._fd
            if fd is not None:
                fd.close()
            fd = self._fd = open(fname, 'a')
            fd.write('Date,Time,RatID,Block,Trial,OdorName, OdorIndex,'
                     'TrialSide,SideWent,Outcome,Rewarded?,TTNP,TINP,TTRP,'
                     'ITI\n')
            self._filename = fname
        elif not filename:
            return
        else:
            fd = self._fd
