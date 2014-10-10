# -*- coding: utf-8 -*-
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.properties import (ObjectProperty, StringProperty, NumericProperty,
                             BooleanProperty)
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.app import App
from kivy.utils import get_color_from_hex as rgb
from kivy.garden.graph import MeshLinePlot, SmoothLinePlot
from kivy.clock import Clock

from os import path
Builder.load_file(path.join(path.dirname(__file__), 'display.kv'))


class MainView(BoxLayout):
    pass


class DeviceSwitch(Factory.get('SwitchIcon')):

    def __init__(self, **kw):
        super(DeviceSwitch, self).__init__(**kw)
        Clock.schedule_once(self._bind_button)

    def _bind_button(self, *largs):
        if not App.get_running_app().simulate:
            self.bind(state=self.update_from_button)

    dev = ObjectProperty(None, allownone=True)

    dev_idx = NumericProperty(0)

    _dev = None

    channel = StringProperty(None)

    def on_dev(self, *largs):
        if self._dev:
            self._dev.unbind(**{self.channel: self.update_from_channel})
        self._dev = self.dev
        if self.dev and not App.get_running_app().simulate:
            self.dev.bind(**{self.channel: self.update_from_channel})

    def update_from_channel(self, *largs):
        '''A convenience method which takes the state of the simulated device
        (buttons) and the state of the actual device and returns if the
        simulated device should be `'down'` or `'normal'`.

        It is used to set the button state to match the actual device state,
        if not simulating.
        '''
        self.state = 'down' if getattr(self.dev, self.channel) else 'normal'

    def update_from_button(self, *largs):
        '''A convenience method which takes the state of the simulated device
        (buttons) and sets the state of the actual device to match it when not
        simulating.
        '''
        dev = self.dev
        if dev is not None:
            if self.state == 'down':
                dev.set_state(high=[self.channel])
            else:
                dev.set_state(low=[self.channel])


class OdorContainer(GridLayout):

    def __init__(self, dev_idx=0, num_boards=2, **kw):
        super(OdorContainer, self).__init__(**kw)
        switch = [Factory.get('OdorSwitch'), Factory.get('OdorDarkSwitch')]
        for i in range(8 * num_boards):
            self.add_widget(switch[i % 2](channel='p{}'.format(i),
                                          dev_idx=dev_idx))


class ExperimentStatus(Label):
    parent = ObjectProperty(None, allownone=True, rebind=True)


class BoxDisplay(BoxLayout):

    next_btn = ObjectProperty(None)

    exp_status = NumericProperty(0)

    stage = ObjectProperty(None, rebind=True)

    adc = ObjectProperty(None)

    adc_channel = NumericProperty(0)

    box = NumericProperty(0)

    graph = ObjectProperty(None)

    plot = ObjectProperty(None)

    def on_graph(self, *largs):
        self.plot = plot = MeshLinePlot(color=(.95, .29, .043, 1))
        self.graph.add_plot(plot)

    def on_adc(self, *largs):
        self.adc.bind(data=self.update_graph)

    def __init__(self, **kw):
        super(BoxDisplay, self).__init__(**kw)
        self.padding = [5]
        self.spacing = 10

    def init_trials(self, odors):
        status = self.ids.status
        status.clear_widgets()
        self.exp_status = 0
        ExperimentStatus = Factory.get('ExperimentStatus')
        add = status.add_widget

        add(ExperimentStatus(text='Pre'))
        for o in odors[:-1]:
            add(ExperimentStatus(text=o))
            add(ExperimentStatus(text='ITI'))
        if len(odors):
            add(ExperimentStatus(text=odors[-1]))
        add(ExperimentStatus(text='Post'))

    def update_graph(self, *l):
        data = self.adc.data[self.adc_channel]
        if not data:
            return
        xmax = self.graph.xmax
        points = self.plot.points
        f = self.adc.frequency
        if (len(data) + len(points)) / f > xmax:
            self.plot.points = [(i / f, d) for i, d in enumerate(data)]
        else:
            s = len(points) / f
            self.plot.points.extend([(i / f + s, d)
                                     for i, d in enumerate(data)])
        self.plot.points = self.plot.points
