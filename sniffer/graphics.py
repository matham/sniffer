# -*- coding: utf-8 -*-

from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (
    ObjectProperty, NumericProperty, OptionProperty, StringProperty)
from kivy.lang import Builder
from kivy.garden.graph import MeshLinePlot
from kivy.app import App

from moa.base import NamedMoaBehavior
import cplcom.graphics
from sniffer.analysis import get_filter

from os import path
from scipy import signal
Builder.load_file(path.join(path.dirname(__file__), 'display.kv'))


class BoxDisplay(NamedMoaBehavior, BoxLayout):

    adc = ObjectProperty(None)

    adc_channel = NumericProperty(0)

    plot = ObjectProperty(None)

    filter_type = StringProperty('Rat')

    def on_filter_type(self, *largs):
        adc = self.adc
        if adc is None:
            return
        self.b, self.a = get_filter(adc.frequency, self.filter_type)
        self.z_initial = None

    plot_type = OptionProperty('filtered', options=['raw', 'filtered'])

    def on_plot_type(self, *l):
        self.z_initial = None

    z_initial = None

    b = None

    a = None

    last_adc_data = None

    def init_adc(self, adc, adc_channel):
        self.plot = plot = MeshLinePlot(color=(.95, .29, .043, 1))
        self.ids.graph.add_plot(plot)

        self.adc, self.adc_channel = adc, adc_channel
        self.b, self.a = get_filter(adc.frequency, self.filter_type)
        self.z_initial = None
        adc.bind(data=self.update_graph)

    def __init__(self, **kw):
        super(BoxDisplay, self).__init__(**kw)
        self.padding = [5]
        self.spacing = 5
        video_root = App.get_running_app().root.ids.video

    def update_graph(self, instance, data):
        data = data[self.adc_channel]
        if self.last_adc_data is data:
            return
        self.last_adc_data = data
        if not data:
            return

        xmax = self.ids.graph.xmax
        points = self.plot.points
        if self.plot_type == 'filtered':
            zi = self.z_initial
            if zi is None:
                zi = [0] * (max(len(self.a), len(self.b)) - 1)
            data, self.z_initial = signal.lfilter(self.b, self.a, data, zi=zi)

        f = self.adc.frequency
        if (len(data) + len(points)) / f > 1.1 * xmax:
            self.plot.points = [(i / f, d) for i, d in enumerate(data)]
        else:
            s = len(points) / f
            self.plot.points.extend([(i / f + s, d)
                                     for i, d in enumerate(data)])
        self.plot.points = self.plot.points
