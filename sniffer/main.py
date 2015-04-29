'''The main module that starts the experiment.
'''

__all__ = ('ExperimentApp', 'run_app')


from functools import partial
from os.path import join, dirname

from cplcom.app import ExperimentApp, run_app as run_cpl_app

from kivy.properties import ObjectProperty, StringProperty
from kivy.resources import resource_add_path
from kivy.lang import Builder

import sniffer.stages


class SnifferApp(ExperimentApp):

    filter_type = StringProperty('Rat')

    def __init__(self, **kw):
        super(SnifferApp, self).__init__(**kw)
        resource_add_path(join(dirname(dirname(__file__)), 'data'))
        Builder.load_file(join(dirname(__file__), 'Experiment.kv'))
        self.inspect = True

run_app = partial(run_cpl_app, SnifferApp)

if __name__ == '__main__':
    run_app()
