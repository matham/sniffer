from setuptools import setup, find_packages
import sniffer


setup(
    name='Sniffer',
    version=sniffer.__version__,
    packages=find_packages(),
    install_requires=['moa', 'pybarst', 'moadevs'],
    author='Matthew Einhorn',
    author_email='moiein2000@gmail.com',
    url='https://cpl.cornell.edu/',
    license='MIT',
    description='Sniffer box project.',
    entry_points={'console_scripts':
                  ['sniffer=sniffer.main:run_app']},
    )
