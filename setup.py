from setuptools import setup

from pip.req import parse_requirements
install_reqs = parse_requirements('environment.yml')

setup(name='hetmech',
      description='A search engine for hetnets',
      long_description='Matrix implementations of path-count-based measures',
      url='https://github.com/greenelab/hetmech',
      license='BSD 3-Clause License',
      packages=['hetmech'],
      install_requires=['hetio']
      )
