from abcmeta import ABC, abstractmethod
from contextlib2 import AbstractContextManager
from typing import Self
from mergedeep import merge

import os, yaml


class BaseConfig(ABC, AbstractContextManager):
    '''
    The Abstract Base Class for the Config Class.
    '''

    @abstractmethod
    def configure(self, config: dict) -> Self:
        '''
        Abstract method.
        Accepts self, configuration dictionary.
        Returns self, with configuration applied.
        '''

    @abstractmethod
    def from_file(format: str=None, path: str=None) -> dict:
        '''
        Abstract method.
        Accepts format, path of a configuration file.
        Returns dictionary of configuration values.
        '''


class Config(BaseConfig):
    '''
    Config Class for the LLM framework, implements BaseConfig.
    '''

    def __enter__(self):
        return self

    def __exit__(sefl, exc_type, exc_value, exc_traceback):
        return False

    def __init__(self):
        '''
        Create the empty top level configuration dict.
        '''
        self.configuration: dict={}

    def configure(self, config: dict) -> Self:
        if config == None:
            path = os.environ.get('PYTHONPATH')
            try:
                with open(f'{path}/defaults/config.yaml') as file:
                    defaults = yaml.load(file, Loader=yaml.FullLoader)
                    values = merge(self.configuration, defaults)
                    self.configuration.update(values)
                    return
            except:
                raise Exception("No configuration found!")
        else:
            try:
                values = merge(self.configuration, config)
                self.configuration.update(values)
                return
            except:
                raise Exception("No configuration found!")

    def from_file(format: str=None, path: str=None) -> dict:
        config: dict={}
        if format == None & path == None:
            return config
        else:
            return config
