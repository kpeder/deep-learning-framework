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
    def configure(self, config: dict=None) -> Self:
        '''
        Abstract method.
        Accepts self, configuration dictionary.
        Returns self, with configuration applied.
        '''

    @staticmethod
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

    def __enter__(self) -> Self:
        '''
        Context Manager entry method.
        '''
        return self

    def __exit__(self, *args) -> bool:
        '''
        Context Manager exit method.
        '''
        self.configuration = None
        return False

    def __init__(self):
        '''
        Create the empty top level configuration dict.
        '''
        self.configuration: dict={}

    def configure(self, config: dict=None) -> Self:
        if config == None:
            path = os.environ.get('PYTHONPATH')
            try:
                with open(f'{path}/defaults/config.yaml') as file:
                    defaults = yaml.load(file, Loader=yaml.FullLoader)
                    values = merge(self.configuration, defaults)
                    self.configuration.update(values)
                    return
            except:
                raise Exception("Could not load default configuration!")
        else:
            try:
                values = merge(self.configuration, config)
                self.configuration.update(values)
                return
            except:
                raise Exception("Could not load configuration!")

    @staticmethod
    def from_file(format: str=None, path: str=None) -> dict:
        config: dict={}
        if format == 'YAML' and path != None:
            try:
                with open(f'{path}') as file:
                    config = yaml.load(file, Loader=yaml.FullLoader)
                    return config
            except:
                raise Exception("Could not load configuration from file!")
        else:
            raise Exception("Valid path to YAML config file required!")
