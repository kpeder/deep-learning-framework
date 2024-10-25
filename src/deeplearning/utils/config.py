from abcmeta import ABC, abstractmethod
from contextlib2 import AbstractContextManager
from typing import Self
from mergedeep import merge

import logging
import os
import yaml


logger = logging.getLogger(__name__)


class BaseConfig(ABC, AbstractContextManager):
    '''
    An Abstract Base Class for the Config() class.
    '''

    @abstractmethod
    def configure(self, config: dict = None) -> Self:
        '''
        Abstract method to configure the class from a configuration dictionary.

        Args:
            self (Self): A concrete subclass of BaseConfig().
            config (dict): Dictionary containing the configuration parameters.

        Returns:
            self (Self): A concrete subclass of BaseConfig(), with self.configuration populated.

        Raises:
            e (Exception): Any unhandled exception, as necessary.
        '''

    @staticmethod
    @abstractmethod
    def from_file(format: str = None, path: str = None) -> dict:
        '''
        Static abstract method to fetch configuration from file.

        Args:
            format (str): The format of a configuration file.
            path (str): The path to a configuration file.

        Returns:
            config (dict): A dictionary of configuration values.

        Raises:
            e (Exception): Any unhandled exception, as necessary.
        '''


class Config(BaseConfig):
    '''
    A configuration class for the deeplearning framework. Implements BaseConfig().
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

    def __init__(self) -> Self:
        '''
        Create the empty top level configuration dict.

        Args:
            self (Self): A concrete subclass of BaseConfig().

        Returns:
            self (Self): An initialized concrete subclass of BaseConfig().
        '''
        self.configuration: dict = {}

    def configure(self, config: dict = None) -> Self:
        '''
        Method to configure the class from a configuration dictionary.

        Args:
            self (Self): An instance of Config().
            config (dict): Dictionary containing the configuration parameters.

        Returns:
            self (Self): An instance of Config(), with self.configuration populated.

        Raises:
            e (Exception): Any unhandled exception, as necessary.
        '''
        if config is None:
            path = os.environ.get('PYTHONPATH')
            try:
                config = self.from_file(format='YAML', path=f'{path}/deeplearning/config.yaml')
                values = merge(self.configuration, config)
                self.configuration.update(values)
                return
            except Exception as e:
                logger.exception(e)
                raise e
        else:
            values = merge(self.configuration, config)
            self.configuration.update(values)
            return

    @staticmethod
    def from_file(format: str = None, path: str = None) -> dict:
        '''
        Static method to fetch configuration from file.

        Args:
            format (str): The format of a configuration file.
            path (str): The path to a configuration file.

        Returns:
            config (dict): A dictionary of configuration values.

        Raises:
            e (Exception): Any unhandled exception, as necessary.
        '''
        config: dict = {}
        if format.upper() != 'YAML':
            try:
                raise Exception('Unsupported configuration file format!')
            except Exception as e:
                logger.exception(e)
                raise e
        elif format.upper() == 'YAML' and path is not None:
            try:
                with open(f'{path}') as file:
                    config = yaml.load(file, Loader=yaml.FullLoader)
                    return config
            except Exception as e:
                logger.exception(e)
                raise e
