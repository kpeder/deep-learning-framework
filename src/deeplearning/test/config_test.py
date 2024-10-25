from deeplearning.utils.config import Config

import logging
import os


logger = logging.getLogger(__name__)


def test_default_config():
    '''
    Function to test the default configuration of the Config() class.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with Config() as conf:
        conf.configure(config=None)
        try:
            assert conf.configuration == {
                'keras': {
                    'backend': 'tensorflow'
                },
                'logging': {
                    'format': '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                    'level': 'INFO',
                    'path': 'stdout',
                    'type': 'stream'
                },
                'multiprocessing': False
            }
        except Exception as e:
            logger.exception(e)
            raise e


def test_update_config():
    '''
    Function to test updates to the default configuration of the Config() class.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with Config() as conf:
        conf.configure(config=None)
        conf.configure(config={
            'multiprocessing': True
        })
        try:
            assert conf.configuration == {
                'keras': {
                    'backend': 'tensorflow'
                },
                'logging': {
                    'format': '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                    'level': 'INFO',
                    'path': 'stdout',
                    'type': 'stream'
                },
                'multiprocessing': True
            }
        except Exception as e:
            logger.exception(e)
            raise e


def test_dict_config():
    '''
    Function to test the configuration of the Config() class from dictionary.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with Config() as conf:
        conf.configure(config={
            'keras': {
                'backend': 'pytorch'
            },
            'logging': {
                'format': '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                'level': 'INFO',
                'path': None,
                'type': 'file'
            },
            'multiprocessing': False
        })
        try:
            assert conf.configuration == {
                'keras': {
                    'backend': 'pytorch'
                },
            'logging': {
                'format': '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                'level': 'INFO',
                'path': None,
                'type': 'file'
            },
                'multiprocessing': False
            }
        except Exception as e:
            logger.exception(e)
            raise e


def test_from_file_config():
    '''
    Function to test the configuration of the Config() class from file.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with Config() as conf:
        path = f'{os.environ.get('PYTHONPATH')}/deeplearning/config.yaml'
        config = conf.from_file(format='YAML',
                                path=path)
        conf.configure(config=config)
        try:
            assert conf.configuration == {
                'keras': {
                    'backend': 'tensorflow'
                },
                'logging': {
                    'format': '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                    'level': 'INFO',
                    'path': 'stdout',
                    'type': 'stream'
                },
                'multiprocessing': False
            }
        except Exception as e:
            logger.exception(e)
            raise e
