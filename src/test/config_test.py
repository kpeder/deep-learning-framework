from utils.config import Config

import os


def test_default_config():
    with Config() as conf:
        conf.configure(config=None)
        print(conf.configuration)
        assert conf.configuration == {
            'keras': {
                'backend': 'tensorflow'
            },
            'logging': {
                'formatter': '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                'handler': 'stream',
                'level': 'INFO',
                'stream': 'stdout'
            },
            'multiprocessing': False,
            'processors': {
                'cpus': '1,',
                'gpus': 0
            },
            'tensorflow': {}
        }


def test_update_config():
    with Config() as conf:
        conf.configure(config=None)
        conf.configure(config={
            'keras': {
                'backend': 'tensorflow'
            },
            'logging': {
                'formatter': '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                'handler': 'stream',
                'level': 'INFO',
                'stream': 'stdout'
            },
            'multiprocessing': True,
            'processors': {
                'cpus': 0,
                'gpus': 1
            },
            'tensorflow': {}
        })
        print(conf.configuration)
        assert conf.configuration == {
            'keras': {
                'backend': 'tensorflow'
            },
            'logging': {
                'formatter': '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                'handler': 'stream',
                'level': 'INFO',
                'stream': 'stdout'
            },
            'multiprocessing': True,
            'processors': {
                'cpus': 0,
                'gpus': 1
            },
            'tensorflow': {}
        }


def test_dict_config():
    with Config() as conf:
        conf.configure(config={
            'keras': {
                'backend': 'pytorch'
            },
            'multiprocessing': False,
            'processors': {
                'cpus': 0,
                'gpus': 1
            },
            'pytorch': {}
        })
        print(conf.configuration)
        assert conf.configuration == {
            'keras': {
                'backend': 'pytorch'
            },
            'multiprocessing': False,
            'processors': {
                'cpus': 0,
                'gpus': 1
            },
            'pytorch': {}
        }


def test_from_file_config():
    with Config() as conf:
        path = f'{os.environ.get('PYTHONPATH')}/defaults/config.yaml'
        config = conf.from_file(format='YAML',
                                path=path)
        conf.configure(config=config)
        print(conf.configuration)
        assert conf.configuration == {
            'keras': {
                'backend': 'tensorflow'
            },
            'logging': {
                'formatter': '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                'handler': 'stream',
                'level': 'INFO',
                'stream': 'stdout'
            },
            'multiprocessing': False,
            'processors': {
                'cpus': '1,',
                'gpus': 0
            },
            'tensorflow': {}
        }
