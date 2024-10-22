from config import Config


def test_default_config():
    with Config() as conf:
        conf.configure(config=None)
        print(conf.configuration)
        assert(conf.configuration == {'keras': {'backend': 'tensorflow'}, 'multiprocessing': False, 'processors': {'cpus': '1,', 'gpus': 0}, 'tensorflow': {}})

def test_update_config():
    with Config() as conf:
        conf.configure(config=None)
        conf.configure(config={'keras': {'backend': 'tensorflow'}, 'multiprocessing': True, 'processors': {'cpus': 0, 'gpus': 1}, 'tensorflow': {}})
        print(conf.configuration)
        assert(conf.configuration == {'keras': {'backend': 'tensorflow'}, 'multiprocessing': True, 'processors': {'cpus': 0, 'gpus': 1}, 'tensorflow': {}})

def test_dict_config():
    with Config() as conf:
        conf.configure(config={'keras': {'backend': 'pytorch'}, 'multiprocessing': False, 'processors': {'cpus': 0, 'gpus': 1}, 'pytorch': {}})
        print(conf.configuration)
        assert(conf.configuration == {'keras': {'backend': 'pytorch'}, 'multiprocessing': False, 'processors': {'cpus': 0, 'gpus': 1}, 'pytorch': {}})
