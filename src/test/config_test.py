from config import Config


if __name__ == "__main__":
    with Config() as conf:
        conf.configure(config=None)
        print(conf.configuration)

    with Config() as conf:
        conf.configure(config={'keras': {'backend': 'pytorch'}, 'multiprocessing': False, 'pytorch': {'cpus': 0, 'gpus': 1}})
        print(conf.configuration)
