from contextlib2 import contextmanager

import logging
import sys


logger = logging.getLogger(__name__)


@contextmanager
def getContextLogger(format: str = '%(asctime)s: %(name)s: %(levelname)s: %(message)s',
                     level: int = logging.WARN,
                     name: str = __name__,
                     path: str = 'stdout',
                     type: str = 'stream'):
    '''
    A function to return a Logger() instance with a Context Manager.

    Args:
        format (str): The format string to use for log entries.
        level (int): The initial loglevel to use. One of [0, 10, 20, 30, 40, 50]
        name (str): A name for the log.
        path (str): A file path or stream to log to.
        type (str): The type of log handler to use. One of ['file', 'stream'].

    Returns:
        ctxtlogger (logging.Logger): A configured instance of the Logger class, enabled with the @contextmanager decorator.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    ctxtlogger: logging.Logger | None
    handler: logging.Handler

    ctxtlogger = logging.getLogger(name)

    try:
        formatter = logging.Formatter(format)

    except Exception as e:
        logger.exception(e)
        raise e

    if type.lower() == 'stream':
        try:
            handler = logging.StreamHandler()
            handler.setStream(getattr(sys, path.lower()))

        except Exception as e:
            logger.exception(e)
            raise e

    elif type.lower() == 'file':
        try:
            handler = logging.FileHandler(path)

        except Exception as e:
            logger.exception(e)
            raise e

    else:
        try:
            raise Exception(f'Unsupported value for handler type {type}')

        except Exception as e:
            logger.exception(e)
            raise e

    try:
        handler.setFormatter(formatter)
        ctxtlogger.addHandler(handler)
        if level in [0, 10, 20, 30, 40, 50]:
            ctxtlogger.setLevel(level)
            logger.warning(f'Loglevel has been set to {ctxtlogger.getEffectiveLevel()} for log {name}.')

    except Exception as e:
        logger.exception(e)
        raise e

    try:
        yield ctxtlogger

    finally:
        ctxtlogger.removeHandler(handler)
        ctxtlogger = None
