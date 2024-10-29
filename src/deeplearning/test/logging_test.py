from deeplearning.utils.logger import getContextLogger

import logging
import os


logger = logging.getLogger(__name__)


def test_default_context_logger():
    '''
    Function to test the default configuration of the getContextLogger() function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with getContextLogger(name='__test__') as ctxtlogger:
        try:
            assert ctxtlogger.disabled is False
            assert ctxtlogger.level == 30
            assert len(ctxtlogger.handlers) == 1
            for handler in ctxtlogger.handlers:
                assert isinstance(handler, logging.StreamHandler)
                assert handler.formatter._fmt == '%(asctime)s: %(name)s: %(levelname)s: %(message)s'
            assert ctxtlogger.name == '__test__'
        except Exception as e:
            logger.exception(e)
            raise e


def test_custom_context_logger():
    '''
    Function to test custom configuration of the getContextLogger() function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with getContextLogger(level='INFO', format='%(message)s', name='__test__', path=f'{os.getcwd()}/log/test.log', type='file') as ctxtlogger:
        try:
            assert ctxtlogger.disabled is False
            assert ctxtlogger.level == 20
            assert len(ctxtlogger.handlers) == 1
            for handler in ctxtlogger.handlers:
                assert isinstance(handler, logging.FileHandler)
                assert handler.formatter._fmt == '%(message)s'
                assert handler.level == logging.NOTSET
            assert ctxtlogger.name == '__test__'
        except Exception as e:
            logger.exception(e)
            raise e

        with getContextLogger(level='DEBUG', name='__test__', path='stdout', type='stream') as ctxtlogger2:
            try:
                assert ctxtlogger.disabled is False
                assert ctxtlogger.level == 10
                assert len(ctxtlogger2.handlers) == 2
                for handler in ctxtlogger2.handlers:
                    if isinstance(handler, logging.FileHandler):
                        assert handler.formatter._fmt == '%(message)s'
                        assert handler.level == logging.NOTSET
                    elif isinstance(handler, logging.StreamHandler):
                        assert handler.formatter._fmt == '%(asctime)s: %(name)s: %(levelname)s: %(message)s'
                        assert handler.level == logging.NOTSET
                    else:
                        raise TypeError(f'Unsupported Handler subclass {handler.__class__}.')
                assert ctxtlogger.name == '__test__'
            except Exception as e:
                logger.exception(e)
                raise e

        try:
            assert ctxtlogger.disabled is False
            assert ctxtlogger.level == 10
            assert len(ctxtlogger.handlers) == 1
            for handler in ctxtlogger.handlers:
                assert isinstance(handler, logging.FileHandler)
                assert handler.formatter._fmt == '%(message)s'
                assert handler.level == logging.NOTSET
            assert ctxtlogger.name == '__test__'
        except Exception as e:
            logger.exception(e)
            raise e
