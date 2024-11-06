from deeplearning.utils.logger import getContextLogger
from queue import Queue
from typing import Optional

import logging
import os

logger = logging.getLogger(__name__)


def callback_logger(results: list[tuple], logname: str = '__main__'):
    ''' A callback to log the results of experiments.'''
    cblogger = logging.getLogger(logname)

    try:
        for result in results:
            cblogger.info(result)
    except Exception as e:
        logger.exception(e)
        raise e


def do(logname: str, queue: Optional[Queue] = None):
    '''
    Within the context of a logger to document the experiment,
    do some experiment, log the activity and return the result.

    Args:
        logname (string): The name of the log to write to.
        queue (multiprocessing.Queue): An optional queue object to write results to.

    Returns:
        retval (tuple): An enqueuable tuple of relevant values.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        with getContextLogger(name=logname) as ctxtlogger:
            ctxtlogger.setLevel(logging.DEBUG)
            ctxtlogger.warning(f'Loglevel has been set to {ctxtlogger.getEffectiveLevel()} for log {logname}.')
            ctxtlogger.debug('A logline.')
            ctxtlogger.info(f'Process PID: {os.getpid()}')
            ctxtlogger.info(f'Parent PID: {os.getppid()}')
            ctxtlogger.debug('Another logline.')
            retval = logname, (os.getpid(), os.getppid())  # A convenient tuple that demonstrates multiprocessing
            if queue is not None:
                enqueue(retval, queue)
            return retval
    except Exception as e:
        logger.exception(e)
        raise e


def dequeue(queue: Queue):
    ''' Dequeue wrapper.'''
    try:
        result = queue.get(block=False)
        return result
    except Exception as e:
        logger.exception(e)
        raise e


def enqueue(item: tuple, queue: Queue):
    ''' Enqueue wrapper.'''
    try:
        queue.put(item)
    except Exception as e:
        logger.exception(e)
        raise e
