from deeplearning.utils.logger import getContextLogger
from queue import Queue
from typing import Optional

import logging
import os


logger = logging.getLogger(__name__)


def pid_logger(logname: str, queue: Optional[Queue] = None, loglevel: int = logging.INFO):
    '''
    Within the context of a logger to document the experiment,
    do some experiment, log the activity and return the result.

    Args:
        logname (string): The name of the log to write to.
        loglevel (int): The loglevel to set the logger to.
        queue (Queue): An optional queue to write results to.

    Returns:
        pids (tuple): A tuple of logname, process and parent ids.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        with getContextLogger(level=loglevel, name=logname) as pidlogger:
            pidlogger.info(f'Process PID: {os.getpid()}')
            pidlogger.info(f'Parent PID: {os.getppid()}')
            pids = (logname, os.getpid(), os.getppid())  # A convenient tuple that demonstrates multiprocessing
            if queue is not None:
                enqueue(pids, queue)
            return pids
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
