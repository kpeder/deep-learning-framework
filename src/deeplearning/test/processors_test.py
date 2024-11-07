from contextlib2 import contextmanager
from deeplearning.utils.processors import dequeue, enqueue, pid_logger
from queue import Queue

import logging


logger = logging.getLogger(__name__)


@contextmanager
def getContextQueue():
    ''' A function to make the Queue class context managed.'''
    queue = Queue()

    try:
        yield queue
    finally:
        queue = None


def test_enqueue_dequeue():
    '''
    Function to test the enqueue and dequeue functions.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''

    with getContextQueue() as queue:
        try:
            args_list: list = [(None, 12, 4), ('__main__', logging.INFO), ([], None, 42)]
            for args in args_list:
                enqueue(args, queue)

            result_list: list = []
            while not queue.empty():
                result_list.append(dequeue(queue))

            assert args_list == result_list
        except Exception as e:
            logger.exception(e)
            raise e


def test_pid_logger(caplog):
    '''
    Function to test the pid_logger function.

    Args:
        caplog (caplog): A PyTest fixture for log capture.

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    caplog.set_level(logging.INFO)

    try:
        result = pid_logger('deeplearning.test.processors_test')
        assert len(result) == 3
        assert isinstance(result, tuple)
        assert isinstance(result[0], str) and result[0] == 'deeplearning.test.processors_test'
        assert isinstance(result[1], int)
        assert isinstance(result[2], int)
        assert len(caplog.records) == 3
        for log in caplog.record_tuples[1:]:
            assert log[0] == 'deeplearning.test.processors_test'
            assert log[1] == logging.INFO
            assert 'PID' in log[2]
    except Exception as e:
        logger.exception(e)
        raise e


def test_pid_logger_with_queue():
    '''
    Function to test the pid_logger function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with getContextQueue() as queue:
        try:
            pid_logger('deeplearning.test.processors_test', queue)
            result = queue.get()
            assert len(result) == 3
            assert isinstance(result, tuple)
            assert isinstance(result[0], str) and result[0] == 'deeplearning.test.processors_test'
            assert isinstance(result[1], int)
            assert isinstance(result[2], int)
        except Exception as e:
            logger.exception(e)
            raise e
