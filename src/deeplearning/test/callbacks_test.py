from deeplearning.utils.callbacks import results_logger

import logging


logger = logging.getLogger(__name__)


def test_results_logger(caplog):
    '''
    Function to test the results_logger callback function.

    Args:
        caplog (caplog): A PyTest fixture for log capture.

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    caplog.set_level(logging.INFO)

    logger = logging.getLogger('deeplearning.test.callbacks_test')
    logger.setLevel(logging.INFO)

    try:
        results_list: list = [(None, 12, 4), ('__main__', logging.INFO), ([], None, 42)]
        string_list: list = []
        for t in results_list:
            string_list.append(str(t))
        results_logger(results_list, 'deeplearning.test.callbacks_test')
        assert len(caplog.records) == len(results_list)
        for log in caplog.record_tuples:
            assert log[0] == 'deeplearning.test.callbacks_test'
            assert log[1] == logging.INFO
            assert log[2] in string_list
    except Exception as e:
        logger.exception(e)
        raise e
