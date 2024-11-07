import logging


logger = logging.getLogger(__name__)


def results_logger(results: list[tuple], logname: str = '__main__'):
    ''' A callback to log the results of experiments to an existing log (usually __main__).

    Args:
        results (list): A list of tuples representing processing results.
        logname (string): The name of the log to write to.

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    cblogger = logging.getLogger(logname)

    try:
        for result in results:
            cblogger.info(result)
    except Exception as e:
        logger.exception(e)
        raise e
