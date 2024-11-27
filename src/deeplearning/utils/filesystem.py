from contextlib2 import contextmanager

import logging
import os
import tempfile as tmp


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@contextmanager
def get_tmp_dir(prefix: str, clean: bool = True):
    '''
    Creates a temporary directory and then cleans it up.

    Args:
        prefix: A string specifying the prefix to use for the temp directory name.
        clean: A flag to specify whether to clean up the directory on exit.

    Returns:
        directory: A string specifying the created directory.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''

    directory = tmp.mkdtemp(prefix=prefix)
    try:
        logger.info(f'Created temporary directory {directory}.')
        yield directory
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        if clean:
            for file in os.listdir(directory):
                path = os.path.join(directory, file)
                if os.path.isfile(path):
                    os.remove(path)
            if len(os.listdir(directory)) == 0:
                os.rmdir(directory)
            else:
                logger.warning(f'Cloud not clean up non-empty directory {directory}.')
