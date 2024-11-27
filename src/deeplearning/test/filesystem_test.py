from deeplearning.utils.filesystem import get_tmp_dir

import logging
import os


logger = logging.getLogger(__name__)


def test_get_tmp_dir():
    '''
    Function to test the get_tmp_dir function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''

    with get_tmp_dir(prefix='test-tmpdir') as temp:
        try:
            filepath = os.path.join(temp, 'tempfile.txt')
            with open(filepath, 'w+') as tempfile:
                tempfile.write('test\n')
                assert isinstance(os.stat(filepath), os.stat_result)
        except Exception as e:
            logger.exception(e)
            raise e

    assert not os.path.isdir(temp)
