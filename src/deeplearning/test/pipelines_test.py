from deeplearning.utils.pipelines import _bytes_feature, _float_feature, _int64_feature, serialize_array, serialize_image_data

import logging
import numpy  # type: ignore
import tensorflow as tf  # type: ignore


logger = logging.getLogger(__name__)


def test_bytes_feature():
    '''
    Function to test the _bytes_feature function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        num: bytes = bytes(123)
        fnum = _bytes_feature(num)
        assert isinstance(fnum, tf.train.Feature)
    except Exception as e:
        logger.exception(e)
        raise e


def test_float_feature():
    '''
    Function to test the _float_feature function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        num: float = 123.987
        fnum = _float_feature(num)
        assert isinstance(fnum, tf.train.Feature)
    except Exception as e:
        logger.exception(e)
        raise e


def test_int64_feature():
    '''
    Function to test the _int64_feature function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        num: int = 123
        fnum = _int64_feature(num)
        assert isinstance(fnum, tf.train.Feature)
    except Exception as e:
        logger.exception(e)
        raise e


def test_serialize_array():
    '''
    Function to test the serialize_array function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        array = numpy.array((2, 2, 2))
        sarray = serialize_array(array)
        assert tf.is_tensor(sarray)
    except Exception as e:
        logger.exception(e)
        raise e


def test_serialize_image():
    '''
    Function to test the serialize_array function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    try:
        image = numpy.array((((2, 2, 2), (2, 2, 2)), ((2, 2, 2), (2, 2, 2))))
        label = numpy.array((1))
        sfeature = serialize_image_data(image, label)
        assert isinstance(sfeature, tf.train.Example)
    except Exception as e:
        logger.exception(e)
        raise e
