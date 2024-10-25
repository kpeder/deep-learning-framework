from deeplearning.utils.config import Config
from deeplearning.models.seq_conv_2d import SequentialConv2D

import logging
import os


logger = logging.getLogger(__name__)

conf = Config()
conf.configure()

os.environ["KERAS_BACKEND"] = conf.configuration["keras"]["backend"]


def test_class_instantiation(shape=(28, 28, 1), classes=10):
    '''
    Function to test the instantiation of the SequentialConv2D() class.

    Args:
        shape (tuple): Shape of the model input tensor.
        classes (int): Number of classes for the model to predict.

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with SequentialConv2D(input_shape=shape, num_classes=classes) as model:
        try:
            assert model.input_shape[1:] == (28, 28, 1)
        except Exception as e:
            logger.exception(e)
            raise e


def test_model_generation(shape=(28, 28, 1), classes=10):
    '''
    Function to test the generation of a valid model using the SequentialConv2D() class.

    Args:
        shape (tuple): Shape of the model input tensor.
        classes (int): Number of classes for the model to predict.

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with SequentialConv2D(input_shape=shape, num_classes=classes) as model:
        try:
            model.summary()
        except Exception as e:
            logger.exception(e)
            raise e
