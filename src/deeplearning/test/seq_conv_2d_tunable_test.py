from deeplearning.utils.config import Config
from deeplearning.models.seq_conv_2d import SequentialConv2DTunable

import keras_tuner as tuner  # type: ignore
import logging
import os


logger = logging.getLogger(__name__)

conf = Config()
conf.configure()

os.environ["KERAS_BACKEND"] = conf.configuration["keras"]["backend"]


def test_class_instantiation(shape=(28, 28, 1), classes=10, verbose=1):
    '''
    Function to test the instantiation of the SequentialConv2D() class.

    Args:
        shape (tuple): Shape of the model input tensor.
        classes (int): Number of classes for the model to predict.
        verbose (int): Level of verbosity, from 0 to 2.

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with SequentialConv2DTunable(input_shape=shape,
                                 num_classes=classes,
                                 verbose=verbose) as model:
        try:
            assert model.input_shape == shape
            assert model.num_classes == classes
            assert model.verbose == verbose
        except Exception as e:
            logger.exception(e)
            raise e


def test_model_generation(shape=(28, 28, 1),
                          classes=10,
                          verbose=1):
    '''
    Function to test the generation of a valid model using the SequentialConv2D() class.

    Args:
        shape (tuple): Shape of the model input tensor.
        classes (int): Number of classes for the model to predict.
        verbose (int): Level of verbosity, from 0 to 2.

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''
    with SequentialConv2DTunable(input_shape=shape,
                                 num_classes=classes,
                                 verbose=verbose) as tunable:
        try:
            hp = tuner.HyperParameters()
            model = tunable.build(hp)
            model.summary()
        except Exception as e:
            logger.exception(e)
            raise e
