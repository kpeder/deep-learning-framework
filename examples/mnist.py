from deeplearning.utils.config import Config

import argparse
import datetime
import logging
import numpy
import os
import sys


'''
Set up the Python Logger using the configuration class defaults.
'''
logger = logging.getLogger(__name__)

conf = Config()
conf.configure(config=None)

try:
    formatter = logging.Formatter(conf.configuration["logging"]["format"])

    if conf.configuration["logging"]["type"] == 'stream':
        handler = logging.StreamHandler()
        handler.setStream(getattr(sys, conf.configuration["logging"]["path"]))

    if conf.configuration["logging"]["type"] == 'file':
        logdate = datetime.datetime.now()
        handler = logging.FileHandler(f'{os.environ["PWD"]}/log/{logdate.strftime("%Y%m%d")}_mnist.log')

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if hasattr(logging, conf.configuration["logging"]["level"].upper()):
        logger.setLevel(getattr(logging, conf.configuration["logging"]["level"].upper()))
        logger.warning(f'Loglevel has been set to {logger.getEffectiveLevel()} for log {__name__}.')

except Exception as e:
    raise e

'''
Configure argument parsing, for convenience.
'''
parser = argparse.ArgumentParser()

''' Optional kwarg for keras backend override.'''
parser.add_argument('--keras-backend-override', action='store', dest='keras_backend_override')

args = parser.parse_args()

'''
Configure and import deep learning modules. We can't reconfigure the keras backend once it's imported.
'''
os.environ["KERAS_BACKEND"] = (args.keras_backend_override or conf.configuration["keras"]["backend"])
logger.info(f'Configuring Keras backend as "{os.environ["KERAS_BACKEND"]}".')

import keras  # noqa: E402
from deeplearning.models.seq_conv_2d import SequentialConv2D  # noqa: E402


logger.info(f'Using keras version {keras.__version__}.')

'''
The keras mnist dataset is a set of character images.
Each image is 28 x 28 BW pixels and contains a digit [0-9].
'''

'''
Load some data from the keras mnist dataset into tensors.
Data is split for training and testing purposes.
  x_* tensors hold the data.
  y_* tensors hold the model output.
'''
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
logger.info('Loaded the mnist dataset to training and test tensors.')

'''
Cast image pixels to float, then normalize pixel brightness to scale [0,1].
'''
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
logger.info('Normalized tensor data to scale [0,1].')

'''
Expand the shape of the dataframes to include binary color channel.
'''
x_train = numpy.expand_dims(x_train, -1)
x_test = numpy.expand_dims(x_test, -1)
logger.info('Expanded data shape to add color channel.')

'''
Check the shape of the dataframes and log the information.
'''
logger.info(f'Train dataframe shape (images, pwidth, pheight, channels): {x_train.shape}')
logger.info(f'Test dataframe shape (images, pwidth, pheight, channels): {x_test.shape}')

'''
The num_classes parameter indicates how many unique labels the model will predict.
'''
num_classes: int = 10
logger.info(f'Number of classifications for the model to predict: {num_classes}')

'''
Implement a prepared, sequential convolutional neural network model.
'''
with SequentialConv2D(input_shape=x_train.shape[1:], num_classes=num_classes) as model:
    model.summary()
