from config import Config
from models.seq_conv_2d import SequentialConv2D

import argparse
import datetime
import logging
import numpy
import os
import sys


'''
Set up the Python Logger using class Config().
'''
logname = "mnist_example.log"
logger = logging.getLogger(logname)

conf = Config()
conf.configure(config=None)

formatter = logging.Formatter(conf.configuration["logging"]["formatter"])

if conf.configuration["logging"]["handler"] == "stream":
    handler = logging.StreamHandler()
    handler.setStream(getattr(sys, conf.configuration["logging"]["stream"]))

if conf.configuration["logging"]["handler"] == "file":
    logdate = datetime.datetime.now()
    handler = logging.FileHandler(
        f'{os.environ["PWD"]}/log/{logdate.strftime("%Y%m%d")}_{logname}')

handler.setFormatter(formatter)
logger.addHandler(handler)

if hasattr(logging, conf.configuration["logging"]["level"].upper()):
    logger.setLevel(getattr(logging, conf.configuration["logging"]["level"].
                            upper()))
    logger.warning(
        f'Package loglevel has been set to {logger.getEffectiveLevel()}')

'''
Configure argument parsing, for convenience.
'''
parser = argparse.ArgumentParser()

''' Optional kwarg for keras backend override.'''
parser.add_argument('--keras-backend-override',
                    action='store',
                    dest='keras_backend_override')

args = parser.parse_args()

'''
Configure and import deep learning modules.
'''
os.environ["KERAS_BACKEND"] = (args.keras_backend_override or
                               conf.configuration["keras"]["backend"])
logger.info(f'Configuring Keras backend as "{os.environ["KERAS_BACKEND"]}".')

import keras  # noqa: E402


'''
The keras mnist dataset is a set of character images.
Each image is 28 x 28 BW pixels and contains a digit [0-9].
'''

'''
Load some data from the keras mnist dataset into dataframes.
Data is split for training and testing purposes.
  x_* dataframes hold the images.
  y_* dataframes hold the classification labels (model output).
'''
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
logger.info('Loaded the mnist dataset to training and test dataframes.')

'''
Cast image pixels to float, then normalize pixel brightness to scale [0,1]
using the maximum value as divisor.
'''
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
logger.info('Normalized data to scale [0,1].')

'''
Expand the shape of the dataframes to include binary color channel (BW)
as the last dimension.
'''
x_train = numpy.expand_dims(x_train, -1)
x_test = numpy.expand_dims(x_test, -1)
logger.info('Expanded data shape to add color channel.')

'''
Check the shape of the dataframes and log the information.
'''
logger.info(
    f'Train dataframe shape (images, width, height, channel): {x_train.shape}')
logger.info(
    f'Test dataframe shape (images, width, height, channel): {x_test.shape}')

'''
The num_classes parameter indicates how many unique labels the model will
predict (we have 10 possible digits to match).
'''
num_classes: int = 10
logger.info(
    f'Number of classifications for the model to apply: {num_classes}')

'''
Implement a prepared, sequential convolutional neural network model
(...yikes! I had to look that up!).
'''
with SequentialConv2D(input_shape=x_train.shape[1:],
                      num_classes=num_classes) as model:
    model.summary()
