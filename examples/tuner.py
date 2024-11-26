from deeplearning.utils.config import Config
from deeplearning.utils.logger import getContextLogger
from deeplearning.models.seq_conv_2d import SequentialConv2DTunable

import argparse
import datetime
import keras_tuner as tuner  # type: ignore
import logging
import numpy  # type: ignore
import os
import sys


''' Set up the Python Logger using the configuration class defaults.'''
handler: logging.Handler
logger: logging.Logger = logging.getLogger(__name__)

conf = Config()
conf.configure(config=None)

try:
    formatter = logging.Formatter(conf.configuration["logging"]["format"])

    if conf.configuration["logging"]["type"] == 'stream':
        handler = logging.StreamHandler()
        handler.setStream(getattr(sys, conf.configuration["logging"]["path"]))

    if conf.configuration["logging"]["type"] == 'file':
        logdate = datetime.datetime.now()
        handler = logging.FileHandler(f'{os.environ["PWD"]}/log/{logdate.strftime("%Y%m%d")}_template.log')

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if hasattr(logging, conf.configuration["logging"]["level"].upper()):
        logger.setLevel(getattr(logging, conf.configuration["logging"]["level"].upper()))
        logger.warning(f'Loglevel has been set to {logger.getEffectiveLevel()} for log {__name__}.')

except Exception as e:
    raise e

''' Configure argument parsing, for convenience.'''
parser = argparse.ArgumentParser()

''' Optional kwarg for keras backend override.'''
parser.add_argument('--keras-backend-override', action='store', dest='keras_backend_override')

args = parser.parse_args()

'''
Configure and import Keras and our custom sequential model class.
We can't reconfigure the keras backend once it's imported.
'''
os.environ["KERAS_BACKEND"] = (args.keras_backend_override or conf.configuration["keras"]["backend"])
logger.info(f'Configuring Keras backend as "{os.environ["KERAS_BACKEND"]}".')

import keras  # type: ignore # noqa: E402


logger.info(f'Using keras version {keras.__version__}.')

''' Get a context managed logger.'''
with getContextLogger(name='__ran0__') as ctxtlogger:

    ''' Set the loglevel.'''
    ctxtlogger.setLevel(logging.INFO)
    ctxtlogger.warning(f'Loglevel has been set to {ctxtlogger.getEffectiveLevel()} for log {ctxtlogger.name}.')

    '''
    Load some data from the keras mnist dataset into tensors.
    Data is split for training and testing purposes.
    x_* tensors hold the data.
    y_* tensors hold the model output.
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    ctxtlogger.info('Loaded the mnist dataset to training and test tensors.')

    ''' Cast image pixels to float, then normalize pixel brightness to scale [0,1].'''
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    ctxtlogger.info('Normalized tensor data to scale [0,1].')

    '''Expand the shape of the dataframes to include binary color channel.'''
    x_train = numpy.expand_dims(x_train, -1)
    x_test = numpy.expand_dims(x_test, -1)
    ctxtlogger.info('Expanded data shape to add color channel.')

    ''' Check the shape of the dataframes and log the information.'''
    ctxtlogger.info(f'Train dataframe shape (images, pwidth, pheight, channels): {x_train.shape}')
    ctxtlogger.info(f'Test dataframe shape (images, pwidth, pheight, channels): {x_test.shape}')

    ''' Initialize a tunable Hypermodel.'''
    with SequentialConv2DTunable(x_train.shape[1:],
                                 num_classes=10,
                                 metrics=[keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')],
                                 verbose=1) as tunable:

        ctxtlogger.info(f'Created tunable Hypermodel {tunable.name}.')

        ''' Build our default model, for a sanity check.'''
        hp = tuner.HyperParameters()
        ctxtlogger.info('Created Hyperparameters class.')

        try:
            model = tunable.build(hp)
            ctxtlogger.info('Built model from default parameters (sanity test).')
        except Exception as e:
            ctxtlogger.exception(e)
            raise e

        model.summary()

        try:
            ''' Validation configuration. Overwritten later.'''
            hp.Fixed('epochs', 1)
            hp.Float('validation_split', 0.15, 0.15)
            hp.Int('batch_size', 128, 128)
            tunable.fit(hp, model, x_train, y_train, callbacks=None)
            ctxtlogger.info('Fit model with one epoch from default parameters (sanity test).')
        except Exception as e:
            ctxtlogger.exception(e)
            raise e

        ''' Configure callbacks.'''
        try:
            callbacks: keras.callbacks.CallbackList = [
                keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy',
                                              mode='max',
                                              patience=2)]
            ctxtlogger.info(f'Configured callback list with {len(callbacks)} callbacks.')
        except Exception as e:
            ctxtlogger.exception(e)
            raise e

        ''' Update the parameters for trial runs.'''
        hp = tuner.HyperParameters()
        hp.Fixed('epochs', 8)
        hp.Float('validation_split', 0.1, 0.3, step=0.05)
        hp.Int('batch_size', 32, 256, step=2, sampling='log')

        ''' Run Hypermodel trials to find the best fit.'''
        try:
            tune = tuner.RandomSearch(objective=tuner.Objective('sparse_categorical_accuracy', direction='max'),
                                      max_trials=4,
                                      hypermodel=tunable,
                                      hyperparameters=hp,
                                      directory='tune',
                                      project_name='ran0',
                                      overwrite=True)
            ctxtlogger.info(f'Configured {type(tune).__name__} tuner with {type(tunable).__name__} hypermodel.')
        except Exception as e:
            ctxtlogger.exception(e)
            raise e

        ctxtlogger.info(f'Running a parameter space search on the {type(tunable).__name__} hypermodel.')
        tune.search(x_train=x_train, y_train=y_train, callbacks=callbacks)

        trials = tune.oracle.get_best_trials(num_trials=1)
        for trial in trials:
            logger.info(f'Best score {trial.score} achieved in trial {trial.trial_id} with parameters {trial.hyperparameters.values}.')
