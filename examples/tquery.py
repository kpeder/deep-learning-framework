from deeplearning.utils.config import Config
from deeplearning.utils.logger import getContextLogger
from deeplearning.models.seq_conv_2d import SequentialConv2DTunable

import argparse
import datetime
import keras_tuner as tuner  # type: ignore
import logging
import numpy  # type: ignore
import os
import string
import sys


'''
Set up the Python Logger using the configuration class defaults.
'''
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

'''
Configure argument parsing, for convenience.
'''
parser = argparse.ArgumentParser()

''' Optional backend override.'''
parser.add_argument('--keras-backend-override', action='store', dest='keras_backend_override')

''' Project prefix.'''
parser.add_argument('--project-prefix', action='store', dest='project_prefix')
parser.add_argument('--top-trials', action='store', dest='top_trials', default=1, type=int)

args = parser.parse_args()

'''
Configure and import Keras.
'''
os.environ["KERAS_BACKEND"] = (args.keras_backend_override or conf.configuration["keras"]["backend"])
logger.info(f'Configuring Keras backend as "{os.environ["KERAS_BACKEND"]}".')

import keras  # type: ignore # noqa: E402


logger.info(f'Using keras version {keras.__version__}.')

''' Get a logger.'''
with getContextLogger(name='__rslt__') as ctxtlogger:

    ''' Set the loglevel.'''
    ctxtlogger.setLevel(logging.INFO)
    ctxtlogger.warning(f'Loglevel has been set to {ctxtlogger.getEffectiveLevel()} for log {ctxtlogger.name}.')

    ''' Initialize a tunable Hypermodel.'''
    with SequentialConv2DTunable(input_shape=(28, 28, 1),
                                 num_classes=10,
                                 metrics=[keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')],
                                 verbose=1) as tunable:

        ctxtlogger.info(f'Created tunable Hypermodel {tunable.name}.')

        project_list: list = []
        for dir in os.listdir('tune'):
            if dir.startswith(args.project_prefix):
                project_list.append(dir)

        ctxtlogger.info(f'Processing trials in project directories {project_list}')

        combined_trials: list = []
        for project in project_list:
            try:
                tune = tuner.RandomSearch(objective=tuner.Objective('sparse_categorical_accuracy', direction='max'),
                                          #max_trials=6,
                                          hypermodel=tunable,
                                          directory="tune",
                                          project_name=project,
                                          overwrite=False)
                ctxtlogger.info(f'Configured {type(tune).__name__} tuner with {type(tunable).__name__} hypermodel.')
            except Exception as e:
                ctxtlogger.exception(e)
                raise e

            trials = tune.oracle.get_best_trials(num_trials=16)
            for trial in trials:
                if trial.status == 'COMPLETED':
                    combined_trials.append(trial)

        if len(combined_trials) == 0:
            ctxtlogger.info(f'No completed trials available.')
        else:
            combined_trials.sort(key=lambda x: x.score, reverse=True)

        for t in combined_trials[:args.top_trials]:
            ctxtlogger.info(f'Trial score {t.score} achieved with parameters {t.hyperparameters.values}.')
