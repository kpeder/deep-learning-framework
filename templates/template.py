from deeplearning.utils.config import Config

import argparse
import datetime
import logging
import os
import sys


'''
Set up the Python Logger using the configuration class defaults.
'''
logger = logging.getLogger(__name__)

conf = Config()
conf.configure(config=None)

formatter = logging.Formatter(conf.configuration["logging"]["formatter"])

if conf.configuration["logging"]["handler"] == "stream":
    handler = logging.StreamHandler()
    handler.setStream(getattr(sys, conf.configuration["logging"]["stream"]))

if conf.configuration["logging"]["handler"] == "file":
    logdate = datetime.datetime.now()
    handler = logging.FileHandler(f'{os.environ["PWD"]}/log/{logdate.strftime("%Y%m%d")}_template.log')

handler.setFormatter(formatter)
logger.addHandler(handler)

if hasattr(logging, conf.configuration["logging"]["level"].upper()):
    logger.setLevel(getattr(logging, conf.configuration["logging"]["level"].upper()))
    logger.warning(f'Program loglevel has been set to {logger.getEffectiveLevel()}')

'''
Configure argument parsing, for convenience.
'''
parser = argparse.ArgumentParser()

''' Optional backend override.'''
parser.add_argument('--keras-backend-override', action='store', dest='keras_backend_override')

args = parser.parse_args()

'''
Configure and import Keras.
'''
os.environ["KERAS_BACKEND"] = (args.keras_backend_override or conf.configuration["keras"]["backend"])
logger.info(f'Configuring Keras backend as "{os.environ["KERAS_BACKEND"]}".')

import keras  # noqa: E402

logger.info(f'Using keras version {keras.__version__}.')
