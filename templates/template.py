from deeplearning.utils.config import Config
from deeplearning.utils.logger import getContextLogger

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

args = parser.parse_args()

'''
Configure and import Keras.
'''
os.environ["KERAS_BACKEND"] = (args.keras_backend_override or conf.configuration["keras"]["backend"])
logger.info(f'Configuring Keras backend as "{os.environ["KERAS_BACKEND"]}".')

import keras  # noqa: E402


logger.info(f'Using keras version {keras.__version__}.')

with getContextLogger(name='__cntx__') as ctxlogger:
    ctxlogger.setLevel(logging.DEBUG)
    ctxlogger.warning(f'Loglevel has been set to {ctxlogger.getEffectiveLevel()} for log __cntx__.')
    ctxlogger.debug('A logline.')

    with getContextLogger(format=conf.configuration["logging"]["format"],
                          level=conf.configuration["logging"]["level"],
                          name='__with__',
                          path=conf.configuration["logging"]["path"],
                          type=conf.configuration["logging"]["type"]) as withlogger:
        withlogger.setLevel(logging.DEBUG)
        withlogger.warning(f'Loglevel has been set to {withlogger.getEffectiveLevel()} for log __with__.')
        withlogger.debug('A logline.')

    ctxlogger.debug('Another logline.')
