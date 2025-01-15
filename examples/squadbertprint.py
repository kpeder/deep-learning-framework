from deeplearning.utils.config import Config
from deeplearning.utils.filesystem import data_fetcher
from deeplearning.utils.logger import getContextLogger

import argparse
import datetime
import json
import logging
import numpy  # type: ignore
import os
import sys
import tempfile


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
        handler = logging.FileHandler(f'{os.environ["PWD"]}/log/{logdate.strftime("%Y%m%d")}_mnist.log')

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
from deeplearning.models.seq_conv_2d import SequentialConv2D  # noqa: E402


logger.info(f'Using keras version {keras.__version__}.')

with getContextLogger(name='__bert__', level=logging.INFO) as bertlogger:

    _squad_dataset_source_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'

    with tempfile.TemporaryDirectory(prefix='squad') as squad_data:

        _squadbert_dataset_location = data_fetcher(name='squadbert', source=_squad_dataset_source_url, dest=squad_data)

        topics: int = 0
        paragraphs: int = 0
        questions: int = 0
        answers: int = 0
        allanswerspossibleorimpossible: int = 0

        with open(_squadbert_dataset_location) as datafile:
            _raw_squadbert_data = json.load(datafile)
            bertlogger.info(f'SQuAD Data Version: {_raw_squadbert_data['version']}.')
            for topic in _raw_squadbert_data['data']:
                bertlogger.info(f'Found topic {topic['title']} with {len(topic['paragraphs'])} paragraphs.')
                for i in range(0, len(topic['paragraphs'])):
                    bertlogger.info(f'Found paragraph {topic['paragraphs'][i]['context']}.')
                    diff = answers
                    for j in range(0, len(topic['paragraphs'][i]['qas'])):
                        bertlogger.info(f'Found question {topic['paragraphs'][i]['qas'][j]['question']} with {len(topic['paragraphs'][i]['qas'][j]['answers'])} possible answers.')
                        questions += 1
                        answers += len(topic['paragraphs'][i]['qas'][j]['answers'])
                    if answers - diff == 0:
                        allanswerspossibleorimpossible += 1
                    if answers - diff == len(topic['paragraphs'][i]['qas']):
                        allanswerspossibleorimpossible += 1
                    paragraphs += 1
                topics += 1
            bertlogger.info(f'Total records: {topics}.')
            bertlogger.info(f'Total paragraphs: {paragraphs}.')
            bertlogger.info(f'Total questions: {questions}.')
            bertlogger.info(f'Total answers: {answers}.')
            bertlogger.info(f'Total unmixed questions: {answers}.')
