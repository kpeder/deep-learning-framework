from deeplearning.utils.config import Config
from deeplearning.utils.filesystem import data_fetcher
from deeplearning.utils.logger import getContextLogger
from opentelemetry import (
    metrics,
    trace
)

import argparse
import datetime
import json
import logging
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
parser.add_argument('--enable-metrics', action='store', dest='metrics')
parser.add_argument('--enable-tracing', action='store', dest='tracing')

args = parser.parse_args()

if args.metrics or conf.configuration['telemetry']['metrics']:
    meter = metrics.get_meter_provider().get_meter('__test__')

if args.tracing or conf.configuration['telemetry']['tracing']:
    tracer = trace.get_tracer('__test__')

with tracer.start_as_current_span('__test__') as testspan:

    with getContextLogger(name='__test__', level=logging.INFO) as testlogger:

            _squad_dataset_source_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'

            testspan.set_attribute('data.url', _squad_dataset_source_url)
            testlogger.info('Fetching data from {}'.format(_squad_dataset_source_url))

            with tempfile.TemporaryDirectory(prefix='squad') as squad_data:

                _squadbert_dataset_location = data_fetcher(name='squadbert', source=_squad_dataset_source_url, dest=squad_data)

                topics = meter.create_counter('topics')
                paragraphs = meter.create_counter('paragraphs')
                questions = meter.create_counter('questions')
                answers = meter.create_counter('answers')
                                                
                with open(_squadbert_dataset_location) as datafile:
                    _raw_squadbert_data = json.load(datafile)
                    testlogger.info('SQuAD Data Version: {}'.format(_raw_squadbert_data['version']))
                    for topic in _raw_squadbert_data['data']:
                        topics.add(1)
                        testlogger.info('Found topic {} with {} paragraphs.'.format(topic['title'], len(topic['paragraphs'])))
                        for i in range(0, len(topic['paragraphs'])):
                            paragraphs.add(1)
                            testlogger.info('Found paragraph {} with {} questions.'.format(topic['paragraphs'][i]['context'], len(topic['paragraphs'][i]['qas'])))
                            for j in range(0, len(topic['paragraphs'][i]['qas'])):
                                questions.add(1)
                                answers.add(len(topic['paragraphs'][i]['qas'][j]['answers']))
                                testlogger.info('Found question {} with {} possible answers.'.format(topic['paragraphs'][i]['qas'][j]['question'], len(topic['paragraphs'][i]['qas'][j]['answers'])))
