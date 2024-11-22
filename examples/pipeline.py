from deeplearning.utils.config import Config
from deeplearning.utils.pipelines import get_tmp_dir, create_csv_pipeline

import datetime
import itertools
import kfp
import logging
import os
import sys
import tensorflow_datasets as tfds  # type: ignore
import tensorflow as tf  # type: ignore
import tfx.v1 as tfx  # type: ignore
import urllib.request as geturl

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
        handler = logging.FileHandler(f'{os.environ["PWD"]}/log/{logdate.strftime("%Y%m%d")}_example_pipeline.log')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    if hasattr(logging, conf.configuration["logging"]["level"].upper()):
        logger.setLevel(getattr(logging, conf.configuration["logging"]["level"].upper()))
        logger.warning(f'Loglevel has been set to {logger.getEffectiveLevel()} for log {__name__}.')

except Exception as e:
    raise e

PIPELINE_NAME: str = 'example'
PIPELINE_PATH: str = os.path.join('pipelines', PIPELINE_NAME)
METADATA_PATH: str = os.path.join(PIPELINE_PATH, 'metadata/metadata.db')
SERVING_PATH: str = os.path.join(PIPELINE_PATH, 'serving')

logger.info(f'Using Tensorflow version {tf.__version__}')
logger.info(f'Using TFDS version {tfds.__version__}')
logger.info(f'Using TFX version {tfx.__version__}')
logger.info(f'Using KFP version {kfp.__version__}')  # type: ignore

with get_tmp_dir(prefix='example-data') as DATA_ROOT:
    _data_url = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled/penguins_processed.csv'
    _data_filepath = os.path.join(DATA_ROOT, "data.csv")
    geturl.urlretrieve(_data_url, _data_filepath)

    with open(_data_filepath) as file:
        logger.info(f'Top records from {_data_filepath}:')
        for line in itertools.islice(file, 0, 10):
            logger.info(line.rstrip())

    pipeline = create_csv_pipeline(pipeline_name=PIPELINE_NAME,
                                   pipeline_root=PIPELINE_PATH,
                                   data_root=DATA_ROOT)

    tfx.orchestration.LocalDagRunner().run(pipeline)
