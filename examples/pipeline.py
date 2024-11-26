from deeplearning.utils.config import Config
from deeplearning.utils.logger import getContextLogger
from deeplearning.utils.pipelines import get_tmp_dir, create_csv_pipeline, create_tfr_pipeline, serialize_image_data

import datetime
import itertools
import kfp  # type: ignore
import logging
import numpy  # type: ignore
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

logger.info(f'Using Tensorflow version {tf.__version__}')
logger.info(f'Using TFDS version {tfds.__version__}')
logger.info(f'Using TFX version {tfx.__version__}')
logger.info(f'Using KFP version {kfp.__version__}')  # type: ignore

with getContextLogger(name='__csvf__') as csvflogger:
    csvflogger.setLevel(logging.INFO)
    csvflogger.propagate = False

    with get_tmp_dir(prefix='example-data') as CSV_DATA_ROOT:
        _data_url = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled/penguins_processed.csv'
        _data_filepath = os.path.join(CSV_DATA_ROOT, 'data.csv')
        geturl.urlretrieve(_data_url, _data_filepath)

        CSV_PIPELINE_NAME: str = 'csv_penguins'
        CSV_PIPELINE_PATH: str = os.path.join('pipelines', CSV_PIPELINE_NAME)
        CSV_METADATA_PATH: str = os.path.join(CSV_PIPELINE_PATH, 'metadata/metadata.db')
        CSV_SERVING_PATH: str = os.path.join(CSV_PIPELINE_PATH, 'serving')

        with open(_data_filepath) as file:
            csvflogger.info(f'Top records from {_data_filepath}:')
            for line in itertools.islice(file, 0, 10):
                csvflogger.info(line.rstrip())

        pipeline = create_csv_pipeline(pipeline_name=CSV_PIPELINE_NAME,
                                       pipeline_root=CSV_PIPELINE_PATH,
                                       data_root=CSV_DATA_ROOT,
                                       metadata_path=CSV_METADATA_PATH)

        tfx.orchestration.LocalDagRunner().run(pipeline)

with getContextLogger(name='__tfrf__') as mnstlogger:
    mnstlogger.setLevel(logging.INFO)
    mnstlogger.propagate = False

    with get_tmp_dir(prefix='mnist-data') as TFR_DATA_ROOT:
        _data_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
        _data_filepath = os.path.join(TFR_DATA_ROOT, 'mnist.npz')
        geturl.urlretrieve(_data_url, _data_filepath)

        TFR_PIPELINE_NAME: str = 'tfr_mnist'
        TFR_PIPELINE_PATH: str = os.path.join('pipelines', TFR_PIPELINE_NAME)
        TFR_METADATA_PATH: str = os.path.join(TFR_PIPELINE_PATH, 'metadata/metadata.db')
        TFR_SERVING_PATH: str = os.path.join(TFR_PIPELINE_PATH, 'serving')

        with numpy.load(_data_filepath, allow_pickle=True) as data:
            mnstlogger.info(f'Image datasets from {_data_filepath}: {numpy.array(list(data.keys()))}')
            x_train, y_train = data['x_train'], data['y_train']
            x_test, y_test = data['x_test'], data['y_test']

            ''' Normalize the data.'''
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            mnstlogger.info(f'Normalized {x_train.shape[0] + x_test.shape[0]}  to scale [0,1].')

            x_train = numpy.expand_dims(x_train, -1)
            mnstlogger.info(f'Reshaped training data to add a color channel: {x_train.shape}.')
            x_test = numpy.expand_dims(x_test, -1)
            mnstlogger.info(f'Reshaped test data to add a color channel: {x_test.shape}.')

            y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
            mnstlogger.info(f'Reshaped training labels to categorical for digits 0-9: {y_train.shape}.')
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
            mnstlogger.info(f'Reshaped test labels to categorical for digits 0-9: {y_test.shape}.')

            with get_tmp_dir(prefix='mnist-tfr') as TFR_ROOT:
                _tfrecord_filepath = os.path.join(TFR_ROOT, 'mnist.tfrecord')
                records = tf.io.TFRecordWriter(_tfrecord_filepath)

                for image, label in zip(x_train, y_train):
                    records.write(serialize_image_data(image, label).SerializeToString())

                for image, label in zip(x_test, y_test):
                    records.write(serialize_image_data(image, label).SerializeToString())

                records.close()
                mnstlogger.info(f'Created TFRecords file {_tfrecord_filepath} with size {round(os.stat(_tfrecord_filepath).st_size / (1024 * 1024))} MiB.')

                pipeline = create_tfr_pipeline(pipeline_name=TFR_PIPELINE_NAME,
                                               pipeline_root=TFR_PIPELINE_PATH,
                                               data_root=TFR_ROOT,
                                               metadata_path=TFR_METADATA_PATH)

                tfx.orchestration.LocalDagRunner().run(pipeline)
