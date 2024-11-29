from deeplearning.utils.config import Config
from deeplearning.utils.filesystem import data_fetcher, data_pusher
from deeplearning.utils.logger import getContextLogger
from deeplearning.utils.pipelines import create_csv_pipeline, create_tfr_pipeline, serialize_image_data

import argparse
import datetime
import kfp  # type: ignore
import kfp.gcp as gcp  # type: ignore
import logging
import numpy  # type: ignore
import os
import sys
import tempfile
import tensorflow as tf  # type: ignore
import tfx.v1 as tfx  # type: ignore


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
logger.info(f'Using TFX version {tfx.__version__}')
logger.info(f'Using KFP version {kfp.__version__}')  # type: ignore

''' Configure argument parsing, for convenience.'''
parser = argparse.ArgumentParser()

''' Specify the pipeline's storage bucket.'''
parser.add_argument('--gcs-bucket', action='store', dest='gcs_bucket')

args = parser.parse_args()

''' Name of the CSV pipeline.'''
_csv_pipeline_name: str = 'penguins'

''' Whether to enable the cache on the CSV pipeline.'''
_csv_pipeline_cache: bool = False

''' Root path of the pipeline. Should be a cloud storage path.'''
_csv_pipeline_root: str = os.path.join(f'gs://{args.gcs_bucket}/pipelines',
                                       _csv_pipeline_name)

_csv_data_root: str = os.path.join(f'gs://{args.gcs_bucket}/data',
                                   _csv_pipeline_name)

_csv_data_source_url: str = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled/penguins_processed.csv'

''' Serving path for the model. Should be a cloud storage path.'''
_csv_serving_root = os.path.join(f'gs://{args.gcs_bucket}/serving-models',
                                 _csv_pipeline_name)

with getContextLogger(name='__csvf__') as csvflogger:
    csvflogger.setLevel(logging.INFO)
    csvflogger.propagate = False

    ''' Fetch the pipeline input data and upload to storage.'''
    csv_location = data_fetcher(name=_csv_pipeline_name,
                                source=_csv_data_source_url,
                                dest=_csv_data_root)
    csvflogger.info(f'Fetched data and uploaded to bucket location {csv_location}')

    ''' Initialize the pipeline.'''
    csv_pipeline = create_csv_pipeline(
        pipeline_name=_csv_pipeline_name,
        pipeline_root=_csv_pipeline_root,
        data_root=_csv_data_root,
        enable_cache=_csv_pipeline_cache
    )
    csvflogger.info(f'Created csv pipeline with id {csv_pipeline.id}.')

    csv_kubeflow_metadata_config = tfx.orchestration.experimental.get_default_kubeflow_metadata_config()
    csv_kubeflow_metadata_config.grpc_config.grpc_service_host.value = "metadata-grpc-service.kubeflow"
    csv_kubeflow_metadata_config.grpc_config.grpc_service_port.value = "8080"
    csvflogger.info('Created kubeflow pipeline metadata config.')

    csv_pipeline_config = tfx.orchestration.experimental.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=csv_kubeflow_metadata_config,
        pipeline_operator_funcs=[
            gcp.use_gcp_secret('user-gcp-sa')
        ],
        tfx_image=f'gcr.io/tfx-oss-public/tfx:{tfx.__version__}')
    csvflogger.info('Created kubeflow pipeline runner config.')

    csv_pipeline_runner = tfx.orchestration.experimental.KubeflowDagRunner(
        config=csv_pipeline_config,
        output_filename=f'{_csv_pipeline_name}.yaml'
    )
    csvflogger.info('Created kubeflow pipeline runner.')

    csv_pipeline_runner.run(csv_pipeline)
    csvflogger.info(f'Created kubeflow pipeline runner deployment {_csv_pipeline_name}.yaml')


''' Name of the TFRecord pipeline.'''
_mnst_pipeline_name: str = 'mnist'

''' Whether to enable the cache on the CSV pipeline.'''
_mnst_pipeline_cache: bool = False

''' Root path of the pipeline. Should be a cloud storage path.'''
_mnst_pipeline_root: str = os.path.join(f'gs://{args.gcs_bucket}/pipelines',
                                        _mnst_pipeline_name)

_mnst_data_root: str = os.path.join(f'gs://{args.gcs_bucket}/data',
                                    _mnst_pipeline_name)

_mnst_data_source_url: str = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

''' Serving path for the model. Should be a cloud storage path.'''
_mnst_serving_root = os.path.join(f'gs://{args.gcs_bucket}/serving-models',
                                  _mnst_pipeline_name)

''' Create a pipeline with TFR data source.'''
with getContextLogger(name='__mnst__') as mnstlogger:
    mnstlogger.setLevel(logging.INFO)
    mnstlogger.propagate = False

    with tempfile.TemporaryDirectory(prefix=_mnst_pipeline_name) as _mnst_data_temp:

        ''' Fetch the pipeline input data and upload to storage.'''
        mnst_location = data_fetcher(name=_mnst_pipeline_name,
                                     source=_mnst_data_source_url,
                                     dest=_mnst_data_temp)
        mnstlogger.info(f'Fetched data and saved to {mnst_location} for processing.')

        ''' Load the MNIST dataset in numpy.'''
        with numpy.load(mnst_location, allow_pickle=True) as _tfr_images:
            mnstlogger.info(f'Image datasets from {mnst_location}: {numpy.array(list(_tfr_images.keys()))}')
            x_train, y_train = _tfr_images['x_train'], _tfr_images['y_train']
            x_test, y_test = _tfr_images['x_test'], _tfr_images['y_test']

            ''' Normalize the image data.'''
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            mnstlogger.info(f'Normalized {x_train.shape[0] + x_test.shape[0]}  to scale [0,1].')

            ''' Reshape the image data.'''
            x_train = numpy.expand_dims(x_train, -1)
            mnstlogger.info(f'Reshaped training data to add a color channel: {x_train.shape}.')
            x_test = numpy.expand_dims(x_test, -1)
            mnstlogger.info(f'Reshaped test data to add a color channel: {x_test.shape}.')

            ''' Reshape the labels.'''
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
            mnstlogger.info(f'Reshaped training labels to categorical for digits 0-9: {y_train.shape}.')
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
            mnstlogger.info(f'Reshaped test labels to categorical for digits 0-9: {y_test.shape}.')

            ''' Get a temp dir that will be cleaned up after example generation.'''
            with tempfile.TemporaryDirectory(prefix='-'.join((_mnst_pipeline_name, 'tfrecords'))) as _tfrecords_out:
                _tfrecords_filepath = os.path.join(_tfrecords_out, 'mnist.tfrecord')
                tfrecords = tf.io.TFRecordWriter(_tfrecords_filepath)

                ''' Serialize the training dataset to TFRecord format.'''
                for image, label in zip(x_train, y_train):
                    tfrecords.write(serialize_image_data(image, label).SerializeToString())

                ''' Serialize the test dataset to TFRecord format.'''
                for image, label in zip(x_test, y_test):
                    tfrecords.write(serialize_image_data(image, label).SerializeToString())

                tfrecords.close()
                mnstlogger.info(f'Created TFRecords file {_tfrecords_filepath} with size {round(os.stat(_tfrecords_filepath).st_size / (1024 * 1024))} MiB.')

                mnst_location = data_pusher(name=_mnst_pipeline_name, source=_tfrecords_filepath, dest=_mnst_data_root)

    ''' Initialize the pipeline.'''
    pipeline = create_tfr_pipeline(
        pipeline_name=_mnst_pipeline_name,
        pipeline_root=_mnst_pipeline_root,
        data_root=_mnst_data_root,
        enable_cache=_mnst_pipeline_cache
    )
    mnstlogger.info(f'Created tfrecord pipeline with id {pipeline.id}.')

    mnst_kubeflow_metadata_config = tfx.orchestration.experimental.get_default_kubeflow_metadata_config()
    mnst_kubeflow_metadata_config.grpc_config.grpc_service_host.value = "metadata-grpc-service.kubeflow"
    mnst_kubeflow_metadata_config.grpc_config.grpc_service_port.value = "8080"
    mnstlogger.info('Created kubeflow pipeline metadata config.')

    mnst_pipeline_config = tfx.orchestration.experimental.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=mnst_kubeflow_metadata_config,
        pipeline_operator_funcs=[
            gcp.use_gcp_secret('user-gcp-sa')
        ],
        tfx_image=f'gcr.io/tfx-oss-public/tfx:{tfx.__version__}')
    mnstlogger.info('Created kubeflow pipeline runner config.')

    mnst_pipeline_runner = tfx.orchestration.experimental.KubeflowDagRunner(
        config=mnst_pipeline_config,
        output_filename=f'{_mnst_pipeline_name}.yaml'
    )
    mnstlogger.info('Created kubeflow pipeline runner.')

    mnst_pipeline_runner.run(pipeline)
    mnstlogger.info(f'Created kubeflow pipeline runner deployment {_mnst_pipeline_name}.yaml')
