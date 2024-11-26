from contextlib2 import contextmanager
from ml_metadata.proto import metadata_store_pb2  # type: ignore
from typing import Optional

import logging
import numpy  # type: ignore
import os
import tempfile as tmp
import tensorflow as tf  # type: ignore
import tfx.v1 as tfx  # type: ignore


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def create_csv_pipeline(pipeline_name: str,
                        pipeline_root: str,
                        data_root: str,
                        enable_cache: bool = False,
                        metadata_path: Optional[str] = None):
    '''
    Builds a TFX pipeline from imported CSV data.
    '''
    components: list = []

    if metadata_path is not None:
        metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path)
    else:
        metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(os.path.join(pipeline_root, 'metadata/metadata.db'))

    example_gen = tfx.components.CsvExampleGen(input_base=data_root)
    components.append(example_gen)

    try:
        return _create_pipeline(pipeline_name=pipeline_name,
                                pipeline_root=pipeline_root,
                                components=components,
                                enable_cache=enable_cache,
                                metadata_connection_config=metadata_connection_config)
    except Exception as e:
        logger.exception(e)


def create_tfr_pipeline(pipeline_name: str,
                        pipeline_root: str,
                        data_root: str,
                        enable_cache: bool = False,
                        metadata_path: Optional[str] = None):
    '''
    Builds a TFX pipeline from imported TF Records.
    '''
    components: list = []

    if metadata_path is not None:
        metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path)
    else:
        metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(os.path.join(pipeline_root, 'metadata/metadata.db'))

    example_gen = tfx.components.ImportExampleGen(input_base=data_root)
    components.append(example_gen)

    try:
        return _create_pipeline(pipeline_name=pipeline_name,
                                pipeline_root=pipeline_root,
                                components=components,
                                enable_cache=enable_cache,
                                metadata_connection_config=metadata_connection_config)
    except Exception as e:
        logger.exception(e)


def _create_pipeline(pipeline_name: str,
                     pipeline_root: str,
                     components: list,
                     enable_cache: bool,
                     metadata_connection_config: metadata_store_pb2.ConnectionConfig = None):
    '''
    Build a TFX pipeline.
    '''
    try:
        return tfx.dsl.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=components,
            enable_cache=enable_cache,
            metadata_connection_config=metadata_connection_config)
    except Exception as e:
        logger.exception(e)


def _bytes_feature(value):
    '''Returns a bytes_list from a string / byte.'''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    '''Returns a float_list from a float / double.'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    '''Returns an int64_list from a bool / enum / int / uint.'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


@contextmanager
def get_tmp_dir(prefix: str, clean: bool = True):
    ''' Creates a temporary directory and then cleans it up.'''

    directory = tmp.mkdtemp(prefix=prefix)
    try:
        logger.info(f'Created temporary directory {directory}.')
        yield directory
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        if clean:
            for file in os.listdir(directory):
                path = os.path.join(directory, file)
                if os.path.isfile(path):
                    os.remove(path)
            if len(os.listdir(directory)) == 0:
                os.rmdir(directory)
            else:
                logger.warning(f'Cloud not clean up non-empty directory {directory}.')


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def serialize_image_data(image: numpy.array, label: int):
    ''' Serializes images and labels to TF Record.'''
    record = {
        'height': _int64_feature(value=image.shape[0]),
        'width': _int64_feature(value=image.shape[1]),
        'depth': _int64_feature(value=image.shape[2]),
        'raw_image': _bytes_feature(serialize_array(image)),
        'label': _bytes_feature(serialize_array(label))
    }
    return tf.train.Example(features=tf.train.Features(feature=record))
