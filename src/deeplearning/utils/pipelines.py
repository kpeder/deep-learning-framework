from ml_metadata.proto import metadata_store_pb2  # type: ignore
from typing import Optional

import logging
import numpy  # type: ignore
import os
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

    Args:
        pipeline_name: A string specifying the pipeline's name.
        pipeline_root: A string specifying the pipeline's root path.
        data_root: A string specifying the path of the CSV data source.
        enable_cache: A bool indicating whether pipeline caching should be enabled.
        metadata_path: A optional string specifying the location of the metadata store, relative to the pipeline root.

    Returns:
        A tfx.dsl.Pipeline, configured for tfx.components.CSVExampleGen input.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
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

    Args:
        pipeline_name: A string specifying the pipeline's name.
        pipeline_root: A string specifying the pipeline's root path.
        data_root: A string specifying the path of the serialized (TF Record) data source.
        enable_cache: A bool indicating whether pipeline caching should be enabled.
        metadata_path: A optional string specifying the location of the metadata store, relative to the pipeline root.

    Returns:
        A tfx.dsl.Pipeline, configured for tfx.components.ImportExampleGen input.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
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

    Args:
        pipeline_name: A string specifying the pipeline's name.
        pipeline_root: A string specifying the pipeline's root path.
        components: A list of TFX components to add to the pipeline, can be empty.
        enable_cache: A bool indicating whether pipeline caching should be enabled.
        metadata_connection_config: A ConnectionConfig object specifying the metadata store connection.

    Returns:
        A tfx.dsl.Pipeline, configured with the components supplied in the components list.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
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
    '''
    Creates a tf.train.Feature from a bytes object.

    Args:
        value: A bytes object.

    Returns:
        A tf.train.Feature object containing a tf.train.BytesList.
    '''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    ''' Creates a tf.train.Feature from a float or double.

    Args:
        value: A number of type float.

    Returns:
        A tf.train.Feature object containing a tf.train.FloatList.
    '''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    '''
    Returns an tf.train.Feature from a bool, enum, int or uint.

    Args:
        value: A number of type int.

    Returns:
        A tf.train.Feature object containing a tf.train.Int64List.
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    '''
    Serialize a raw tensor.

    Args:
        array: A tensor (numpy.ndarray).

    Returns:
        array: A serialized array.
    '''
    array = tf.io.serialize_tensor(array)
    return array


def serialize_image_data(image: numpy.array, label: numpy.array):
    '''
    Serializes images and labels to tf.train.Example format.

    Args:
        image: An image tensor, as numpy.array of bytes.
        label: A categorical label, as a numpy.array of binary byte values.

    Returns:
        A tf.train.Example object containing a tf.train.Features record.
    '''
    record = {
        'height': _int64_feature(value=image.shape[0]),
        'width': _int64_feature(value=image.shape[1]),
        'depth': _int64_feature(value=image.shape[2]),
        'raw_image': _bytes_feature(serialize_array(image)),
        'label': _bytes_feature(serialize_array(label))
    }
    return tf.train.Example(features=tf.train.Features(feature=record))
