from google.cloud import storage  # type: ignore

import logging
import os
import pandas
import tempfile
import urllib.request as geturl


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def data_fetcher(name: str, source: str, dest: str):
    '''
    Fetches data from external URL and writes to pipeline data root. Supports GCS destinations.

    Args:
        name: A string specifying the pipeline name.
        source: A string specifying the URL of a public data source.
        dest: A string specifying filesystem or cloud storage destination.

    Returns:
        destination: A string specifying the location of the data.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''

    ''' Get a temporary directory with context manager.'''
    with tempfile.TemporaryDirectory(prefix=name) as _temp_root:

        ''' Name the destination file after the pipeline.'''
        if source.split('/')[-1].lower().split('.')[-1]:
            _data_filepath = os.path.join(_temp_root, '.'.join((name, source.split('/')[-1].lower().split('.')[-1])))
        else:
            _data_filepath = os.path.join(_temp_root, name)

        geturl.urlretrieve(source, _data_filepath)
        logger.info(f'Fetched data to file {_data_filepath}.')

        if dest.startswith('gs://'):
            cli = storage.Client()
            bucket = cli.bucket(dest.split('/')[2])
            blob = bucket.blob(os.path.join('/'.join(dest.split('/')[3:]), _data_filepath.split('/')[-1]))
            blob.upload_from_filename(_data_filepath)
            logger.info(f'Uploaded blob {blob.name} to bucket {bucket.name} from file {_data_filepath}.')
            location = os.path.join('gs://', f'{bucket.name}', f'{blob.name}')
        else:
            if not os.path.exists(dest):
                os.makedirs(dest)
            os.rename(_data_filepath, '/'.join((dest, _data_filepath.split('/')[-1])))
            location = '/'.join((dest, _data_filepath.split('/')[-1]))

        return location


def data_pusher(name: str, source: str, dest: str):
    '''
    Pushes data to pipeline data root. Supports GCS destinations.

    Args:
        name: A string specifying the pipeline name.
        source: A string specifying the data source.
        dest: A string specifying filesystem or cloud storage destination.

    Returns:
        destination: A string specifying the location of the data.

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''

    ''' Name the destination file after the pipeline.'''
    if source.split('/')[-1].lower().split('.')[-1]:
        _data_filepath = os.path.join('/'.join(source.split('/')[:-1]), '.'.join((name, source.split('/')[-1].lower().split('.')[-1])))
    else:
        _data_filepath = os.path.join('/'.join(source.split('/')[:-1]), name)

    if source != _data_filepath:
        os.rename(source, _data_filepath)

    if dest.startswith('gs://'):
        cli = storage.Client()
        bucket = cli.bucket(dest.split('/')[2])
        blob = bucket.blob(os.path.join('/'.join(dest.split('/')[3:]), _data_filepath.split('/')[-1]))
        ''' Fix chunk size for large(ish) files.'''
        if os.stat(_data_filepath).st_size > 20 * 1024 * 1024:
            blob.chunk_size = 5 * 1024 * 1024
        logger.info(f'Uploading blob {blob.name} to bucket {bucket.name} from file {_data_filepath}.')
        blob.upload_from_filename(_data_filepath)
        logger.info(f'Uploaded blob {blob.name} to bucket {bucket.name} from file {_data_filepath}.')
        location = os.path.join('gs://', f'{bucket.name}', f'{blob.name}')
    else:
        if not os.path.exists(dest):
            os.makedirs(dest)
        os.rename(_data_filepath, '/'.join((dest, _data_filepath.split('/')[-1])))
        location = '/'.join((dest, _data_filepath.split('/')[-1]))

    return location


def get_data_from_bucket(bucket_name: str, data_dir: str, data_type: str):
    ''' Parse the data directory on a bucket and return the content as a dataframe.'''

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=data_dir))
    paths = []
    dfs = []
    data = pandas.DataFrame()

    ''' Process the files locally, but clean up afterward.'''
    with tempfile.TemporaryDirectory(prefix='{}-data'.format(data_type)) as tempdir:
        logger.info('Found {} data files.'.format(len(blobs)))
        logger.info('Downloading data to temporary directory {}.'.format(tempdir))
        for blob in blobs:
            path = '/'.join(list([tempdir, blob.name.split('/')[-1].lower()]))
            blob.download_to_filename(path)
            logger.info('Downloaded blob {} to file {}.'.format(blob.name, path))
            paths.append(path)
        for file in paths:
            if data_type == 'csv':
                dfs.append(pandas.read_csv(file))
            else:
                logger.warning('Unsupported data type, {}.'.format(data_type))
        if len(dfs) > 0:
            data = pandas.concat(dfs)

    return data
