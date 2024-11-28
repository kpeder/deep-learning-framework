from google.cloud import storage  # type: ignore

import logging
import os
import tempfile
import urllib.request as geturl


logger = logging.getLogger(__name__)


def data_fetcher(name: str, source: str, dest: str):
    '''
    Fetches data from external URL and writes to pipeline data root. Supports GCS destinations.

    Args:
        name: A string specifying the pipeline name.
        source: A string specifying the URL of a public data source.
        dest: A string specifying filesystem or cloud storage destination.

    Returns:
        None

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

        else:
            if not os.path.exists(dest):
                os.makedirs(dest)
            os.rename(_data_filepath, '/'.join((dest, _data_filepath.split('/')[-1])))
