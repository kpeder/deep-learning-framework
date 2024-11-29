from deeplearning.utils.filesystem import data_fetcher, data_pusher

import logging
import os
import tempfile


logger = logging.getLogger(__name__)


def test_data_fetcher():
    '''
    Function to test the data_fetcher function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''

    with tempfile.TemporaryDirectory(prefix='snowcrabs-data-test') as temp_root:
        _pipeline_name = 'snowcrabs'
        _data_source_url = 'https://api-proxy.edh.azure.cloud.dfo-mpo.gc.ca/catalogue/records/7b278dfe-4fab-11ed-b9bb-1860247f53e3/attachments/Snow_crab_abundance_and_biomass_-_Abondance_et_biomasse_du_crabe.csv'
        _data_root = temp_root

        try:
            location = data_fetcher(name=_pipeline_name, source=_data_source_url, dest=_data_root)
            assert os.path.exists(os.path.join(temp_root, '.'.join((_pipeline_name, _data_source_url.split('.')[-1]))))
            assert location.startswith(_data_root)
        except Exception as e:
            logger.exception(e)
            raise e


def test_data_pusher():
    '''
    Function to test the data_pusher function.

    Args:
        None

    Returns:
        None

    Raises:
        e (Exception): Any unhandled exception, as necessary.
    '''

    with tempfile.TemporaryDirectory(prefix='snowcrabs-data-src') as temp_source:
        _pipeline_name = 'snowcrabs'
        _data_source = '/'.join((temp_source, 'snowcrabbes.txt'))

        _data_file = open(_data_source, 'w')
        _data_file.write('TEST\n')
        _data_file.close()

        with tempfile.TemporaryDirectory(prefix='snowcrabs-data-dest') as temp_dest:
            try:
                location = data_pusher(name=_pipeline_name, source=_data_source, dest=temp_dest)
                assert os.path.exists(location)
                with open(location, 'r') as _dest_file:
                    assert _dest_file.read() == 'TEST\n'
            except Exception as e:
                logger.exception(e)
                raise e
