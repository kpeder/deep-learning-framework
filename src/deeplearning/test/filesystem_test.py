from deeplearning.utils.filesystem import data_fetcher

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
            data_fetcher(name=_pipeline_name, source=_data_source_url, dest=_data_root)
            assert os.path.exists(os.path.join(temp_root, '.'.join((_pipeline_name, _data_source_url.split('.')[-1]))))
        except Exception as e:
            logger.exception(e)
            raise e
