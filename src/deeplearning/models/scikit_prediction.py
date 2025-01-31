from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import joblib
import logging
import pickle


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
logger.warning('Loglevel for module {} set to {}'.format(__name__, logger.getEffectiveLevel()))


def get_model(model_type: str, **kwargs):
    ''' Select a model for training by algorithm and return an instance of it.'''

    match model_type:
        case 'rfclassifier':
            model = RandomForestClassifier(kwargs)
            logger.info('Configured RandomForestClassifier model.')
        case 'rfregressor':
            model = RandomForestRegressor(kwargs)
            logger.info('Configured RandomForestRegressor model.')
        case 'knregressor':
            model = KNeighborsRegressor(kwargs)
            logger.info('Configured KNeighborsRegressor model.')
        case _:
            logger.warning('Unsupported model_type {}.'.format(model_type))
            model = None

    return model


def save_model(model: any, name: str = 'model', format: str = 'joblib'):
    ''' Save a model.'''

    match format:
        case 'joblib':
            file = '{}.joblib'.format(name)
            joblib.dump(model, file)
            logger.info('Saved model to {}'.format(file))
            return file
        case 'pickle':
            file = '{}.pkl'.format(name)
            with open(file, 'wb') as f:
                pickle.dump(model, f)
                logger.info('Saved model to {}'.format(file))
            return file
        case _:
            logger.warning('Unsupported format {}.'.format(format))
            return


def upload_model(files: list, bucket_name: str, model_dir: str):
    ''' Upload a model to GCS.'''

    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)
    logger.info('Uploading files to bucket {}.'.format(bucket.name))
    blobs = []
    for file in files:
        blob_name = '/'.join([model_dir, file.split('/')[-1].lower()])
        blob = bucket.blob(blob_name)
        logger.info('Uploading to blob {}.'.format(blob.name))
        with open(file, 'rb') as f:
            blob.upload_from_file(f)
        blobs.append(blob.name)

    return blobs


def download_model():
    ''' Download a model from GCS.'''
    pass


def load_model():
    ''' Load a saved model.'''
    pass
